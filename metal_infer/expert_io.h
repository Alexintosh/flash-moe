// expert_io.h — Expert I/O infrastructure, thread pool, parallel pread, LRU cache, prefetch
// Part of Flash-MoE modular decomposition (unity build).
// Included by infer.m — do NOT compile separately.

// ============================================================================
// Expert callback infrastructure (distributed inference / flashswarm)
// ============================================================================

// Forward-declare callback types for unity build (full definition in FlashMoEEngine.h)
#ifndef FLASHMOE_ENGINE_H
typedef ssize_t (*expert_read_fn)(int layer, int expert, void *dst, size_t size,
                                  off_t offset, void *user_data);
typedef int (*expert_compute_fn)(int layer, int expert, const float *input,
                                 float *output, int hidden_dim, void *user_data);
#endif

// Global callback state -- wired from FlashMoEContext during load/set.
// These are checked in the expert I/O path to route remote experts.
static expert_read_fn g_expert_read_cb = NULL;
static void *g_expert_read_user_data = NULL;
static expert_compute_fn g_expert_compute_cb = NULL;
static void *g_expert_compute_user_data = NULL;
static const uint8_t *g_expert_remote_bitmap = NULL;
static int g_expert_remote_bitmap_size = 0;

// Check if an expert is marked as remote in the bitmap.
// Returns 1 if remote (should use callback), 0 if local (native pread).
// If bitmap is NULL and any callback is set, ALL experts are remote (legacy).
static inline int expert_is_remote(int layer, int expert) {
    if (!g_expert_read_cb && !g_expert_compute_cb) return 0;
    if (!g_expert_remote_bitmap) return 1;
    int bit_idx = layer * cfg.num_experts + expert;
    int byte_idx = bit_idx / 8;
    if (byte_idx >= g_expert_remote_bitmap_size) return 0;
    return (g_expert_remote_bitmap[byte_idx] >> (bit_idx % 8)) & 1;
}

// ============================================================================
// Parallel I/O infrastructure for expert pread (from proven main.m pattern)
// ============================================================================

#define NUM_IO_THREADS 8  // 8 threads for K=8 experts (one per expert)

typedef struct {
    int fd;
    void *dst;
    off_t offset;
    size_t size;
    ssize_t result;
    const void *mmap_base;  // if non-NULL, memcpy from mmap instead of pread
    // LZ4 compression fields (set by caller when reading compressed experts)
    void *lz4_comp_buf;     // if non-NULL: pread into this, then LZ4 decompress into dst
    uint32_t lz4_comp_size; // compressed size to read from disk
    // Expert identification (for callback routing in distributed inference)
    int layer_idx;
    int expert_idx;
} InferPreadTask;

typedef struct {
    InferPreadTask *tasks;
    int num_tasks;
    int thread_id;
} InferPreadThreadArg;

static void *infer_pread_thread_fn(void *arg) {
    InferPreadThreadArg *ta = (InferPreadThreadArg *)arg;
    for (int i = ta->thread_id; i < ta->num_tasks; i += NUM_IO_THREADS) {
        InferPreadTask *t = &ta->tasks[i];
        t->result = pread(t->fd, t->dst, t->size, t->offset);
    }
    return NULL;
}

// ============================================================================
// Persistent I/O Thread Pool — eliminates pthread_create/join per layer
// ============================================================================

typedef struct {
    pthread_t threads[NUM_IO_THREADS];
    pthread_mutex_t mutex;
    pthread_cond_t work_ready;
    pthread_cond_t work_done;
    InferPreadTask *tasks;
    int num_tasks;
    int tasks_completed;
    int generation;          // incremented each dispatch — workers wait for new gen
    int completed_generation;
    volatile int shutdown;
} IOThreadPool;

static IOThreadPool g_io_pool;
static int g_io_pool_initialized = 0;

static void *io_pool_worker(void *arg) {
    int tid = (int)(intptr_t)arg;
    int my_gen = 0;
    pthread_mutex_lock(&g_io_pool.mutex);
    while (1) {
        while (g_io_pool.generation == my_gen && !g_io_pool.shutdown)
            pthread_cond_wait(&g_io_pool.work_ready, &g_io_pool.mutex);
        if (g_io_pool.shutdown) break;
        my_gen = g_io_pool.generation;

        // Snapshot work for this generation
        int num_tasks = g_io_pool.num_tasks;
        InferPreadTask *tasks = g_io_pool.tasks;
        pthread_mutex_unlock(&g_io_pool.mutex);

        // Process assigned tasks (stride by thread count)
        for (int i = tid; i < num_tasks; i += NUM_IO_THREADS) {
            InferPreadTask *t = &tasks[i];
            if (g_expert_read_cb && expert_is_remote(t->layer_idx, t->expert_idx)) {
                t->result = g_expert_read_cb(t->layer_idx, t->expert_idx,
                                             t->dst, t->size, t->offset,
                                             g_expert_read_user_data);
            } else if (t->lz4_comp_buf && t->lz4_comp_size > 0) {
                // LZ4 path: read compressed from SSD, decompress into dst
                ssize_t nr = pread(t->fd, t->lz4_comp_buf, t->lz4_comp_size, t->offset);
                if (nr == (ssize_t)t->lz4_comp_size) {
                    size_t dec = compression_decode_buffer(
                        t->dst, t->size, t->lz4_comp_buf, t->lz4_comp_size,
                        NULL, COMPRESSION_LZ4);
                    t->result = (ssize_t)dec;
                } else {
                    t->result = -1;
                }
            } else {
                t->result = pread(t->fd, t->dst, t->size, t->offset);
            }
        }

        pthread_mutex_lock(&g_io_pool.mutex);
        g_io_pool.tasks_completed++;
        if (g_io_pool.tasks_completed == NUM_IO_THREADS) {
            g_io_pool.completed_generation = my_gen;
            pthread_cond_signal(&g_io_pool.work_done);
        }
    }
    pthread_mutex_unlock(&g_io_pool.mutex);
    return NULL;
}

static void io_pool_init(void) {
    if (g_io_pool_initialized) return;
    pthread_mutex_init(&g_io_pool.mutex, NULL);
    pthread_cond_init(&g_io_pool.work_ready, NULL);
    pthread_cond_init(&g_io_pool.work_done, NULL);
    g_io_pool.shutdown = 0;
    g_io_pool.generation = 0;
    g_io_pool.completed_generation = 0;
    g_io_pool.tasks = NULL;
    for (int i = 0; i < NUM_IO_THREADS; i++)
        pthread_create(&g_io_pool.threads[i], NULL, io_pool_worker, (void*)(intptr_t)i);
    g_io_pool_initialized = 1;
}

static dispatch_queue_t g_io_gcd_queue = NULL;

// Async start — returns generation number for later wait
static int io_pool_start(InferPreadTask *tasks, int num_tasks) {
    if (num_tasks == 0) return 0;
    pthread_mutex_lock(&g_io_pool.mutex);
    g_io_pool.tasks = tasks;
    g_io_pool.num_tasks = num_tasks;
    g_io_pool.tasks_completed = 0;
    g_io_pool.generation++;
    int gen = g_io_pool.generation;
    pthread_cond_broadcast(&g_io_pool.work_ready);
    pthread_mutex_unlock(&g_io_pool.mutex);
    return gen;
}

// Wait for a specific generation to complete
static void io_pool_wait_generation(int target_gen) {
    if (target_gen <= 0) return;
    pthread_mutex_lock(&g_io_pool.mutex);
    while (g_io_pool.completed_generation < target_gen) {
        pthread_cond_wait(&g_io_pool.work_done, &g_io_pool.mutex);
    }
    pthread_mutex_unlock(&g_io_pool.mutex);
}

// Synchronous dispatch — start + wait
static void io_pool_dispatch(InferPreadTask *tasks, int num_tasks) {
    if (num_tasks == 0) return;
    int my_gen = io_pool_start(tasks, num_tasks);
    io_pool_wait_generation(my_gen);
}

// ---- Async expert pread pipeline ----
// Uses GCD dispatch_group for truly async start+wait (no generation conflicts
// with the persistent io_pool). When cache_io_split > 1, each expert blob is
// split into N page-aligned chunks for parallel SSD reads.
#define MAX_CACHE_IO_SPLIT 8

static inline int active_cache_io_split(size_t esz) {
    int chunks = g_cache_io_split;
    if (chunks < 1) chunks = 1;
    if (chunks > MAX_CACHE_IO_SPLIT) chunks = MAX_CACHE_IO_SPLIT;

    // Expert blobs are page-cache-backed. Keep chunk boundaries page aligned
    // so fanout mode still matches the underlying VM layout.
    const size_t page_bytes = 16 * 1024;
    if (esz == 0 || (esz % page_bytes) != 0) return 1;

    size_t pages = esz / page_bytes;
    if ((size_t)chunks > pages) chunks = (int)pages;
    if (chunks < 1) chunks = 1;
    return chunks;
}

typedef struct {
    InferPreadTask tasks[MAX_K * MAX_CACHE_IO_SPLIT];
    int num_tasks;
    int num_experts;
    int chunks_per_expert;
    int valid[MAX_K];
    dispatch_group_t group;
    int active;
} AsyncPreadState;
static AsyncPreadState g_async_pread = {0};

static void async_pread_start(int packed_fd, int *expert_indices, int K,
                               id<MTLBuffer> __strong *dst_bufs, const void *mmap_base,
                               int layer_idx) {
    (void)mmap_base;
    size_t esz = active_expert_size();
    int chunks = active_cache_io_split(esz);
    const size_t page_bytes = 16 * 1024;

    g_async_pread.num_experts = K;
    g_async_pread.chunks_per_expert = chunks;
    g_async_pread.num_tasks = K * chunks;
    g_async_pread.active = 1;
    if (!g_async_pread.group) g_async_pread.group = dispatch_group_create();

    for (int k = 0; k < K; k++) {
        // Per-expert offset and size (tiered: variable, uniform: computed from index)
        size_t this_esz;
        off_t this_offset;
        if (g_use_tiered && g_tiered_manifest) {
            TieredExpertInfo *ti = &TIERED(layer_idx, expert_indices[k]);
            this_esz = ti->size;
            this_offset = (off_t)ti->offset;
        } else {
            this_esz = esz;
            this_offset = (off_t)expert_indices[k] * esz;
        }

        size_t total_pages = (chunks > 1) ? (this_esz / page_bytes) : 0;
        char *dst_base = (char *)[dst_bufs[k] contents];
        size_t page_cursor = 0;
        for (int c = 0; c < chunks; c++) {
            size_t chunk_off = 0;
            size_t chunk_sz = this_esz;
            if (chunks > 1) {
                size_t pages_this_chunk = total_pages / (size_t)chunks;
                if ((size_t)c < (total_pages % (size_t)chunks)) pages_this_chunk++;
                chunk_off = page_cursor * page_bytes;
                chunk_sz = pages_this_chunk * page_bytes;
                page_cursor += pages_this_chunk;
            }

            int task_idx = k * chunks + c;
            g_async_pread.tasks[task_idx].fd = packed_fd;
            g_async_pread.tasks[task_idx].dst = dst_base + chunk_off;
            g_async_pread.tasks[task_idx].offset = this_offset + (off_t)chunk_off;
            g_async_pread.tasks[task_idx].size = chunk_sz;
            g_async_pread.tasks[task_idx].result = 0;
            g_async_pread.tasks[task_idx].mmap_base = NULL;
            g_async_pread.tasks[task_idx].lz4_comp_buf = NULL;
            g_async_pread.tasks[task_idx].lz4_comp_size = 0;
            g_async_pread.tasks[task_idx].layer_idx = layer_idx;
            g_async_pread.tasks[task_idx].expert_idx = expert_indices[k];
        }
    }

    // Fire off parallel preads on GCD -- dispatch_group guarantees all blocks
    // complete before dispatch_group_wait returns (no generation counter race).
    // Remote experts (per bitmap) use the read callback instead of pread.
    static dispatch_queue_t io_q = NULL;
    if (!io_q) io_q = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);
    int total_tasks = g_async_pread.num_tasks;
    for (int i = 0; i < total_tasks; i++) {
        InferPreadTask *t = &g_async_pread.tasks[i];
        if (g_expert_read_cb && expert_is_remote(t->layer_idx, t->expert_idx)) {
            expert_read_fn cb = g_expert_read_cb;
            void *ud = g_expert_read_user_data;
            dispatch_group_async(g_async_pread.group, io_q, ^{
                t->result = cb(t->layer_idx, t->expert_idx, t->dst, t->size, t->offset, ud);
            });
        } else {
            dispatch_group_async(g_async_pread.group, io_q, ^{
                t->result = pread(t->fd, t->dst, t->size, t->offset);
            });
        }
    }
}

static void async_pread_wait(void) {
    if (!g_async_pread.active) return;
    dispatch_group_wait(g_async_pread.group, DISPATCH_TIME_FOREVER);
    // Validate each chunk against its OWN expected size (not uniform esz).
    // Critical for tiered mode where cold 2-bit experts are smaller than hot 4-bit.
    for (int k = 0; k < g_async_pread.num_experts; k++) {
        int ok = 1;
        for (int c = 0; c < g_async_pread.chunks_per_expert; c++) {
            int task_idx = k * g_async_pread.chunks_per_expert + c;
            if (g_async_pread.tasks[task_idx].result != (ssize_t)g_async_pread.tasks[task_idx].size) {
                ok = 0;
                break;
            }
        }
        g_async_pread.valid[k] = ok;
    }
    g_async_pread.active = 0;
}

static void io_pool_shutdown(void) {
    if (!g_io_pool_initialized) return;
    pthread_mutex_lock(&g_io_pool.mutex);
    g_io_pool.shutdown = 1;
    pthread_cond_broadcast(&g_io_pool.work_ready);
    pthread_mutex_unlock(&g_io_pool.mutex);
    for (int i = 0; i < NUM_IO_THREADS; i++)
        pthread_join(g_io_pool.threads[i], NULL);
    pthread_mutex_destroy(&g_io_pool.mutex);
    pthread_cond_destroy(&g_io_pool.work_ready);
    pthread_cond_destroy(&g_io_pool.work_done);
    g_io_pool_initialized = 0;
}

// Parallel pread of K experts into Metal buffers using pthreads.
// Returns number of successfully loaded experts, sets valid[] flags.
static int parallel_pread_experts(
    int packed_fd,
    int *expert_indices,
    int K,
    int *valid,  // [MAX_K] output: 1 if expert loaded successfully
    const void *mmap_base,  // mmap'd layer file (NULL to use pread)
    int layer_idx  // needed for tiered manifest lookup
) {
    size_t esz = active_expert_size();
    InferPreadTask tasks[MAX_K];
    for (int k = 0; k < K; k++) {
        size_t this_esz;
        off_t this_offset;
        if (g_use_tiered && g_tiered_manifest) {
            TieredExpertInfo *ti = &TIERED(layer_idx, expert_indices[k]);
            this_esz = ti->size;
            this_offset = (off_t)ti->offset;
        } else {
            this_esz = esz;
            this_offset = (off_t)expert_indices[k] * esz;
        }
        tasks[k].fd = packed_fd;
        tasks[k].dst = [g_metal->buf_multi_expert_data[k] contents];
        tasks[k].offset = this_offset;
        tasks[k].size = this_esz;
        tasks[k].result = 0;
        tasks[k].mmap_base = mmap_base;
        tasks[k].layer_idx = layer_idx;
        tasks[k].expert_idx = expert_indices[k];
    }

    io_pool_dispatch(tasks, K);

    int loaded = 0;
    for (int k = 0; k < K; k++) {
        valid[k] = (tasks[k].result == (ssize_t)tasks[k].size);
        if (valid[k]) loaded++;
        else {
            fprintf(stderr, "WARNING: expert %d pread: %zd/%zu\n",
                    expert_indices[k], tasks[k].result, tasks[k].size);
        }
    }
    return loaded;
}

// ============================================================================
// Parallel pread into explicit buffer set (for double buffering).
// Same as parallel_pread_experts but reads into caller-specified MTLBuffers.
// ============================================================================
static int parallel_pread_experts_into(
    int packed_fd,
    int *expert_indices,
    int K,
    id<MTLBuffer> __strong *dst_bufs,  // target Metal buffers (set A or B)
    int *valid,  // [MAX_K] output: 1 if expert loaded successfully
    int layer_idx  // needed for tiered manifest lookup
) {
    size_t esz = active_expert_size();
    InferPreadTask tasks[MAX_K];
    for (int k = 0; k < K; k++) {
        size_t this_esz;
        off_t this_offset;
        if (g_use_tiered && g_tiered_manifest) {
            TieredExpertInfo *ti = &TIERED(layer_idx, expert_indices[k]);
            this_esz = ti->size;
            this_offset = (off_t)ti->offset;
        } else {
            this_esz = esz;
            this_offset = (off_t)expert_indices[k] * esz;
        }
        tasks[k].fd = packed_fd;
        tasks[k].dst = [dst_bufs[k] contents];
        tasks[k].offset = this_offset;
        tasks[k].size = this_esz;
        tasks[k].result = 0;
        tasks[k].layer_idx = layer_idx;
        tasks[k].expert_idx = expert_indices[k];
    }

    io_pool_dispatch(tasks, K);

    int loaded = 0;
    for (int k = 0; k < K; k++) {
        valid[k] = (tasks[k].result == (ssize_t)tasks[k].size);
        if (valid[k]) loaded++;
        else {
            fprintf(stderr, "WARNING: expert %d pread: %zd/%zu\n",
                    expert_indices[k], tasks[k].result, tasks[k].size);
        }
    }
    return loaded;
}

// ============================================================================
// Expert LRU Cache: keeps recently-used expert Metal buffers in GPU memory.
//
// Key: (layer_idx, expert_idx) -> Metal buffer containing 7.08MB expert data.
// On cache HIT:  skip pread entirely, use the cached Metal buffer for GPU dispatch.
// On cache MISS: pread into a new/evicted Metal buffer, insert into cache.
// LRU eviction:  when cache is full, evict the least recently used entry.
//
// Memory budget: 2000 entries * 7.08MB = 14.2GB. With 5.5GB non-expert weights
// + 14.2GB cache = 19.7GB total. Fits in 48GB with room for OS.
//
// Unlike Python/MLX where LRU caching caused Metal heap pressure and slower
// mx.eval(), here Metal buffers ARE the cache -- no conversion overhead.
// ============================================================================

typedef struct {
    int layer_idx;
    int expert_idx;
    id<MTLBuffer> buffer;    // Metal buffer holding cfg.expert_size_4bit bytes
    uint64_t last_used;      // monotonic counter for LRU ordering
} ExpertCacheEntry;

typedef struct {
    ExpertCacheEntry *entries;
    int max_entries;
    int num_entries;
    int used_entries;
    int *entry_idx;  // flattened [num_layers * num_experts], -1 = not cached
    uint64_t access_counter; // monotonic, incremented on every access
    id<MTLDevice> device;    // for allocating new Metal buffers
    // Stats
    uint64_t hits;
    uint64_t misses;
} ExpertLRUCache;

static ExpertLRUCache *g_expert_cache = NULL;

// Speculative early routing stats
static uint64_t g_spec_route_attempts = 0;   // total speculative routing attempts
static uint64_t g_spec_route_hits = 0;        // correctly predicted experts (found in cache at real routing time)
static uint64_t g_spec_route_preloads = 0;    // async preloads initiated (cache misses at speculation time)

// ---- Temporal prediction pipeline ----
// Stores previous token's expert routing per layer. On the next token,
// predicted experts are preloaded into buf_multi_expert_data_B during CMD1_wait
// idle time. After routing, hits use buf_B, misses sync-pread into buf_A.
// Different from previous failed speculative attempts:
//   - Loads into scratch buffers (no cache pollution)
//   - Uses CMD1_wait idle time (no additional CPU cost)
//   - Only sync-preads misses (not all K experts)
static int g_pred_valid = 0;                       // 1 after first token completes (predictions available)
// g_pred_enabled, g_pred_hits, g_pred_misses, g_pred_layers declared near timing (line ~163)

static ExpertLRUCache *expert_cache_new(id<MTLDevice> device, int max_entries) {
    ExpertLRUCache *cache = calloc(1, sizeof(ExpertLRUCache));
    cache->entries = calloc(max_entries, sizeof(ExpertCacheEntry));
    cache->entry_idx = malloc(cfg.num_layers * cfg.num_experts * sizeof(int));
    cache->max_entries = max_entries;
    cache->num_entries = 0;
    cache->used_entries = 0;
    cache->access_counter = 0;
    cache->device = device;
    cache->hits = 0;
    cache->misses = 0;
    for (int l = 0; l < cfg.num_layers; l++) {
        for (int e = 0; e < cfg.num_experts; e++) {
            cache->entry_idx[(l) * cfg.num_experts + (e)] = -1;
        }
    }
    // Pre-allocate ALL Metal buffers at startup (avoids allocation overhead at runtime)
    size_t esz = active_expert_size();
    double t_prealloc = now_ms();
    for (int i = 0; i < max_entries; i++) {
        cache->entries[i].buffer = [device newBufferWithLength:esz
                                                      options:MTLResourceStorageModeShared];
        cache->entries[i].layer_idx = -1;
        cache->entries[i].expert_idx = -1;
        cache->entries[i].last_used = 0;
        if (!cache->entries[i].buffer) {
            fprintf(stderr, "WARNING: expert_cache: pre-alloc failed at entry %d\n", i);
            max_entries = i;
            cache->max_entries = i;
            break;
        }
    }
    cache->num_entries = max_entries; // All slots pre-allocated (but empty keys)
    printf("[expert_cache] Initialized: max_entries=%d (%.1f GB budget), pre-alloc %.0f ms\n",
           max_entries, (double)max_entries * esz / 1e9, now_ms() - t_prealloc);
    return cache;
}

static void expert_cache_free(ExpertLRUCache *cache) {
    if (!cache) return;
    printf("[expert_cache] Final stats: %llu hits, %llu misses (%.1f%% hit rate)\n",
           cache->hits, cache->misses,
           (cache->hits + cache->misses) > 0
               ? 100.0 * cache->hits / (cache->hits + cache->misses) : 0.0);
    // Metal buffers released by ARC when entries are freed
    free(cache->entries);
    free(cache);
}

// Lookup: returns the cached Metal buffer if found, otherwise NULL.
// On hit, updates the LRU timestamp.
static id<MTLBuffer> expert_cache_lookup(ExpertLRUCache *cache, int layer_idx, int expert_idx) {
    int idx = cache->entry_idx[(layer_idx) * cfg.num_experts + (expert_idx)];
    if (idx >= 0) {
        cache->entries[idx].last_used = ++cache->access_counter;
        cache->hits++;
        cache_telemetry_touch(layer_idx, expert_idx);
        return cache->entries[idx].buffer;
    }
    cache->misses++;
    cache_telemetry_miss(layer_idx, expert_idx);
    return nil;
}

// Insert: adds a new entry. If the cache is full, evicts the LRU entry.
// Returns the Metal buffer to pread into (either newly allocated or evicted+reused).
static id<MTLBuffer> expert_cache_insert(ExpertLRUCache *cache, int layer_idx, int expert_idx) {
    id<MTLBuffer> buf = nil;

    int existing = cache->entry_idx[(layer_idx) * cfg.num_experts + (expert_idx)];
    if (existing >= 0) {
        cache->entries[existing].last_used = ++cache->access_counter;
        return cache->entries[existing].buffer;
    }

    // Find a slot: first try an unused slot (layer_idx == -1), then LRU evict
    int target = -1;
    if (cache->used_entries < cache->num_entries) {
        target = cache->used_entries++;
    }
    if (target >= 0) {
        // Unused pre-allocated slot
        buf = cache->entries[target].buffer;
        cache->entries[target].layer_idx = layer_idx;
        cache->entries[target].expert_idx = expert_idx;
        cache->entries[target].last_used = ++cache->access_counter;
        cache->entry_idx[(layer_idx) * cfg.num_experts + (expert_idx)] = target;
        return buf;
    }

    // Cache full: find LRU entry (smallest last_used)
    int lru_idx = 0;
    uint64_t min_used = cache->entries[0].last_used;
    for (int i = 1; i < cache->num_entries; i++) {
        if (cache->entries[i].last_used < min_used) {
            min_used = cache->entries[i].last_used;
            lru_idx = i;
        }
    }

    // Reuse the evicted entry's Metal buffer (same size, no realloc needed)
    int old_layer = cache->entries[lru_idx].layer_idx;
    int old_expert = cache->entries[lru_idx].expert_idx;
    cache_telemetry_evict(old_layer, old_expert);
    if (old_layer >= 0 && old_expert >= 0) {
        cache->entry_idx[(old_layer) * cfg.num_experts + (old_expert)] = -1;
    }
    buf = cache->entries[lru_idx].buffer;
    cache->entries[lru_idx].layer_idx = layer_idx;
    cache->entries[lru_idx].expert_idx = expert_idx;
    cache->entries[lru_idx].last_used = ++cache->access_counter;
    cache->entry_idx[(layer_idx) * cfg.num_experts + (expert_idx)] = lru_idx;
    return buf;
}

// ============================================================================
// Malloc-based expert frequency cache.
// Stores expert data in regular malloc'd memory (not Metal buffers) to avoid
// GPU memory pressure. On hit, memcpy to Metal scratch buffer. Much larger
// capacity than Metal buffer LRU cache at the cost of one memcpy per hit.
// ============================================================================

typedef struct {
    void **data;           // [max_entries] page-aligned malloc'd cfg.expert_size_4bit buffers
    id<MTLBuffer> __strong *metal_bufs;  // [max_entries] zero-copy Metal buffer wrappers
    int *layer_idx;        // [max_entries] layer index for each entry
    int *expert_idx;       // [max_entries] expert index for each entry
    uint64_t *last_used;   // [max_entries] monotonic counter for LRU
    int max_entries;
    int num_entries;
    int used_entries;
    int *entry_idx;  // flattened [num_layers * num_experts], -1 = not cached
    uint64_t access_counter;
    uint64_t hits;
    uint64_t misses;
} MallocExpertCache;

static MallocExpertCache *g_malloc_cache = NULL;

static MallocExpertCache *malloc_cache_init(int max_entries, id<MTLDevice> device) {
    MallocExpertCache *cache = calloc(1, sizeof(MallocExpertCache));
    cache->data = calloc(max_entries, sizeof(void *));
    cache->metal_bufs = (__strong id<MTLBuffer> *)calloc(max_entries, sizeof(id<MTLBuffer>));
    cache->layer_idx = calloc(max_entries, sizeof(int));
    cache->expert_idx = calloc(max_entries, sizeof(int));
    cache->last_used = calloc(max_entries, sizeof(uint64_t));
    cache->entry_idx = malloc(cfg.num_layers * cfg.num_experts * sizeof(int));
    cache->max_entries = max_entries;
    cache->num_entries = 0;
    cache->used_entries = 0;
    cache->access_counter = 0;
    cache->hits = 0;
    cache->misses = 0;
    for (int l = 0; l < cfg.num_layers; l++) {
        for (int e = 0; e < cfg.num_experts; e++) {
            cache->entry_idx[(l) * cfg.num_experts + (e)] = -1;
        }
    }

    size_t esz = active_expert_size();
    printf("[malloc_cache] Initializing: %d entries (%.1f GB) with zero-copy Metal wrappers\n",
           max_entries, (double)max_entries * esz / 1e9);
    double t_start = now_ms();

    size_t page_size = (size_t)getpagesize();
    // Round expert size up to page boundary for newBufferWithBytesNoCopy
    size_t aligned_size = (esz + page_size - 1) & ~(page_size - 1);

    for (int i = 0; i < max_entries; i++) {
        // Page-aligned allocation for zero-copy Metal buffer
        void *buf = NULL;
        if (posix_memalign(&buf, page_size, aligned_size) != 0 || !buf) {
            fprintf(stderr, "WARNING: malloc_cache: alloc failed at entry %d\n", i);
            max_entries = i;
            cache->max_entries = i;
            break;
        }
        memset(buf, 0, aligned_size);
        cache->data[i] = buf;

        // Create zero-copy Metal buffer wrapping the malloc'd memory
        // nil deallocator = Metal doesn't free the memory
        cache->metal_bufs[i] = [device newBufferWithBytesNoCopy:buf
                                                         length:aligned_size
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
        cache->layer_idx[i] = -1;
        cache->expert_idx[i] = -1;
        cache->last_used[i] = 0;
    }
    cache->num_entries = max_entries;

    printf("[malloc_cache] Pre-allocated %d entries in %.0f ms\n",
           max_entries, now_ms() - t_start);
    return cache;
}

// Lookup: returns Metal buffer wrapping cached data, or nil. Zero-copy dispatch.
static id<MTLBuffer> malloc_cache_lookup(MallocExpertCache *cache, int layer, int expert) {
    int idx = cache->entry_idx[(layer) * cfg.num_experts + (expert)];
    if (idx >= 0) {
        cache->last_used[idx] = ++cache->access_counter;
        cache->hits++;
        cache_telemetry_touch(layer, expert);
        return cache->metal_bufs[idx];
    }
    cache->misses++;
    cache_telemetry_miss(layer, expert);
    return nil;
}

// Insert: evict LRU if needed, return entry index for pread target.
// Returns the Metal buffer for this entry (caller should pread into cache->data[idx]).
static id<MTLBuffer> malloc_cache_insert(MallocExpertCache *cache, int layer, int expert, int *out_idx) {
    int existing = cache->entry_idx[(layer) * cfg.num_experts + (expert)];
    if (existing >= 0) {
        cache->last_used[existing] = ++cache->access_counter;
        if (out_idx) *out_idx = existing;
        return cache->metal_bufs[existing];
    }

    // Find a free slot (layer_idx == -1) or evict LRU
    int target = -1;
    if (cache->used_entries < cache->num_entries) {
        target = cache->used_entries++;
    }

    if (target < 0) {
        // Cache full: evict entry with smallest last_used
        target = 0;
        uint64_t min_used = cache->last_used[0];
        for (int i = 1; i < cache->num_entries; i++) {
            if (cache->last_used[i] < min_used) {
                min_used = cache->last_used[i];
                target = i;
            }
        }
        cache_telemetry_evict(cache->layer_idx[target], cache->expert_idx[target]);
        if (cache->layer_idx[target] >= 0 && cache->expert_idx[target] >= 0) {
            cache->entry_idx[(cache->layer_idx[target]) * cfg.num_experts + (cache->expert_idx[target])] = -1;
        }
    }

    cache->layer_idx[target] = layer;
    cache->expert_idx[target] = expert;
    cache->last_used[target] = ++cache->access_counter;
    cache->entry_idx[(layer) * cfg.num_experts + (expert)] = target;
    if (out_idx) *out_idx = target;
    return cache->metal_bufs[target];
}

static void malloc_cache_free(MallocExpertCache *cache) {
    if (!cache) return;
    printf("[malloc_cache] Final stats: %llu hits, %llu misses (%.1f%% hit rate)\n",
           cache->hits, cache->misses,
           (cache->hits + cache->misses) > 0
               ? 100.0 * cache->hits / (cache->hits + cache->misses) : 0.0);
    for (int i = 0; i < cache->num_entries; i++) {
        cache->metal_bufs[i] = nil;  // release Metal buffer wrapper
        free(cache->data[i]);
    }
    free(cache->data);
    free(cache->metal_bufs);
    free(cache->layer_idx);
    free(cache->expert_idx);
    free(cache->last_used);
    free(cache);
}

// ============================================================================
// Background prefetch thread for double-buffered expert I/O (from main.m).
// Runs pread on a background thread while main thread does GPU compute.
// Uses pure C I/O plan to avoid ARC issues across threads.
// ============================================================================

typedef struct {
    void *dst[MAX_K];       // raw pointers from [buf contents] (no ARC)
    off_t offset[MAX_K];    // file offsets per expert
    size_t size[MAX_K];     // bytes to read per expert (may vary in tiered mode)
    int K;                  // number of experts
    int fd;                 // file descriptor for this layer
    int valid[MAX_K];       // output: 1 if pread succeeded
    int loaded;             // output: count of successfully loaded experts
} InferIOPlan;

typedef struct {
    InferIOPlan plan;       // pre-built I/O plan (pure C, no ARC)
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int start;              // signal: set to 1 to start prefetch
    int done;               // signal: set to 1 when prefetch complete
    int shutdown;           // signal: set to 1 to exit thread
} InferPrefetchCtx;

static void *infer_prefetch_thread_fn(void *arg) {
    InferPrefetchCtx *pf = (InferPrefetchCtx *)arg;

    while (1) {
        pthread_mutex_lock(&pf->mutex);
        while (!pf->start && !pf->shutdown) {
            pthread_cond_wait(&pf->cond, &pf->mutex);
        }
        if (pf->shutdown) {
            pthread_mutex_unlock(&pf->mutex);
            break;
        }
        pf->start = 0;
        pthread_mutex_unlock(&pf->mutex);

        // Execute parallel pread (pure C, no ARC objects)
        InferIOPlan *plan = &pf->plan;
        InferPreadTask tasks[MAX_K];
        for (int k = 0; k < plan->K; k++) {
            tasks[k].fd = plan->fd;
            tasks[k].dst = plan->dst[k];
            tasks[k].offset = plan->offset[k];
            tasks[k].size = plan->size[k];
            tasks[k].result = 0;
        }

        io_pool_dispatch(tasks, plan->K);

        plan->loaded = 0;
        for (int k = 0; k < plan->K; k++) {
            plan->valid[k] = (tasks[k].result == (ssize_t)plan->size[k]);
            if (plan->valid[k]) plan->loaded++;
        }

        // Signal completion
        pthread_mutex_lock(&pf->mutex);
        pf->done = 1;
        pthread_cond_signal(&pf->cond);
        pthread_mutex_unlock(&pf->mutex);
    }

    return NULL;
}

// Build I/O plan on main thread (ARC-safe: extracts void* from id<MTLBuffer>),
// then signal background prefetch thread.
static void infer_prefetch_start(InferPrefetchCtx *pf, int packed_fd,
                                  int *expert_indices, int K,
                                  id<MTLBuffer> __strong *dst_bufs,
                                  int layer_idx) {
    pthread_mutex_lock(&pf->mutex);
    InferIOPlan *plan = &pf->plan;
    plan->fd = packed_fd;
    plan->K = K;
    for (int k = 0; k < K; k++) {
        off_t eoff; size_t esz;
        expert_offset_size(layer_idx, expert_indices[k], &eoff, &esz);
        plan->dst[k] = [dst_bufs[k] contents];
        plan->offset[k] = eoff;
        plan->size[k] = esz;
        plan->valid[k] = 0;
    }
    plan->loaded = 0;
    pf->done = 0;
    pf->start = 1;
    pthread_cond_signal(&pf->cond);
    pthread_mutex_unlock(&pf->mutex);
}

// Wait for background prefetch to complete. Returns number of loaded experts.
// Copies valid[] flags into caller's array.
static int infer_prefetch_wait(InferPrefetchCtx *pf, int *valid_out, int K) {
    pthread_mutex_lock(&pf->mutex);
    while (!pf->done) {
        pthread_cond_wait(&pf->cond, &pf->mutex);
    }
    int loaded = pf->plan.loaded;
    for (int k = 0; k < K; k++) {
        valid_out[k] = pf->plan.valid[k];
    }
    pthread_mutex_unlock(&pf->mutex);
    return loaded;
}

static InferPrefetchCtx *g_prefetch = NULL;
static pthread_t g_prefetch_tid;

static void infer_prefetch_init(void) {
    if (g_prefetch) return;
    g_prefetch = calloc(1, sizeof(InferPrefetchCtx));
    pthread_mutex_init(&g_prefetch->mutex, NULL);
    pthread_cond_init(&g_prefetch->cond, NULL);
    g_prefetch->shutdown = 0;
    pthread_create(&g_prefetch_tid, NULL, infer_prefetch_thread_fn, g_prefetch);
}

static void infer_prefetch_shutdown(void) {
    if (!g_prefetch) return;
    pthread_mutex_lock(&g_prefetch->mutex);
    g_prefetch->shutdown = 1;
    pthread_cond_signal(&g_prefetch->cond);
    pthread_mutex_unlock(&g_prefetch->mutex);
    pthread_join(g_prefetch_tid, NULL);
    pthread_mutex_destroy(&g_prefetch->mutex);
    pthread_cond_destroy(&g_prefetch->cond);
    free(g_prefetch);
    g_prefetch = NULL;
}
