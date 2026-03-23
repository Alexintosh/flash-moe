// weights.h — bf16 conversion, tensor manifest, hash table, WeightFile mmap
// Part of Flash-MoE modular decomposition (unity build).
// Included by infer.m — do NOT compile separately.

// ============================================================================
// bf16 <-> f32 conversion (CPU side)
// ============================================================================

static float bf16_to_f32(uint16_t bf16) {
    uint32_t bits = (uint32_t)bf16 << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

__attribute__((unused))
static uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    return (uint16_t)(bits >> 16);
}

// ============================================================================
// JSON parser (minimal, for model_weights.json)
// ============================================================================

// We use NSJSONSerialization via ObjC since we already link Foundation

typedef struct {
    const char *name;
    size_t offset;
    size_t size;
    int ndim;
    int shape[4];
    char dtype[8];  // "U32", "BF16", "F32"
} TensorInfo;

typedef struct {
    TensorInfo *tensors;
    int num_tensors;
    int capacity;
} TensorManifest;

static TensorManifest *load_manifest(const char *json_path) {
    @autoreleasepool {
        NSData *data = [NSData dataWithContentsOfFile:
            [NSString stringWithUTF8String:json_path]];
        if (!data) {
            fprintf(stderr, "ERROR: Cannot read %s\n", json_path);
            return NULL;
        }

        NSError *error = nil;
        NSDictionary *root = [NSJSONSerialization JSONObjectWithData:data
                                                             options:0
                                                               error:&error];
        if (!root) {
            fprintf(stderr, "ERROR: JSON parse failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            return NULL;
        }

        NSDictionary *tensors = root[@"tensors"];
        if (!tensors) {
            fprintf(stderr, "ERROR: No 'tensors' key in manifest\n");
            return NULL;
        }

        TensorManifest *m = calloc(1, sizeof(TensorManifest));
        m->capacity = (int)[tensors count] + 16;
        m->tensors = calloc(m->capacity, sizeof(TensorInfo));
        m->num_tensors = 0;

        for (NSString *key in tensors) {
            NSDictionary *info = tensors[key];
            TensorInfo *t = &m->tensors[m->num_tensors];

            const char *name = [key UTF8String];
            t->name = strdup(name);
            t->offset = [info[@"offset"] unsignedLongLongValue];
            t->size = [info[@"size"] unsignedLongLongValue];

            NSArray *shape = info[@"shape"];
            t->ndim = (int)[shape count];
            for (int i = 0; i < t->ndim && i < 4; i++) {
                t->shape[i] = [shape[i] intValue];
            }

            const char *dtype = [info[@"dtype"] UTF8String];
            strncpy(t->dtype, dtype, 7);

            m->num_tensors++;
        }

        printf("[manifest] Loaded %d tensors from %s\n", m->num_tensors, json_path);
        return m;
    }
}

// Hash table for O(1) tensor lookup (replaces O(N) linear scan).
// FNV-1a hash, open addressing with linear probing.
#define TENSOR_HT_SIZE 8192  // power of 2, > 4x num_tensors (2092)

typedef struct {
    const char *key;     // tensor name (pointer into TensorInfo)
    TensorInfo *value;   // pointer to tensor info
} TensorHTEntry;

static TensorHTEntry tensor_ht[TENSOR_HT_SIZE];
static int tensor_ht_built = 0;

static uint32_t fnv1a(const char *s) {
    uint32_t h = 2166136261u;
    for (; *s; s++) {
        h ^= (uint8_t)*s;
        h *= 16777619u;
    }
    return h;
}

static void build_tensor_ht(TensorManifest *m) {
    if (tensor_ht_built) return;
    memset(tensor_ht, 0, sizeof(tensor_ht));
    for (int i = 0; i < m->num_tensors; i++) {
        uint32_t idx = fnv1a(m->tensors[i].name) & (TENSOR_HT_SIZE - 1);
        while (tensor_ht[idx].key) {
            idx = (idx + 1) & (TENSOR_HT_SIZE - 1);
        }
        tensor_ht[idx].key = m->tensors[i].name;
        tensor_ht[idx].value = &m->tensors[i];
    }
    tensor_ht_built = 1;
}

static TensorInfo *find_tensor(TensorManifest *m, const char *name) {
    if (!tensor_ht_built) build_tensor_ht(m);
    uint32_t idx = fnv1a(name) & (TENSOR_HT_SIZE - 1);
    while (tensor_ht[idx].key) {
        if (strcmp(tensor_ht[idx].key, name) == 0) {
            return tensor_ht[idx].value;
        }
        idx = (idx + 1) & (TENSOR_HT_SIZE - 1);
    }
    return NULL;
}

// ============================================================================
// Weight file: mmap'd binary blob
// ============================================================================

typedef struct {
    void *data;          // mmap base (or part 0 base for split files)
    size_t size;         // total logical size (sum of both parts for split)
    TensorManifest *manifest;
    // Split weight support: two independently mmap'd regions
    int is_split;
    void *split_data[2]; // [0] = part 0 base, [1] = part 1 base
    size_t split_size[2]; // sizes of each part
} WeightFile;

// Try to mmap a split weight file (model_weights_0.bin + model_weights_1.bin)
// into a single contiguous virtual region so tensor offsets work unchanged.
static int open_split_weights(const char *dir_path, void **out_data, size_t *out_size,
                               void **out_mmap0, size_t *out_size0,
                               void **out_mmap1, size_t *out_size1) {
    char path0[1024], path1[1024];
    snprintf(path0, sizeof(path0), "%s/model_weights_0.bin", dir_path);
    snprintf(path1, sizeof(path1), "%s/model_weights_1.bin", dir_path);

    int fd0 = open(path0, O_RDONLY);
    int fd1 = open(path1, O_RDONLY);
    if (fd0 < 0 || fd1 < 0) {
        if (fd0 >= 0) close(fd0);
        if (fd1 >= 0) close(fd1);
        return -1;
    }

    struct stat st0, st1;
    fstat(fd0, &st0); fstat(fd1, &st1);
    size_t size0 = st0.st_size, size1 = st1.st_size;
    size_t total = size0 + size1;

    // mmap each file independently (MAP_FIXED not available on iOS)
    void *m0 = mmap(NULL, size0, PROT_READ, MAP_PRIVATE, fd0, 0);
    close(fd0);
    if (m0 == MAP_FAILED) {
        fprintf(stderr, "ERROR: Split weight mmap part 0 failed: %s\n", strerror(errno));
        close(fd1);
        return -1;
    }

    void *m1 = mmap(NULL, size1, PROT_READ, MAP_PRIVATE, fd1, 0);
    close(fd1);
    if (m1 == MAP_FAILED) {
        fprintf(stderr, "ERROR: Split weight mmap part 1 failed: %s\n", strerror(errno));
        munmap(m0, size0);
        return -1;
    }

    madvise(m0, size0, MADV_SEQUENTIAL);
    madvise(m1, size1, MADV_SEQUENTIAL);
    printf("[weights] mmap'd split weights: %.2f GB + %.2f GB = %.2f GB\n",
           size0 / 1e9, size1 / 1e9, total / 1e9);

    // Use m0 as the base pointer. Tensors in part 1 (offset >= size0) will
    // need special handling in get_tensor_ptr via the split metadata.
    *out_data = m0;
    *out_size = total;
    *out_mmap0 = m0; *out_size0 = size0;
    *out_mmap1 = m1; *out_size1 = size1;
    return 0;
}

static WeightFile *open_weights(const char *bin_path, const char *json_path) {
    void *data = NULL;
    size_t size = 0;
    int is_split = 0;
    void *split_mmap0 = NULL, *split_mmap1 = NULL;
    size_t split_size0 = 0, split_size1 = 0;

    // Try single file first
    int fd = open(bin_path, O_RDONLY);
    if (fd >= 0) {
        struct stat st;
        fstat(fd, &st);
        size = st.st_size;
        data = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (data == MAP_FAILED) {
            fprintf(stderr, "ERROR: mmap failed: %s\n", strerror(errno));
            data = NULL;
        } else {
            madvise(data, size, MADV_SEQUENTIAL);
            printf("[weights] mmap'd %.2f GB from %s\n", size / 1e9, bin_path);
        }
    }

    // If single file failed, try split weights (model_weights_0.bin + _1.bin)
    if (!data) {
        // Extract directory from bin_path
        char dir_path[1024];
        strlcpy(dir_path, bin_path, sizeof(dir_path));
        char *last_slash = strrchr(dir_path, '/');
        if (last_slash) *last_slash = '\0';

        if (open_split_weights(dir_path, &data, &size,
                               &split_mmap0, &split_size0,
                               &split_mmap1, &split_size1) == 0) {
            is_split = 1;
        }
    }

    if (!data) {
        fprintf(stderr, "ERROR: Cannot open weights from %s (also tried split files)\n", bin_path);
        return NULL;
    }

    TensorManifest *manifest = load_manifest(json_path);
    if (!manifest) {
        if (is_split) munmap(data, size);
        else munmap(data, size);
        return NULL;
    }

    WeightFile *wf = calloc(1, sizeof(WeightFile));
    wf->data = data;
    wf->size = size;
    wf->manifest = manifest;
    wf->is_split = is_split;
    if (is_split) {
        wf->split_data[0] = split_mmap0;
        wf->split_size[0] = split_size0;
        wf->split_data[1] = split_mmap1;
        wf->split_size[1] = split_size1;
    }

    return wf;
}

static void *get_tensor_ptr(WeightFile *wf, const char *name) {
    TensorInfo *t = find_tensor(wf->manifest, name);
    if (!t) {
        fprintf(stderr, "WARNING: tensor '%s' not found\n", name);
        return NULL;
    }
    if (wf->is_split && t->offset >= wf->split_size[0]) {
        // Tensor lives in part 1
        return (char *)wf->split_data[1] + (t->offset - wf->split_size[0]);
    }
    return (char *)wf->data + t->offset;
}

static TensorInfo *get_tensor_info(WeightFile *wf, const char *name) {
    return find_tensor(wf->manifest, name);
}
