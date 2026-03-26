/*
 * ModelListView.swift — Model discovery and loading
 *
 * Lists locally available models and allows downloading from HuggingFace.
 * For v1, supports loading models already present on device.
 */

import SwiftUI
import UniformTypeIdentifiers
#if os(macOS)
import AppKit
#endif

// MARK: - Local Model Entry

struct LocalModel: Identifiable {
    let id = UUID()
    let name: String
    let path: String
    let sizeBytes: UInt64
    let hasTiered: Bool
    let has4bit: Bool
    let has2bit: Bool

    var sizeMB: Double { Double(sizeBytes) / 1_048_576 }
    var sizeGB: Double { sizeMB / 1024 }
}

// MARK: - Setting Info

struct SettingInfo: Identifiable {
    let id: String
    let title: String
    let analogy: String
    let technical: String

    static let all: [String: SettingInfo] = [
        "activeExperts": SettingInfo(
            id: "activeExperts",
            title: "Active Experts (K)",
            analogy: "Imagine asking a question to a room of 256 specialists. K controls how many you consult. K=8 means you ask 8 experts and combine their answers. K=4 means you only ask 4 — faster (less reading from disk) but you might miss a specialist who had a great insight.",
            technical: "Each transformer layer routes the token to K out of 256 experts via a learned gating network. Each expert is a ~1.7MB weight matrix loaded from SSD via pread(). Lower K = fewer SSD reads per layer = proportionally less I/O time (the dominant bottleneck at 56% of per-token latency). Quality degrades gracefully because the router still picks the best K from the full vocabulary."
        ),
        "ioFanout": SettingInfo(
            id: "ioFanout",
            title: "Expert I/O Fanout",
            analogy: "Think of reading a book page. Instead of reading the whole page in one go, you split it into strips and read them all simultaneously with multiple eyes. Fanout splits each expert weight file read into parallel chunks so the SSD controller can serve them concurrently.",
            technical: "Each expert (~1.7MB for 35B) is read via pread(). Fanout splits this into N page-aligned chunks dispatched via GCD dispatch_group_async. NVMe controllers have multiple queues and can serve parallel reads faster than a single large read. Best value depends on expert size vs NVMe page size (4KB). Diminishing returns above 4 chunks."
        ),
        "cmdMerge": SettingInfo(
            id: "cmdMerge",
            title: "CMD1+CMD2 Merge",
            analogy: "Like combining two errands into one trip instead of driving home between them. CMD1 (attention projections) and CMD2 (output projection + normalization) are separate GPU tasks. Merging them avoids the roundtrip of 'submit, wait, create new, submit' for each of the 30 linear attention layers.",
            technical: "For linear attention layers (GatedDeltaNet), the CPU phase between CMD1 and CMD2 is empty — the GPU already computed everything. CMD2's dispatches (o_proj matmul, residual add, RMS norm, routing, shared expert) are appended to CMD1 with pipeline barriers. Saves ~0.05-0.1ms per layer x 30 layers = 1.5-3ms per token."
        ),
        "fusedAttention": SettingInfo(
            id: "fusedAttention",
            title: "Fused Attention",
            analogy: "Standard attention is like a three-step cooking recipe: measure all ingredients (Q@K scores), mix them (softmax), then combine (scores@V). Fused attention does all three in one pass — like a skilled chef who seasons, mixes, and plates in a single flowing motion. Less cleanup between steps.",
            technical: "Replaces 3 separate GPU kernel dispatches (attn_scores, attn_softmax, attn_values) with a single fused kernel using FlashAttention-2 online softmax. Processes KV positions in blocks of 64, maintaining running max/sum/output. Eliminates 2 command encoder transitions per full-attention layer (10 layers). Uses unnormalized accumulation with single final division."
        ),
        "fusedExpert": SettingInfo(
            id: "fusedExpert",
            title: "Fused Expert Kernel",
            analogy: "Each expert normally does three separate calculations: gate, up, and activation. It's like washing, drying, and folding laundry in three separate trips. The fused kernel does all three in one pass through the data — one trip, everything done.",
            technical: "Combines gate_proj matmul + up_proj matmul + SiLU activation into a single Metal compute kernel (fused_gate_up_swiglu). Both gate and up dot products are computed in one loop over the input vector, then SiLU is applied immediately. Reduces from 3 GPU dispatches to 1 per expert, saving command encoder overhead for K experts x 40 layers."
        ),
        "expertPrefetch": SettingInfo(
            id: "expertPrefetch",
            title: "Expert Prefetch",
            analogy: "While the kitchen (GPU) is cooking layer 5's dish, the waiter (CPU) runs ahead to the pantry (SSD) to grab ingredients for layer 6. When the kitchen finishes layer 5, the ingredients for layer 6 are already on the counter — no waiting.",
            technical: "After CMD3(N) is submitted (deferred GPU execution), the system predicts which experts layer N+1 will need based on routing history. Those experts are pread() into Set B buffers asynchronously. When layer N+1 reaches its I/O phase, prefetch hits skip the pread entirely. Misses fall through to normal loading. Overlaps ~2.4ms of I/O with GPU compute time."
        ),
        "fp16Accum": SettingInfo(
            id: "fp16Accum",
            title: "FP16 Accumulation",
            analogy: "Imagine counting coins on a kitchen scale that rounds to one decimal. Each coin adds a tiny rounding error. After 500 coins, you might be off by one. But the scale reads twice as fast. FP16 does math at 2x the speed of FP32, but accumulates small rounding errors over hundreds of additions.",
            technical: "The dequant matvec inner loop changes from float32 to float16 accumulation. Apple's A-series GPU has dedicated fp16 ALUs at 2x throughput. The FMA becomes half-precision: fma(half(nibble), half(scale*x), half(bias*x)). Final output is promoted to float32 via simd_sum. Risk: fp16 has ~3 decimal digits; sums of 512+ elements may lose precision."
        ),
        "fp8KV": SettingInfo(
            id: "fp8KV",
            title: "FP8 KV Cache",
            analogy: "The KV cache is like a notebook where the model writes down what it's seen. FP32 uses a full page per note. FP8 uses a quarter page — same content, just more compressed handwriting. You fit 4x more notes in the same notebook, so the model can remember 4x more conversation.",
            technical: "Stores attention Key and Value vectors in FP8 E4M3 format (1 byte vs 4 bytes per element) with per-position dynamic scaling. Encoding: absmax/240 scale factor, each float clipped and quantized to 8-bit (1 sign, 4 exponent, 3 mantissa). Decoding is inline during attention compute. 4x memory reduction enables 4x longer context at the same memory budget."
        ),
        "maxContext": SettingInfo(
            id: "maxContext",
            title: "Max Context Length",
            analogy: "Context length is how far back the model can 'see' in the conversation. Like a person's short-term memory — 4K tokens is the last few minutes, 32K is the last hour. More context = better understanding of the conversation, but uses more memory.",
            technical: "Sets the maximum sequence length for KV cache allocation. Memory cost: num_full_attn_layers x 2 (K+V) x kv_heads x head_dim x bytes_per_elem x positions. For the 35B with 10 full-attn layers: 40KB/pos (FP32) or 10KB/pos (FP8). Auto mode uses os_proc_available_memory() to pick the largest safe value."
        ),
        "slidingWindow": SettingInfo(
            id: "slidingWindow",
            title: "Sliding Window",
            analogy: "Instead of remembering everything forever (which fills up memory), the full-attention layers only look at the last N tokens — like a window sliding along the conversation. But the 30 linear attention layers still remember everything through their state matrices. It's like having both short-term and long-term memory working together.",
            technical: "Implements a circular KV buffer for full attention layers. Write: cache_pos = kv->len % window_size. Read: attend only to the most recent window_size positions. The 30 GatedDeltaNet layers maintain full context via their 128x128 state matrices (O(1) memory). Only the 10 full attention layers are windowed. With window 4096 + FP8: fixed 40MB KV regardless of conversation length."
        ),
        "thinking": SettingInfo(
            id: "thinking",
            title: "Thinking Mode",
            analogy: "Like a student who shows their work before giving the final answer. The model reasons step-by-step inside <think> tags before responding. This usually produces better answers, but takes more tokens (and time). At low K values, the model may get stuck thinking forever — disable it for speed.",
            technical: "The chat template includes a <think> tag after the assistant turn header. The model generates reasoning tokens inside the think block, then emits </think> before the actual response. Think budget caps the maximum thinking tokens and force-emits </think>. Set to -1 to disable thinking entirely (removes <think> from the template)."
        ),
        "h2oBudget": SettingInfo(
            id: "h2oBudget",
            title: "KV Cache Budget (H\u{2082}O)",
            analogy: "Instead of remembering everything or only recent things, H\u{2082}O is like a smart student who keeps notes on the most important points from the whole lecture plus the last few minutes. It tracks which tokens the model pays most attention to and keeps those, dropping the rest.",
            technical: "H\u{2082}O (Heavy Hitter Oracle) eviction keeps 3 categories: attention sink tokens (first 4, always high-attention), recent tokens (last N), and heavy hitters (highest cumulative attention score). When the cache exceeds the budget, the lowest-scored non-protected positions are evicted. Reduces KV memory by 50-70% with minimal quality loss. Based on 'H\u{2082}O: Heavy-Hitter Oracle' (Zhang et al., 2023)."
        ),
        "ropeScaling": SettingInfo(
            id: "ropeScaling",
            title: "RoPE Scaling",
            analogy: "The model learns position by spinning numbers at specific speeds. Beyond the training window, these numbers 'go off the map' and output degrades. RoPE scaling adjusts the spin speeds so longer conversations stay on-map. Linear is the simplest (just slow everything down). NTK-aware is smarter (adjusts different dimensions differently). YaRN is best (combines NTK with an attention temperature fix).",
            technical: "Rotary Position Embeddings encode position as frequency-domain rotations: freq_i = 1/base^(2i/d). Linear scaling divides position by the scale factor. NTK-aware scales the base: base' = base * s^(d/(d-2)), spreading frequencies more evenly. YaRN adds per-dimension interpolation based on wavelength vs. original context, plus an attention logit temperature t = sqrt(0.1*ln(s)+1). References: NTK-aware (arXiv:2306.15595), YaRN (arXiv:2309.00071)."
        ),
        "prefillBatch": SettingInfo(
            id: "prefillBatch",
            title: "Prefill Batch Size",
            analogy: "Normally the model reads each prompt word one at a time. Batched prefill reads multiple words simultaneously, sharing the weight data across all of them. It is like a teacher reading one textbook page to the whole class at once instead of whispering it to each student individually.",
            technical: "Converts per-token GEMV (matrix-vector multiply) into batched GEMM (matrix-matrix multiply) during the prefill phase. Each weight row is read once from SSD and multiplied against N input vectors simultaneously, amortizing I/O cost. Uses dedicated Metal kernels with per-token accumulators. Memory overhead: ~33 MB for pfb=32. Causal attention uses the FlashAttention-2 unnormalized accumulator pattern for numerical stability."
        ),
    ]
}

// MARK: - Model List View

struct ModelListView: View {
    @Environment(FlashMoEEngine.self) private var engine
    @State private var localModels: [LocalModel] = []
    @State private var isScanning = true
    @State private var loadError: String?
    @State private var selectedModel: LocalModel?

    // Build info — set at compile time via COMMIT_HASH build setting or fallback
    private var buildCommitShort: String {
        Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "dev"
    }
    @AppStorage("cacheIOSplit") private var cacheIOSplit: Int = 1
    @AppStorage("activeExpertsK") private var activeExpertsK: Int = 0
    @AppStorage("cmdMergeEnabled") private var cmdMergeEnabled: Bool = true
    @AppStorage("fusedAttention") private var fusedAttention: Bool = false
    @AppStorage("thinkingEnabled") private var thinkingEnabled: Bool = true
    @AppStorage("thinkBudget") private var thinkBudget: Int = 2048
    @AppStorage("expertPrefetch") private var expertPrefetch: Bool = false
    @AppStorage("fusedExpert") private var fusedExpert: Bool = true
    @AppStorage("fp16Accumulation") private var fp16Accumulation: Bool = false
    @AppStorage("fp8KVCache") private var fp8KVCache: Bool = false
    @AppStorage("maxContext") private var maxContext: Int = 0
    @AppStorage("slidingWindow") private var slidingWindow: Int = 0
    @AppStorage("h2oBudget") private var h2oBudget: Int = 0
    @AppStorage("ropeScaling") private var ropeScaling: Int = 0  // encoded: mode * 10 + factor_index
    @AppStorage("prefillBatch") private var prefillBatch: Int = 1
    @State private var settingInfo: SettingInfo? = nil
    @State private var showFilePicker = false
    @State private var modelToExport: LocalModel? = nil
    @State private var importedBookmark: Data? = nil
    @State private var showImportActionAlert = false
    @State private var pendingImportURL: URL? = nil
    @State private var importProgress: String? = nil
    @State private var modelToDelete: LocalModel? = nil
    @State private var customRepoURL: String = ""
    @State private var showCustomURLError: String? = nil
    @State private var customEntries: [CatalogEntry] = []
    @State private var isResolvingURL = false
    private let downloadManager = DownloadManager.shared

    var body: some View {
        List {
            Section {
                headerView
            }
            .listRowBackground(Color.clear)

            if isScanning {
                Section {
                    HStack {
                        ProgressView()
                        Text("Scanning for models...")
                            .foregroundStyle(.secondary)
                    }
                }
            } else if localModels.isEmpty {
                Section {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("No models found")
                            .font(.headline)
#if os(iOS)
                        Text("Download a model below, or transfer one via Files.app.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
#else
                        Text("Download a model below, or use \"Open Model Folder\" to load from disk.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
#endif
                    }
                    .padding(.vertical, 4)
                }
            } else {
                Section("On Device") {
                    ForEach(localModels) { model in
                        ModelRow(model: model, isLoading: engine.state == .loading && selectedModel?.id == model.id)
                            .onTapGesture { loadModel(model) }
                            .swipeActions(edge: .leading) {
                                Button {
                                    modelToExport = model
                                } label: {
                                    Label("Move to Files", systemImage: "square.and.arrow.up")
                                }
                                .tint(.blue)
                            }
                            .swipeActions(edge: .trailing, allowsFullSwipe: false) {
                                Button(role: .destructive) {
                                    modelToDelete = model
                                } label: {
                                    Label("Delete", systemImage: "trash")
                                }
                            }
                    }
                }
            }

            // Custom URL download
            Section("Add Model from URL") {
                HStack {
                    TextField("HuggingFace repo (user/model)", text: $customRepoURL)
                        .textFieldStyle(.roundedBorder)
                        .autocorrectionDisabled()
                        #if os(iOS)
                        .textInputAutocapitalization(.never)
                        .keyboardType(.URL)
                        #endif
                    Button {
                        startCustomDownload()
                    } label: {
                        if isResolvingURL {
                            ProgressView()
                                .controlSize(.small)
                        } else {
                            Image(systemName: "plus.circle.fill")
                                .font(.title3)
                        }
                    }
                    .disabled(customRepoURL.trimmingCharacters(in: .whitespaces).isEmpty || isResolvingURL)
                }
                if let error = showCustomURLError {
                    Text(error)
                        .font(.caption)
                        .foregroundStyle(.red)
                }
                Text("Enter a HuggingFace repo ID (e.g. alexintosh/Qwen3.5-35B-A3B-Q4-FlashMoE) or full URL.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            // Download section — hide models already on device
            let localNames = Set(localModels.map { $0.name.lowercased() })
            let allEntries = ModelCatalog.models + customEntries
            let availableEntries = allEntries.filter { entry in
                // Hide if already downloaded on device
                let isOnDevice = downloadManager.isModelDownloaded(entry.id) || localNames.contains(entry.id.lowercased())
                let hasActiveDownload = downloadManager.activeDownload?.catalogId == entry.id
                    && downloadManager.activeDownload?.status != .complete
                return !isOnDevice || hasActiveDownload
            }
            if !availableEntries.isEmpty {
                Section("Download from HuggingFace") {
                    ForEach(availableEntries) { entry in
                        let hasActiveDownload = downloadManager.activeDownload?.catalogId == entry.id
                            && downloadManager.activeDownload?.status != .complete
                        ModelDownloadRow(
                            entry: entry,
                            downloadManager: downloadManager,
                            isDownloaded: !hasActiveDownload && downloadManager.isModelDownloaded(entry.id)
                        )
                    }
                }
            }

            Section("Expert Settings") {
                Picker(selection: $activeExpertsK) {
                    Text("Model default").tag(0)
                    Text("K=2 (fastest)").tag(2)
                    Text("K=3").tag(3)
                    Text("K=4").tag(4)
                    Text("K=5").tag(5)
                    Text("K=6").tag(6)
                    Text("K=7").tag(7)
                    Text("K=8").tag(8)
                    Text("K=9").tag(9)
                    Text("K=10 (full)").tag(10)
                } label: { settingLabel("Active Experts (K)", key: "activeExperts") }
                .pickerStyle(.menu)

                Picker(selection: $cacheIOSplit) {
                    Text("Off").tag(1)
                    Text("2 chunks").tag(2)
                    Text("4 chunks").tag(4)
                    Text("8 chunks").tag(8)
                } label: { settingLabel("I/O Fanout", key: "ioFanout") }
                .pickerStyle(.menu)

                Toggle(isOn: $cmdMergeEnabled) { settingLabel("CMD1+CMD2 Merge", key: "cmdMerge") }
                Toggle(isOn: $fusedAttention) { settingLabel("Fused Attention", key: "fusedAttention") }
                Toggle(isOn: $fusedExpert) { settingLabel("Fused Expert Kernel", key: "fusedExpert") }
                Toggle(isOn: $expertPrefetch) { settingLabel("Expert Prefetch", key: "expertPrefetch") }
                Toggle(isOn: $fp16Accumulation) { settingLabel("FP16 Accumulation", key: "fp16Accum") }
                Toggle(isOn: $fp8KVCache) { settingLabel("FP8 KV Cache", key: "fp8KV") }

                Picker(selection: $prefillBatch) {
                    Text("Off (1)").tag(1)
                    Text("8 tokens").tag(8)
                    Text("16 tokens").tag(16)
                    Text("32 tokens").tag(32)
                } label: { settingLabel("Prefill Batch Size", key: "prefillBatch") }
                .pickerStyle(.menu)

                Picker(selection: $maxContext) {
                    Text("Auto").tag(0)
                    Text("4K").tag(4096)
                    Text("8K").tag(8192)
                    Text("16K").tag(16384)
                    Text("32K").tag(32768)
                } label: { settingLabel("Max Context", key: "maxContext") }
                .pickerStyle(.menu)

                Picker(selection: $slidingWindow) {
                    Text("Off").tag(0)
                    Text("2K").tag(2048)
                    Text("4K").tag(4096)
                    Text("8K").tag(8192)
                } label: { settingLabel("Sliding Window", key: "slidingWindow") }
                .pickerStyle(.menu)

                Picker(selection: $h2oBudget) {
                    Text("Off").tag(0)
                    Text("256").tag(256)
                    Text("512").tag(512)
                    Text("1024").tag(1024)
                    Text("2048").tag(2048)
                } label: { settingLabel("KV Cache Budget (H\u{2082}O)", key: "h2oBudget") }
                .pickerStyle(.menu)

                Picker(selection: $ropeScaling) {
                    Text("Off").tag(0)
                    Text("Linear 2x").tag(12)
                    Text("Linear 4x").tag(14)
                    Text("NTK-aware 2x").tag(22)
                    Text("NTK-aware 4x").tag(24)
                    Text("YaRN 2x").tag(32)
                    Text("YaRN 4x").tag(34)
                } label: { settingLabel("RoPE Scaling", key: "ropeScaling") }
                .pickerStyle(.menu)

                Toggle(isOn: $thinkingEnabled) { settingLabel("Thinking", key: "thinking") }

                if thinkingEnabled {
                    Picker("Think Budget", selection: $thinkBudget) {
                        Text("128").tag(128)
                        Text("256").tag(256)
                        Text("512").tag(512)
                        Text("1024").tag(1024)
                        Text("2048").tag(2048)
                        Text("Unlimited").tag(0)
                    }
                    .pickerStyle(.menu)
                }

                // Reload button — always visible, applies all settings changes
                Button {
                    if let model = selectedModel ?? localModels.first {
                        Task {
                            if engine.state == .ready {
                                engine.unloadModel()
                                try? await Task.sleep(for: .milliseconds(300))
                            }
                            loadModel(model)
                        }
                    }
                } label: {
                    Label("Reload Model", systemImage: "arrow.clockwise")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(.orange)
                .disabled(localModels.isEmpty)
            }

            // Build info
            Section {
                Text("Branch: develop • \(buildCommitShort)")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity)
                    .listRowBackground(Color.clear)
            }

            if let error = downloadManager.error,
               downloadManager.activeDownload == nil {
                Section {
                    Label(error, systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.red)
                        .font(.caption)
                }
            }

            if case .error(let msg) = engine.state {
                Section {
                    Label(msg, systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.red)
                        .font(.caption)
                }
            }
        }
        .navigationTitle("Flash-MoE")
        .toolbar {
#if os(iOS)
            ToolbarItem(placement: .topBarTrailing) {
                Button {
                    showFilePicker = true
                } label: {
                    Label("Import", systemImage: "folder.badge.plus")
                }
            }
#else
            ToolbarItem(placement: .automatic) {
                Button {
                    openMacModelFolder()
                } label: {
                    Label("Open Model Folder...", systemImage: "folder.badge.plus")
                }
            }
#endif
        }
        .sheet(item: $settingInfo) { info in
            NavigationStack {
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        Text(info.analogy)
                            .font(.body)
                            .foregroundStyle(.primary)

                        Divider()

                        Text("How it works")
                            .font(.headline)
                            .foregroundStyle(.secondary)

                        Text(info.technical)
                            .font(.callout)
                            .foregroundStyle(.secondary)
                    }
                    .padding()
                }
                .navigationTitle(info.title)
                #if os(iOS)
                .navigationBarTitleDisplayMode(.inline)
                #endif
                .toolbar {
                    ToolbarItem(placement: .confirmationAction) {
                        Button("Done") { settingInfo = nil }
                    }
                }
            }
            .presentationDetents([.medium])
        }
#if os(iOS)
        .sheet(isPresented: $showFilePicker) {
            FolderImportPicker { url in
                pendingImportURL = url
                showImportActionAlert = true
            }
        }
#endif
#if os(iOS)
        .alert("Import Model", isPresented: $showImportActionAlert) {
            Button("Link (Bookmark)") {
                if let url = pendingImportURL {
                    handleImportedFolder(url)
                }
                pendingImportURL = nil
            }
            Button("Move to App (Documents)") {
                if let url = pendingImportURL {
                    moveImportedFolderToDocuments(url)
                }
                pendingImportURL = nil
            }
            Button("Cancel", role: .cancel) {
                pendingImportURL = nil
            }
        } message: {
            Text("Link keeps the model in its current location. Move to App copies it into the app's Documents folder for better reliability.")
        }
#endif
        .alert("Delete Model", isPresented: Binding(
            get: { modelToDelete != nil },
            set: { if !$0 { modelToDelete = nil } }
        )) {
            Button("Delete", role: .destructive) {
                if let model = modelToDelete {
                    deleteModel(model)
                }
                modelToDelete = nil
            }
            Button("Cancel", role: .cancel) { modelToDelete = nil }
        } message: {
            Text("Delete \"\(modelToDelete?.name ?? "")\" (\(String(format: "%.1f GB", modelToDelete?.sizeGB ?? 0.0)))? This cannot be undone.")
        }
#if os(iOS)
        .sheet(item: $modelToExport) { model in
            FolderExportPicker(sourceURL: URL(fileURLWithPath: model.path)) { destURL in
                // moveToService already moved the files — just refresh the model list
                print("[export] Model moved to: \(destURL.path)")
                scanForModels()
            }
        }
#endif
        .overlay {
            if let progress = importProgress {
                VStack(spacing: 12) {
                    ProgressView()
                    Text(progress)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(24)
                .background(.ultraThinMaterial)
                .clipShape(RoundedRectangle(cornerRadius: 12))
            }
        }
        .onAppear {
            scanForModels()
#if os(iOS)
            restoreBookmarks()
#endif
        }
        .refreshable { scanForModels() }
        .onChange(of: downloadManager.activeDownload?.status) { _, newStatus in
            if newStatus == .complete {
                scanForModels()
            }
        }
    }

    private var headerView: some View {
        VStack(spacing: 8) {
            Image(systemName: "bolt.fill")
                .font(.system(size: 48))
                .foregroundStyle(.orange)
            Text("Flash-MoE")
                .font(.largeTitle.bold())
#if os(iOS)
            Text("Run massive MoE models on iPhone")
                .font(.subheadline)
                .foregroundStyle(.secondary)
#else
            Text("Run massive MoE models on your Mac")
                .font(.subheadline)
                .foregroundStyle(.secondary)
#endif
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical)
    }

    private func scanForModels() {
        isScanning = true
        localModels = []

        Task {
            let models = await ModelScanner.scanLocalModels()
            await MainActor.run {
                localModels = models
                isScanning = false
            }
        }
    }

#if os(macOS)
    private func openMacModelFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.message = "Select a Flash-MoE model directory"
        panel.prompt = "Open"
        if panel.runModal() == .OK, let url = panel.url {
            // On macOS, no sandbox bookmark needed for user-selected directories.
            // Just store the path and scan.
            let fm = FileManager.default
            if FlashMoEEngine.validateModel(at: url.path) {
                // Valid model directory — add to scanned paths
                var paths = UserDefaults.standard.stringArray(forKey: "macModelPaths") ?? []
                if !paths.contains(url.path) {
                    paths.append(url.path)
                    UserDefaults.standard.set(paths, forKey: "macModelPaths")
                }
                scanForModels()
            } else {
                // Check if it contains model subdirectories
                var foundModel = false
                if let entries = try? fm.contentsOfDirectory(atPath: url.path) {
                    for entry in entries {
                        let fullPath = (url.path as NSString).appendingPathComponent(entry)
                        if FlashMoEEngine.validateModel(at: fullPath) {
                            foundModel = true
                            break
                        }
                    }
                }
                var paths = UserDefaults.standard.stringArray(forKey: "macModelPaths") ?? []
                if !paths.contains(url.path) {
                    paths.append(url.path)
                    UserDefaults.standard.set(paths, forKey: "macModelPaths")
                }
                scanForModels()
            }
        }
    }
#endif

    private func loadModel(_ model: LocalModel) {
        guard engine.state != .loading && engine.state != .generating else { return }
        selectedModel = model
        // Store path for benchmark access
        UserDefaults.standard.set(model.path, forKey: "lastLoadedModelPath")

        // Use picker value, or auto-detect for 397B if user hasn't set a preference
        let activeK: Int
        if activeExpertsK > 0 {
            activeK = activeExpertsK  // user selected a value
        } else {
            // Auto-reduce for 397B on constrained devices
            let is397B = model.path.lowercased().contains("397b") || model.name.lowercased().contains("397b")
            let deviceRAM = ProcessInfo.processInfo.physicalMemory / (1024 * 1024 * 1024)
            activeK = (is397B && deviceRAM <= 16) ? 4 : 0
        }

        // Decode ropeScaling tag: tens digit = mode, ones digit encodes factor
        // 0 = off, 12 = linear 2x, 14 = linear 4x, 22 = NTK 2x, 24 = NTK 4x, 32 = YaRN 2x, 34 = YaRN 4x
        let ropeMode = ropeScaling / 10
        let ropeFactor: Float = ropeScaling == 0 ? 1.0 : Float(ropeScaling % 10)

        Task {
            do {
                try await engine.loadModel(
                    at: model.path,
                    maxContext: maxContext,
                    thinkBudget: thinkingEnabled ? thinkBudget : -1,
                    useTiered: model.hasTiered,
                    activeExpertsK: activeK,
                    cacheIOSplit: cacheIOSplit,
                    cmdMerge: cmdMergeEnabled,
                    fusedAttention: fusedAttention,
                    expertPrefetch: expertPrefetch,
                    fusedExpert: fusedExpert,
                    fp16Accumulation: fp16Accumulation,
                    fp8KVCache: fp8KVCache,
                    slidingWindow: slidingWindow,
                    h2oBudget: h2oBudget,
                    ropeScalingMode: ropeMode,
                    ropeScaleFactor: ropeFactor,
                    prefillBatch: prefillBatch,
                    verbose: true
                )
            } catch {
                // Error state is set by the engine
            }
        }
    }

    // MARK: - Custom URL Download

    private func startCustomDownload() {
        showCustomURLError = nil
        isResolvingURL = true
        var input = customRepoURL.trimmingCharacters(in: .whitespacesAndNewlines)

        // Accept full URLs or repo IDs
        // https://huggingface.co/user/repo → user/repo
        if input.hasPrefix("https://huggingface.co/") {
            input = String(input.dropFirst("https://huggingface.co/".count))
        }
        if input.hasPrefix("http://huggingface.co/") {
            input = String(input.dropFirst("http://huggingface.co/".count))
        }
        // Remove trailing slashes and tree/main suffix
        input = input.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        if input.hasSuffix("/tree/main") {
            input = String(input.dropLast("/tree/main".count))
        }

        let parts = input.split(separator: "/")
        guard parts.count >= 2 else {
            showCustomURLError = "Invalid format. Use 'user/model' or a HuggingFace URL."
            isResolvingURL = false
            return
        }
        let repoId = "\(parts[0])/\(parts[1])"
        let modelName = String(parts[1])

        // Query HuggingFace API for file list
        let apiURL = URL(string: "https://huggingface.co/api/models/\(repoId)")!
        Task {
            do {
                let (data, response) = try await URLSession.shared.data(from: apiURL)
                guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
                    await MainActor.run { showCustomURLError = "Repository not found: \(repoId)"; isResolvingURL = false }
                    return
                }
                guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                      let siblings = json["siblings"] as? [[String: Any]] else {
                    await MainActor.run { showCustomURLError = "Failed to parse repository metadata."; isResolvingURL = false }
                    return
                }

                var files: [RepoFile] = []
                var totalSize: UInt64 = 0
                var expertLayers = 0

                for sibling in siblings {
                    guard let filename = sibling["rfilename"] as? String else { continue }
                    // Skip README, .gitattributes, etc.
                    let lower = filename.lowercased()
                    if lower.hasSuffix(".md") || lower.hasSuffix(".gitattributes") || lower.hasPrefix(".") { continue }

                    // Get file size from LFS info or default to 0
                    let size: UInt64
                    if let lfs = sibling["lfs"] as? [String: Any], let s = lfs["size"] as? UInt64 {
                        size = s
                    } else if let s = sibling["size"] as? UInt64 {
                        size = s
                    } else {
                        size = 0
                    }

                    files.append(RepoFile(filename: filename, sizeBytes: size))
                    totalSize += size

                    if filename.contains("packed_experts") && filename.hasSuffix(".bin") && filename.contains("layer_") {
                        expertLayers += 1
                    }
                }

                guard !files.isEmpty else {
                    await MainActor.run { showCustomURLError = "No files found in repository."; isResolvingURL = false }
                    return
                }

                // Check for required files
                let hasConfig = files.contains { $0.filename == "config.json" }
                let hasWeights = files.contains { $0.filename == "model_weights.bin" }
                    || files.contains { $0.filename == "model_weights_0.bin" }
                guard hasConfig && hasWeights else {
                    await MainActor.run {
                        showCustomURLError = "Not a Flash-MoE model (missing config.json or model_weights.bin)."
                        isResolvingURL = false
                    }
                    return
                }

                let entry = CatalogEntry(
                    id: "custom-\(modelName.lowercased())",
                    displayName: modelName,
                    repoId: repoId,
                    description: "Custom model from \(repoId)",
                    totalSizeBytes: totalSize,
                    quantization: "unknown",
                    expertLayers: expertLayers,
                    defaultK: 0,
                    recommendedK: 0,
                    minRAMGB: 0,
                    files: files
                )

                await MainActor.run {
                    customRepoURL = ""
                    showCustomURLError = nil
                    isResolvingURL = false
                    // Add to the download list — user taps the row to start
                    if !customEntries.contains(where: { $0.id == entry.id }) {
                        customEntries.append(entry)
                    }
                }
            } catch {
                await MainActor.run { showCustomURLError = "Network error: \(error.localizedDescription)"; isResolvingURL = false }
            }
        }
    }

    // MARK: - Setting Label with Info Button

    @ViewBuilder
    private func settingLabel(_ title: String, key: String) -> some View {
        HStack(spacing: 6) {
            Button {
                settingInfo = SettingInfo.all[key]
            } label: {
                Image(systemName: "info.circle")
                    .font(.caption)
                    .foregroundStyle(.blue)
            }
            .buttonStyle(.plain)
            Text(title)
        }
    }

    private func deleteModel(_ model: LocalModel) {
        do {
            try FileManager.default.removeItem(atPath: model.path)
            print("[delete] Removed \(model.name) at \(model.path)")
            scanForModels()
        } catch {
            print("ERROR: Failed to delete \(model.name): \(error)")
        }
    }

#if os(iOS)
    // MARK: - File Import (iOS)

    private func handleImportedFolder(_ url: URL) {
        // Save a security-scoped bookmark so we can access this folder across launches
        guard url.startAccessingSecurityScopedResource() else {
            print("ERROR: Failed to access security-scoped resource")
            return
        }

        do {
            let bookmarkData = try url.bookmarkData(
                options: .minimalBookmark,
                includingResourceValuesForKeys: nil,
                relativeTo: nil
            )
            // Save bookmark to UserDefaults
            var bookmarks = UserDefaults.standard.array(forKey: "importedModelBookmarks") as? [Data] ?? []
            bookmarks.append(bookmarkData)
            UserDefaults.standard.set(bookmarks, forKey: "importedModelBookmarks")

            print("[import] Bookmarked external model folder: \(url.path)")
            scanForModels()
        } catch {
            print("ERROR: Failed to create bookmark: \(error)")
        }

        url.stopAccessingSecurityScopedResource()
    }

    private func moveImportedFolderToDocuments(_ url: URL) {
        guard url.startAccessingSecurityScopedResource() else {
            print("ERROR: Failed to access security-scoped resource for move")
            return
        }

        let fm = FileManager.default
        guard let docsDir = fm.urls(for: .documentDirectory, in: .userDomainMask).first else {
            url.stopAccessingSecurityScopedResource()
            return
        }

        let destURL = docsDir.appendingPathComponent(url.lastPathComponent)

        importProgress = "Moving model to Documents..."
        print("[import] Moving \(url.path) -> \(destURL.path)")

        Task {
            do {
                // moveItem is instant if same filesystem, otherwise it copies
                try fm.moveItem(at: url, to: destURL)
                print("[import] Move succeeded")
            } catch {
                print("[import] Move failed: \(error). Trying copy...")
                await MainActor.run { importProgress = "Copying model to Documents (this may take a while)..." }
                do {
                    try fm.copyItem(at: url, to: destURL)
                    print("[import] Copy succeeded")
                } catch {
                    print("ERROR: Copy also failed: \(error)")
                }
            }

            url.stopAccessingSecurityScopedResource()

            await MainActor.run {
                importProgress = nil
                scanForModels()
            }
        }
    }

    private func restoreBookmarks() {
        guard let bookmarks = UserDefaults.standard.array(forKey: "importedModelBookmarks") as? [Data] else { return }

        for bookmark in bookmarks {
            var isStale = false
            if let url = try? URL(resolvingBookmarkData: bookmark, bookmarkDataIsStale: &isStale) {
                if !isStale {
                    _ = url.startAccessingSecurityScopedResource()
                }
            }
        }
    }

    private func moveModelToExternal(model: LocalModel, destination: URL) {
        Task {
            let fm = FileManager.default
            let destPath = destination.appendingPathComponent(URL(fileURLWithPath: model.path).lastPathComponent)

            guard destination.startAccessingSecurityScopedResource() else {
                print("ERROR: Cannot access destination")
                return
            }
            defer { destination.stopAccessingSecurityScopedResource() }

            do {
                // Move (not copy) — instant on same filesystem
                try fm.moveItem(at: URL(fileURLWithPath: model.path), to: destPath)
                print("[export] Moved \(model.name) to \(destPath.path)")
                await MainActor.run { scanForModels() }
            } catch {
                print("ERROR: Move failed: \(error). Trying copy instead...")
                // If move fails (cross-volume), this would be slow for 300GB
                // but at least it works
                do {
                    try fm.copyItem(at: URL(fileURLWithPath: model.path), to: destPath)
                    try fm.removeItem(at: URL(fileURLWithPath: model.path))
                    print("[export] Copied + deleted \(model.name) to \(destPath.path)")
                    await MainActor.run { scanForModels() }
                } catch {
                    print("ERROR: Copy also failed: \(error)")
                }
            }
        }
    }
#endif
}

// MARK: - Folder Import/Export Pickers (iOS)

#if os(iOS)
struct FolderImportPicker: UIViewControllerRepresentable {
    let onPick: (URL) -> Void

    func makeCoordinator() -> Coordinator { Coordinator(onPick: onPick) }

    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        let picker = UIDocumentPickerViewController(forOpeningContentTypes: [.folder])
        picker.allowsMultipleSelection = false
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: UIDocumentPickerViewController, context: Context) {}

    class Coordinator: NSObject, UIDocumentPickerDelegate {
        let onPick: (URL) -> Void
        init(onPick: @escaping (URL) -> Void) { self.onPick = onPick }

        func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
            guard let url = urls.first else { return }
            onPick(url)
        }
    }
}

// MARK: - Folder Export Picker (pick destination to move model to)

struct FolderExportPicker: UIViewControllerRepresentable {
    let sourceURL: URL
    let onPick: (URL) -> Void

    func makeCoordinator() -> Coordinator { Coordinator(onPick: onPick) }

    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        // moveToService: shows full Files browser, user picks destination folder.
        // iOS moves the directory to the chosen location.
        let picker = UIDocumentPickerViewController(urls: [sourceURL], in: .moveToService)
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: UIDocumentPickerViewController, context: Context) {}

    class Coordinator: NSObject, UIDocumentPickerDelegate {
        let onPick: (URL) -> Void
        init(onPick: @escaping (URL) -> Void) { self.onPick = onPick }

        func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
            guard let url = urls.first else { return }
            onPick(url)
        }
    }
}
#endif

// MARK: - Model Row

struct ModelRow: View {
    let model: LocalModel
    let isLoading: Bool

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(model.name)
                    .font(.headline)

                HStack(spacing: 8) {
                    if model.hasTiered {
                        QuantBadge(text: "Tiered", color: .green)
                    } else if model.has4bit {
                        QuantBadge(text: "4-bit", color: .blue)
                    } else if model.has2bit {
                        QuantBadge(text: "2-bit", color: .orange)
                    }

                    Text(String(format: "%.1f GB", model.sizeGB))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            Spacer()

            if isLoading {
                ProgressView()
            } else {
                Image(systemName: "chevron.right")
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 4)
        .contentShape(Rectangle())
    }
}

struct QuantBadge: View {
    let text: String
    let color: Color

    var body: some View {
        Text(text)
            .font(.caption2.bold())
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.15))
            .foregroundStyle(color)
            .clipShape(Capsule())
    }
}

// MARK: - Model Scanner

enum ModelScanner {
    /// Scan common locations for Flash-MoE model directories
    static func scanLocalModels() async -> [LocalModel] {
        var models: [LocalModel] = []
        let fm = FileManager.default

        // Scan app Documents directory
        if let docsDir = fm.urls(for: .documentDirectory, in: .userDomainMask).first {
            await scanDirectory(docsDir.path, into: &models)
        }

#if os(macOS)
        // Scan user-added model directories (macOS)
        if let macPaths = UserDefaults.standard.stringArray(forKey: "macModelPaths") {
            for path in macPaths {
                let fm2 = FileManager.default
                var isDir: ObjCBool = false
                guard fm2.fileExists(atPath: path, isDirectory: &isDir), isDir.boolValue else { continue }
                if FlashMoEEngine.validateModel(at: path) {
                    let size = directorySize(at: path)
                    let hasTiered = fm2.fileExists(atPath: (path as NSString).appendingPathComponent("packed_experts_tiered/layer_00.bin"))
                    let has4bit = fm2.fileExists(atPath: (path as NSString).appendingPathComponent("packed_experts/layer_00.bin"))
                    let has2bit = fm2.fileExists(atPath: (path as NSString).appendingPathComponent("packed_experts_2bit/layer_00.bin"))
                    models.append(LocalModel(
                        name: URL(fileURLWithPath: path).lastPathComponent,
                        path: path,
                        sizeBytes: size,
                        hasTiered: hasTiered,
                        has4bit: has4bit,
                        has2bit: has2bit
                    ))
                } else {
                    await scanDirectory(path, into: &models)
                }
            }
        }
#endif

        // Scan bookmarked external folders (imported via Files picker on iOS)
        if let bookmarks = UserDefaults.standard.array(forKey: "importedModelBookmarks") as? [Data] {
            for bookmark in bookmarks {
                var isStale = false
                if let url = try? URL(resolvingBookmarkData: bookmark, bookmarkDataIsStale: &isStale),
                   !isStale {
                    let accessed = url.startAccessingSecurityScopedResource()
                    // Check if the folder itself is a model
                    if FlashMoEEngine.validateModel(at: url.path) {
                        let size = directorySize(at: url.path)
                        let hasTiered = fm.fileExists(atPath: url.appendingPathComponent("packed_experts_tiered/layer_00.bin").path)
                        let has4bit = fm.fileExists(atPath: url.appendingPathComponent("packed_experts/layer_00.bin").path)
                        let has2bit = fm.fileExists(atPath: url.appendingPathComponent("packed_experts_2bit/layer_00.bin").path)
                        models.append(LocalModel(
                            name: "📁 " + url.lastPathComponent,
                            path: url.path,
                            sizeBytes: size,
                            hasTiered: hasTiered,
                            has4bit: has4bit,
                            has2bit: has2bit
                        ))
                    } else {
                        // Scan subdirectories
                        await scanDirectory(url.path, into: &models)
                    }
                    if accessed { url.stopAccessingSecurityScopedResource() }
                }
            }
        }

        return models.sorted { $0.name < $1.name }
    }

    private static func scanDirectory(_ path: String, into models: inout [LocalModel]) async {
        let fm = FileManager.default

        guard let entries = try? fm.contentsOfDirectory(atPath: path) else { return }

        for entry in entries {
            let fullPath = (path as NSString).appendingPathComponent(entry)
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: fullPath, isDirectory: &isDir), isDir.boolValue else { continue }

            // Check if it's a valid model
            if FlashMoEEngine.validateModel(at: fullPath) {
                // Protect model files from iOS storage optimization / purging
                excludeFromBackup(URL(fileURLWithPath: fullPath))
                let size = directorySize(at: fullPath)
                let hasTiered = fm.fileExists(atPath: (fullPath as NSString).appendingPathComponent("packed_experts_tiered/layer_00.bin"))
                let has4bit = fm.fileExists(atPath: (fullPath as NSString).appendingPathComponent("packed_experts/layer_00.bin"))
                let has2bit = fm.fileExists(atPath: (fullPath as NSString).appendingPathComponent("packed_experts_2bit/layer_00.bin"))

                models.append(LocalModel(
                    name: entry,
                    path: fullPath,
                    sizeBytes: size,
                    hasTiered: hasTiered,
                    has4bit: has4bit,
                    has2bit: has2bit
                ))
            }
        }
    }

    private static func directorySize(at path: String) -> UInt64 {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(atPath: path) else { return 0 }
        var total: UInt64 = 0
        while let file = enumerator.nextObject() as? String {
            let fullPath = (path as NSString).appendingPathComponent(file)
            if let attrs = try? fm.attributesOfItem(atPath: fullPath),
               let size = attrs[.size] as? UInt64 {
                total += size
            }
        }
        return total
    }

    /// Mark a directory (and its contents) as excluded from iCloud backup and
    /// iOS storage optimization, preventing the system from purging model files.
    private static func excludeFromBackup(_ url: URL) {
        var url = url
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? url.setResourceValues(values)

        // Also mark all files inside
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(at: url, includingPropertiesForKeys: nil) else { return }
        while let fileURL = enumerator.nextObject() as? URL {
            var fileURL = fileURL
            try? fileURL.setResourceValues(values)
        }
    }
}
