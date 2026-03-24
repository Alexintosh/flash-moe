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

// MARK: - Model List View

struct ModelListView: View {
    @Environment(FlashMoEEngine.self) private var engine
    @State private var localModels: [LocalModel] = []
    @State private var isScanning = true
    @State private var loadError: String?
    @State private var selectedModel: LocalModel?
    @AppStorage("cacheIOSplit") private var cacheIOSplit: Int = 1
    @AppStorage("activeExpertsK") private var activeExpertsK: Int = 0
    @AppStorage("cmdMergeEnabled") private var cmdMergeEnabled: Bool = true
    @AppStorage("fusedAttention") private var fusedAttention: Bool = false
    @AppStorage("thinkingEnabled") private var thinkingEnabled: Bool = true
    @AppStorage("thinkBudget") private var thinkBudget: Int = 2048
    @AppStorage("expertPrefetch") private var expertPrefetch: Bool = false
    @AppStorage("fusedExpert") private var fusedExpert: Bool = true
    @AppStorage("fp16Accumulation") private var fp16Accumulation: Bool = false
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
                Picker("Active Experts (K)", selection: $activeExpertsK) {
                    Text("Model default").tag(0)
                    Text("K=2 (fastest, lowest quality)").tag(2)
                    Text("K=3").tag(3)
                    Text("K=4").tag(4)
                    Text("K=5").tag(5)
                    Text("K=6").tag(6)
                    Text("K=7").tag(7)
                    Text("K=8").tag(8)
                    Text("K=9").tag(9)
                    Text("K=10 (full quality)").tag(10)
                }
                .pickerStyle(.menu)
                if activeExpertsK > 0 {
                    Text("Uses \(activeExpertsK) experts per token instead of the model default. Lower = faster but less accurate. Reload model to apply.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Picker("Expert I/O Fanout", selection: $cacheIOSplit) {
                    Text("Off (single pread)").tag(1)
                    Text("2 chunks").tag(2)
                    Text("4 chunks").tag(4)
                    Text("8 chunks").tag(8)
                }
                .pickerStyle(.menu)
                if cacheIOSplit > 1 {
                    Text("Splits each expert read into \(cacheIOSplit) page-aligned chunks for parallel SSD reads. Reload model to apply.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Toggle("CMD1+CMD2 Merge", isOn: $cmdMergeEnabled)
                Text(cmdMergeEnabled
                     ? "Merges GPU command buffers for linear attention layers. Faster but experimental."
                     : "Separate command buffers. Safer, slightly slower.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Toggle("Fused Attention", isOn: $fusedAttention)
                Text(fusedAttention
                     ? "Single-kernel online softmax attention. Experimental — reduces GPU dispatches."
                     : "Standard 3-kernel attention pipeline (scores, softmax, values). Proven correct.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Toggle("Fused Expert Kernel", isOn: $fusedExpert)
                Text(fusedExpert
                     ? "Fused gate+up+SwiGLU in one kernel. Fewer GPU dispatches per expert."
                     : "Separate gate, up, SwiGLU dispatches per expert.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Toggle("Expert Prefetch", isOn: $expertPrefetch)
                Text(expertPrefetch
                     ? "Cross-layer prefetch: loads next layer's predicted experts during GPU compute. Hides I/O behind execution."
                     : "Disabled — experts loaded on-demand only. No overlap with GPU compute.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Toggle("FP16 Accumulation", isOn: $fp16Accumulation)
                Text(fp16Accumulation
                     ? "Uses half-precision math in weight matmuls. ~5-10% faster but may reduce output quality."
                     : "Standard float32 accumulation. Maximum precision.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Toggle("Thinking", isOn: $thinkingEnabled)
                Text(thinkingEnabled
                     ? "Model can reason in <think> tags before answering."
                     : "Thinking disabled — model answers directly. Better for low K values.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                if thinkingEnabled {
                    Picker("Think Budget", selection: $thinkBudget) {
                        Text("128 tokens").tag(128)
                        Text("256 tokens").tag(256)
                        Text("512 tokens").tag(512)
                        Text("1024 tokens").tag(1024)
                        Text("2048 tokens (default)").tag(2048)
                        Text("Unlimited").tag(0)
                    }
                    .pickerStyle(.menu)
                }
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

        Task {
            do {
                try await engine.loadModel(
                    at: model.path,
                    thinkBudget: thinkingEnabled ? thinkBudget : -1,
                    useTiered: model.hasTiered,
                    activeExpertsK: activeK,
                    cacheIOSplit: cacheIOSplit,
                    cmdMerge: cmdMergeEnabled,
                    fusedAttention: fusedAttention,
                    expertPrefetch: expertPrefetch,
                    fusedExpert: fusedExpert,
                    fp16Accumulation: fp16Accumulation,
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
