/*
 * DownloadManager.swift — Background download orchestration for HuggingFace models
 *
 * Uses URLSession background downloads that survive app termination.
 * Downloads files sequentially within a model for clean progress tracking.
 * State persisted to downloads.json for resume across app launches.
 */

import Foundation
import Observation

// MARK: - Download State

enum DownloadStatus: String, Codable, Sendable {
    case downloading
    case paused
    case failed
    case complete
}

struct DownloadState: Codable {
    let catalogId: String
    let repoId: String
    var completedFiles: [String]
    var completedBytes: UInt64
    var currentFile: String?
    var status: DownloadStatus
    var errorMessage: String?
}

// MARK: - DownloadManager

@Observable
final class DownloadManager: NSObject, @unchecked Sendable {
    static let shared = DownloadManager()

    // Observable state
    private(set) var activeDownload: DownloadState?
    private(set) var overallProgress: Double = 0
    private(set) var currentFileProgress: Double = 0
    private(set) var bytesDownloaded: UInt64 = 0
    private(set) var totalBytes: UInt64 = 0
    private(set) var error: String?
    private(set) var downloadSpeed: Double = 0 // bytes/sec

    // Background session callback
    var backgroundCompletionHandler: (() -> Void)?

    // Private state
    private var backgroundSession: URLSession!
    private var currentTask: URLSessionDownloadTask?
    private var currentEntry: CatalogEntry?
    private var resumeData: Data?
    private var speedSampleTime: Date?
    private var speedSampleBytes: UInt64 = 0

    private static let sessionIdentifier = "com.flashmoe.model-download"

    // MARK: - Initialization

    override private init() {
        super.init()
        let config = URLSessionConfiguration.background(withIdentifier: Self.sessionIdentifier)
        config.isDiscretionary = false
        config.sessionSendsLaunchEvents = true
        config.allowsCellularAccess = true
        backgroundSession = URLSession(configuration: config, delegate: self, delegateQueue: nil)

        // Restore persisted state
        loadPersistedState()

        // Reconnect to any in-flight background tasks
        backgroundSession.getTasksWithCompletionHandler { [weak self] _, _, downloadTasks in
            if let task = downloadTasks.first {
                self?.currentTask = task
            }
        }
    }

    // MARK: - Public API

    func startDownload(entry: CatalogEntry) {
        guard activeDownload == nil || activeDownload?.status == .complete else {
            error = "A download is already in progress"
            return
        }

        // Check disk space
        let available = availableDiskSpace()
        if available < entry.totalSizeBytes {
            let needed = formatBytes(entry.totalSizeBytes)
            let have = formatBytes(available)
            error = "Not enough space: \(needed) needed, \(have) available"
            return
        }

        error = nil
        currentEntry = entry
        totalBytes = entry.totalSizeBytes

        // Create model directory
        let modelDir = modelDirectory(for: entry.id)
        createDirectoryStructure(for: entry, at: modelDir)

        activeDownload = DownloadState(
            catalogId: entry.id,
            repoId: entry.repoId,
            completedFiles: [],
            completedBytes: 0,
            currentFile: nil,
            status: .downloading
        )
        persistState()
        downloadNextFile()
    }

    func pauseDownload() {
        guard activeDownload?.status == .downloading else { return }

        currentTask?.cancel(byProducingResumeData: { [weak self] data in
            guard let self else { return }
            self.resumeData = data
            self.activeDownload?.status = .paused
            self.persistState()
            self.currentTask = nil
        })
    }

    func resumeDownload() {
        guard activeDownload?.status == .paused || activeDownload?.status == .failed else { return }

        // Resolve the catalog entry
        if currentEntry == nil, let catalogId = activeDownload?.catalogId {
            currentEntry = ModelCatalog.models.first { $0.id == catalogId }
        }
        guard currentEntry != nil else {
            error = "Cannot find model in catalog"
            return
        }

        error = nil
        activeDownload?.status = .downloading
        activeDownload?.errorMessage = nil
        totalBytes = currentEntry?.totalSizeBytes ?? 0
        persistState()

        if let resumeData {
            let task = backgroundSession.downloadTask(withResumeData: resumeData)
            task.resume()
            currentTask = task
            self.resumeData = nil
        } else {
            downloadNextFile()
        }
    }

    func cancelDownload() {
        currentTask?.cancel()
        currentTask = nil
        resumeData = nil

        if let catalogId = activeDownload?.catalogId {
            let dir = modelDirectory(for: catalogId)
            try? FileManager.default.removeItem(at: dir)
        }

        activeDownload = nil
        overallProgress = 0
        currentFileProgress = 0
        bytesDownloaded = 0
        totalBytes = 0
        error = nil
        currentEntry = nil
        clearPersistedState()
    }

    func deleteModel(catalogId: String) {
        let dir = modelDirectory(for: catalogId)
        try? FileManager.default.removeItem(at: dir)

        if activeDownload?.catalogId == catalogId {
            activeDownload = nil
            clearPersistedState()
        }
    }

    func isModelDownloaded(_ catalogId: String) -> Bool {
        let dir = modelDirectory(for: catalogId)
        return FlashMoEEngine.validateModel(at: dir.path)
    }

    func modelPath(for catalogId: String) -> String {
        modelDirectory(for: catalogId).path
    }

    // MARK: - File Management

    private func modelDirectory(for catalogId: String) -> URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent(catalogId)
    }

    private func createDirectoryStructure(for entry: CatalogEntry, at baseURL: URL) {
        let fm = FileManager.default
        try? fm.createDirectory(at: baseURL, withIntermediateDirectories: true)

        // Create subdirectories for expert files
        var subdirs = Set<String>()
        for file in entry.files {
            let url = baseURL.appendingPathComponent(file.filename)
            let parent = url.deletingLastPathComponent()
            if parent != baseURL {
                subdirs.insert(parent.path)
            }
        }
        for dir in subdirs {
            try? fm.createDirectory(atPath: dir, withIntermediateDirectories: true)
        }
    }

    // MARK: - Sequential Download Engine

    private func downloadNextFile() {
        guard var state = activeDownload, let entry = currentEntry else { return }

        // Find next file to download
        let nextFile = entry.files.first { !state.completedFiles.contains($0.filename) }

        guard let file = nextFile else {
            // All files downloaded
            state.status = .complete
            state.currentFile = nil
            activeDownload = state
            overallProgress = 1.0
            persistState()

            // Validate the model
            let dir = modelDirectory(for: entry.id)
            if !FlashMoEEngine.validateModel(at: dir.path) {
                error = "Download complete but model validation failed"
                state.status = .failed
                state.errorMessage = "Validation failed — some files may be corrupt"
                activeDownload = state
                persistState()
            }
            return
        }

        state.currentFile = file.filename
        activeDownload = state
        persistState()

        let url = entry.downloadURL(for: file)
        let task = backgroundSession.downloadTask(with: url)
        task.taskDescription = file.filename
        task.resume()
        currentTask = task
        currentFileProgress = 0
        speedSampleTime = Date()
        speedSampleBytes = bytesDownloaded
    }

    // MARK: - Disk Space

    private func availableDiskSpace() -> UInt64 {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        guard let values = try? docs.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]),
              let capacity = values.volumeAvailableCapacityForImportantUsage else {
            return 0
        }
        return UInt64(capacity)
    }

    // MARK: - State Persistence

    private var stateFileURL: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("downloads.json")
    }

    private func persistState() {
        guard let state = activeDownload else { return }
        if let data = try? JSONEncoder().encode(state) {
            try? data.write(to: stateFileURL)
        }
    }

    private func clearPersistedState() {
        try? FileManager.default.removeItem(at: stateFileURL)
    }

    private func loadPersistedState() {
        guard let data = try? Data(contentsOf: stateFileURL),
              let state = try? JSONDecoder().decode(DownloadState.self, from: data) else {
            return
        }

        activeDownload = state
        currentEntry = ModelCatalog.models.first { $0.id == state.catalogId }

        if let entry = currentEntry {
            totalBytes = entry.totalSizeBytes
            bytesDownloaded = state.completedBytes
            overallProgress = totalBytes > 0 ? Double(bytesDownloaded) / Double(totalBytes) : 0
        }
    }

    // MARK: - Formatting

    private func formatBytes(_ bytes: UInt64) -> String {
        let gb = Double(bytes) / (1024 * 1024 * 1024)
        if gb >= 1 {
            return String(format: "%.1f GB", gb)
        }
        let mb = Double(bytes) / (1024 * 1024)
        return String(format: "%.0f MB", mb)
    }
}

// MARK: - URLSessionDownloadDelegate

extension DownloadManager: URLSessionDownloadDelegate {

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        guard var state = activeDownload,
              let entry = currentEntry,
              let filename = downloadTask.taskDescription else { return }

        // Move from temp to model directory
        let dest = modelDirectory(for: entry.id).appendingPathComponent(filename)
        let fm = FileManager.default
        try? fm.removeItem(at: dest) // remove partial if exists
        do {
            try fm.moveItem(at: location, to: dest)
        } catch {
            self.error = "Failed to save \(filename): \(error.localizedDescription)"
            state.status = .failed
            state.errorMessage = self.error
            activeDownload = state
            persistState()
            return
        }

        // Validate file size
        if let file = entry.files.first(where: { $0.filename == filename }) {
            let attrs = try? fm.attributesOfItem(atPath: dest.path)
            let actualSize = attrs?[.size] as? UInt64 ?? 0
            // Allow 1% tolerance for Content-Length vs actual
            if actualSize > 0 && file.sizeBytes > 0 && actualSize < file.sizeBytes * 9 / 10 {
                self.error = "File \(filename) is too small (\(actualSize) vs expected \(file.sizeBytes))"
                state.status = .failed
                state.errorMessage = self.error
                activeDownload = state
                persistState()
                return
            }
            state.completedBytes += actualSize > 0 ? actualSize : file.sizeBytes
        }

        state.completedFiles.append(filename)
        state.currentFile = nil
        activeDownload = state
        bytesDownloaded = state.completedBytes
        persistState()

        // Start next file
        downloadNextFile()
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        // Per-file progress
        if totalBytesExpectedToWrite > 0 {
            currentFileProgress = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
        }

        // Overall progress
        let completed = activeDownload?.completedBytes ?? 0
        let current = UInt64(totalBytesWritten)
        bytesDownloaded = completed + current
        if totalBytes > 0 {
            overallProgress = Double(bytesDownloaded) / Double(totalBytes)
        }

        // Download speed (sampled every 2 seconds)
        if let sampleTime = speedSampleTime, Date().timeIntervalSince(sampleTime) >= 2 {
            let elapsed = Date().timeIntervalSince(sampleTime)
            let delta = bytesDownloaded - speedSampleBytes
            downloadSpeed = Double(delta) / elapsed
            speedSampleTime = Date()
            speedSampleBytes = bytesDownloaded
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: (any Error)?) {
        guard let error else { return } // success handled in didFinishDownloadingTo

        let nsError = error as NSError
        if nsError.code == NSURLErrorCancelled {
            // User-initiated cancel or pause — resume data handled in pauseDownload()
            return
        }

        // Save resume data if available
        if let resumeData = nsError.userInfo[NSURLSessionDownloadTaskResumeData] as? Data {
            self.resumeData = resumeData
        }

        self.error = error.localizedDescription
        activeDownload?.status = .failed
        activeDownload?.errorMessage = error.localizedDescription
        persistState()
    }

    func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        DispatchQueue.main.async { [weak self] in
            self?.backgroundCompletionHandler?()
            self?.backgroundCompletionHandler = nil
        }
    }
}
