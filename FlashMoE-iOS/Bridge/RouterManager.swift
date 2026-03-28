/*
 * RouterManager.swift — Observable Swift wrapper for flashswarm router mode
 *
 * Manages the master/coordinator: discovers workers, assigns experts,
 * monitors heartbeats. Parallel to SwarmWorker.swift (worker mode).
 */

import Foundation
import Observation

// MARK: - Data types

/// Info about a connected worker from the scheduler snapshot.
struct ClusterWorkerInfo: Identifiable, Codable {
    let node_id: String      // hex, e.g. "0A1B2C3D4E5F6789"
    let addr: String          // "192.168.1.67:10128"
    let capabilities: [String]?
    let active_tasks: Int?

    var id: String { node_id }

    /// Short display name derived from the address.
    var displayName: String {
        if addr.starts(with: "127.0.0.1") { return "This Mac (local)" }
        return addr
    }
}

/// Router status decoded from Rust JSON callback.
struct RouterStatusInfo: Codable {
    let state: String
    let total_experts: Int?
    let unassigned: Int?
    let worker_count: Int?
    let workers_alive: Int?
    let workers_departed: Int?
    let uptime_secs: Int?
    let workers: [ClusterWorkerInfo]?
    let error: String?
    let logs: [String]?
}

// MARK: - RouterManager

@Observable
final class RouterManager: @unchecked Sendable {

    // Published state for SwiftUI
    private(set) var isRunning = false
    private(set) var statusCode: Int32 = 0
    private(set) var totalExperts: Int = 0
    private(set) var unassignedExperts: Int = 0
    private(set) var workerCount: Int = 0
    private(set) var workersAlive: Int = 0
    private(set) var workersDeparted: Int = 0
    private(set) var uptimeSecs: Int = 0
    private(set) var workers: [ClusterWorkerInfo] = []
    private(set) var lastError: String?
    private(set) var logs: [String] = []

    var statusText: String {
        switch statusCode {
        case 0: return "Stopped"
        case 1: return "Starting..."
        case 2: return "Running"
        case 3: return "Error"
        default: return "Unknown"
        }
    }

    // Settings
    var networkKey: String = "flashswarm"
    var directPeers: String = ""  // Comma-separated, e.g. "172.20.10.1:10128"
    var discoveryPort: UInt16 = 10127
    var expertPort: UInt16 = 10128

    init() {
        // Register status callback (pre-init, stored until router_init is called).
        let ud = Unmanaged.passUnretained(self).toOpaque()
        flashswarm_ios_router_set_status_callback(
            { statusCode, infoJson, userdata in
                guard let userdata else { return }
                let mgr = Unmanaged<RouterManager>.fromOpaque(userdata)
                    .takeUnretainedValue()
                let info = infoJson.flatMap { String(cString: $0) } ?? "{}"
                mgr.handleStatusChange(code: statusCode, info: info)
            },
            ud
        )
    }

    deinit {
        flashswarm_ios_router_set_status_callback(nil, nil)
    }

    // MARK: - Lifecycle

    /// Start the router with the given model path.
    func start(modelPath: String) {
        guard !isRunning else { return }

        let pathPtr  = (modelPath as NSString).utf8String
        let keyPtr   = (networkKey as NSString).utf8String
        let peersPtr = directPeers.isEmpty ? nil : (directPeers as NSString).utf8String

        let rc = flashswarm_ios_router_init(pathPtr, keyPtr, peersPtr, discoveryPort, expertPort)
        if rc != 0 {
            DispatchQueue.main.async { self.lastError = "Router init failed" }
            return
        }

        let rc2 = flashswarm_ios_router_start()
        if rc2 != 0 {
            DispatchQueue.main.async { self.lastError = "Router start failed" }
            return
        }

        DispatchQueue.main.async {
            self.isRunning = true
            self.lastError = nil
        }
    }

    /// Stop the router.
    func stop() {
        guard isRunning else { return }
        flashswarm_ios_router_stop()
        DispatchQueue.main.async {
            self.isRunning = false
            self.workerCount = 0
            self.workersAlive = 0
            self.workers = []
            self.statusCode = 0
        }
    }

    // MARK: - Status callback (called from Rust background thread)

    private func handleStatusChange(code: Int32, info: String) {
        guard let data = info.data(using: .utf8),
              let status = try? JSONDecoder().decode(RouterStatusInfo.self, from: data)
        else {
            DispatchQueue.main.async { self.statusCode = code }
            return
        }

        DispatchQueue.main.async {
            self.statusCode = code
            self.totalExperts = status.total_experts ?? self.totalExperts
            self.unassignedExperts = status.unassigned ?? self.unassignedExperts
            self.workerCount = status.worker_count ?? self.workerCount
            self.workersAlive = status.workers_alive ?? self.workersAlive
            self.workersDeparted = status.workers_departed ?? self.workersDeparted
            self.uptimeSecs = status.uptime_secs ?? self.uptimeSecs
            self.workers = status.workers ?? self.workers
            if let err = status.error {
                self.lastError = err
            }
            if let newLogs = status.logs {
                self.logs = newLogs
            }
        }
    }
}
