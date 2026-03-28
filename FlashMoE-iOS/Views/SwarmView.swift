/*
 * SwarmView.swift — Flash-Swarm cluster tab with Worker/Router mode
 *
 * Worker mode: join an existing cluster as an inference worker
 * Router mode: coordinate the cluster, discover workers, assign experts
 */

import SwiftUI
import UniformTypeIdentifiers

// MARK: - Mode

enum ClusterMode: String, CaseIterable {
    case worker = "Worker"
    case router = "Router"
}

// MARK: - SwarmView

struct SwarmView: View {
    @Environment(FlashMoEEngine.self) private var engine
    @Environment(SwarmWorker.self)    private var swarm
    @Environment(RouterManager.self)  private var router

    @State private var mode: ClusterMode = .worker
    @State private var networkKey: String   = "flashswarm"
    @State private var expertPort: String   = "10128"
    @State private var discoveryPort: String = "10127"
    @State private var showAdvanced = false
    @State private var routerModelPath: String = ""
    @State private var showModelPicker = false

    private var isActive: Bool {
        mode == .worker ? swarm.status.isActive : router.isRunning
    }

    var body: some View {
        NavigationStack {
            List {
                // ── Mode selector ─────────────────────────────────────
                Section {
                    Picker("Mode", selection: $mode) {
                        ForEach(ClusterMode.allCases, id: \.self) { m in
                            Text(m.rawValue).tag(m)
                        }
                    }
                    .pickerStyle(.segmented)
                    .disabled(isActive)
                }

                // ── Mode-specific content ─────────────────────────────
                if mode == .worker {
                    workerContent
                } else {
                    routerContent
                }

                // ── Network settings (shared) ─────────────────────────
                Section(header: Text("Network")) {
                    HStack {
                        Label("Key", systemImage: "key.fill")
                        Spacer()
                        TextField("flashswarm", text: $networkKey)
                            .multilineTextAlignment(.trailing)
                            .disabled(isActive)
                            .foregroundStyle(isActive ? .secondary : .primary)
                    }
                }

                // ── Advanced ──────────────────────────────────────────
                Section {
                    DisclosureGroup("Advanced", isExpanded: $showAdvanced) {
                        portRow("Handshake port", binding: $expertPort)
                        portRow("Discovery port", binding: $discoveryPort)
                    }
                }

                // ── Error ─────────────────────────────────────────────
                if let err = (mode == .worker ? swarm.lastError : router.lastError) {
                    Section {
                        Label(err, systemImage: "exclamationmark.triangle.fill")
                            .foregroundStyle(.red)
                            .font(.footnote)
                    }
                }
            }
            .navigationTitle(mode == .worker ? "Swarm Worker" : "Swarm Router")
            .onAppear {
                networkKey    = swarm.networkKey
                expertPort    = String(swarm.expertPort)
                discoveryPort = String(swarm.discoveryPort)
                if let path = engine.loadedModelPath, routerModelPath.isEmpty {
                    routerModelPath = path
                }
            }
        }
    }

    // MARK: - Worker mode

    @ViewBuilder
    private var workerContent: some View {
        // Status card
        Section {
            StatusCard(status: swarm.status,
                       tasksHandled: swarm.tasksHandled,
                       masterAddress: swarm.masterAddress)
        }

        // Join/Leave button
        Section {
            if swarm.status == .stopped {
                Button {
                    applySettings()
                    swarm.start(modelPath: engine.loadedModelPath ?? "")
                } label: {
                    Label("Join Swarm", systemImage: "antenna.radiowaves.left.and.right")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
            } else {
                Button(role: .destructive) {
                    swarm.stop()
                } label: {
                    Label("Leave Swarm", systemImage: "antenna.radiowaves.left.and.right.slash")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.large)
            }
        }
        .disabled(engine.state != .ready && engine.state != .generating)

        if engine.state == .idle || engine.state == .loading {
            Section {
                Label("Load a model first to join the swarm.",
                      systemImage: "info.circle")
                    .foregroundStyle(.secondary)
                    .font(.footnote)
            }
        }
    }

    // MARK: - Router mode

    @ViewBuilder
    private var routerContent: some View {
        // Router status
        Section {
            RouterStatusCard(
                statusCode: router.statusCode,
                statusText: router.statusText,
                masterExperts: router.masterExperts,
                totalExperts: router.totalExperts,
                unassigned: router.unassignedExperts,
                workerCount: router.workerCount
            )
        }

        // Model path (router reads config.json from disk — no GPU load needed)
        Section(header: Text("Model Directory")) {
            HStack {
                TextField("Path to model directory", text: $routerModelPath)
                    .disabled(router.isRunning)
                Button("Browse") {
                    showModelPicker = true
                }
                .disabled(router.isRunning)
            }
            if !routerModelPath.isEmpty {
                Text(routerModelPath)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
        }
        .fileImporter(
            isPresented: $showModelPicker,
            allowedContentTypes: [.folder],
            allowsMultipleSelection: false
        ) { result in
            if case .success(let urls) = result, let url = urls.first {
                routerModelPath = url.path
            }
        }

        // Start/Stop button
        Section {
            if !router.isRunning {
                Button {
                    applyRouterSettings()
                    // Use engine's loaded model path if available, otherwise use manual path
                    let path = engine.loadedModelPath ?? routerModelPath
                    router.start(modelPath: path)
                } label: {
                    Label("Start Router", systemImage: "server.rack")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(routerModelPath.isEmpty && engine.loadedModelPath == nil)
            } else {
                Button(role: .destructive) {
                    router.stop()
                } label: {
                    Label("Stop Router", systemImage: "stop.circle")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.large)
            }
        }

        // Worker list
        if !router.workers.isEmpty {
            Section(header: Text("Connected Workers (\(router.workerCount))")) {
                ForEach(router.workers) { worker in
                    WorkerRow(worker: worker)
                }
            }
        }

        if routerModelPath.isEmpty && engine.loadedModelPath == nil {
            Section {
                Label("Select a model directory or load a model in the Models tab. The router only reads config.json — no GPU memory needed.",
                      systemImage: "info.circle")
                    .foregroundStyle(.secondary)
                    .font(.footnote)
            }
        }

        // Log stream
        if router.isRunning || !router.logs.isEmpty {
            Section(header: HStack {
                Text("Logs")
                Spacer()
                Button("Copy") {
                    #if os(macOS)
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(router.logs.joined(separator: "\n"), forType: .string)
                    #else
                    UIPasteboard.general.string = router.logs.joined(separator: "\n")
                    #endif
                }
                .font(.caption)
            }) {
                ScrollViewReader { proxy in
                    ScrollView {
                        VStack(alignment: .leading, spacing: 1) {
                            ForEach(Array(router.logs.enumerated()), id: \.offset) { idx, line in
                                Text(line)
                                    .font(.system(size: 11, design: .monospaced))
                                    .foregroundStyle(.secondary)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .id(idx)
                            }
                        }
                        .padding(6)
                    }
                    .frame(height: 250)
                    .background(Color(.textBackgroundColor).opacity(0.3))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                    .onChange(of: router.logs.count) {
                        if let last = router.logs.indices.last {
                            proxy.scrollTo(last, anchor: .bottom)
                        }
                    }
                }
            }
        }
    }

    // MARK: - Helpers

    private func portRow(_ label: String, binding: Binding<String>) -> some View {
        HStack {
            Text(label)
            Spacer()
            TextField("port", text: binding)
                .multilineTextAlignment(.trailing)
                #if os(iOS)
                .keyboardType(.numberPad)
                #endif
                .frame(width: 80)
                .disabled(isActive)
                .foregroundStyle(isActive ? .secondary : .primary)
        }
    }

    private func applySettings() {
        swarm.networkKey     = networkKey
        swarm.expertPort     = UInt16(expertPort) ?? 10128
        swarm.discoveryPort  = UInt16(discoveryPort) ?? 10127
    }

    private func applyRouterSettings() {
        router.networkKey     = networkKey
        router.discoveryPort  = UInt16(discoveryPort) ?? 10127
        router.expertPort     = UInt16(expertPort) ?? 10128
    }
}

// MARK: - RouterStatusCard

private struct RouterStatusCard: View {
    let statusCode: Int32
    let statusText: String
    let masterExperts: Int
    let totalExperts: Int
    let unassigned: Int
    let workerCount: Int

    var body: some View {
        VStack(spacing: 12) {
            HStack {
                Circle()
                    .fill(statusColor)
                    .frame(width: 10, height: 10)
                Text(statusText)
                    .font(.headline)
                Spacer()
                if workerCount > 0 {
                    Label("\(workerCount) worker\(workerCount == 1 ? "" : "s")",
                          systemImage: "desktopcomputer")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            }

            if statusCode == 2 {
                HStack(spacing: 16) {
                    StatBadge(label: "Master", value: "\(masterExperts)")
                    StatBadge(label: "Total", value: "\(totalExperts)")
                    StatBadge(label: "Unassigned", value: "\(unassigned)")
                }
            }
        }
        .padding(.vertical, 4)
    }

    private var statusColor: Color {
        switch statusCode {
        case 0: return .gray
        case 1: return .yellow
        case 2: return .green
        case 3: return .red
        default: return .gray
        }
    }
}

// MARK: - StatBadge

private struct StatBadge: View {
    let label: String
    let value: String

    var body: some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.title3.bold())
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
    }
}

// MARK: - WorkerRow

private struct WorkerRow: View {
    let worker: ClusterWorkerInfo

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Image(systemName: "desktopcomputer")
                    .foregroundStyle(.tint)
                Text(worker.hostname)
                    .font(.headline)
                Spacer()
                Text("\(worker.experts_assigned) experts")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            HStack {
                Text(worker.gpu)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Text("\(worker.memory_gb) GB")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 2)
    }
}

// MARK: - StatusCard (worker mode)

private struct StatusCard: View {
    let status: SwarmStatus
    let tasksHandled: Int
    let masterAddress: String?

    var body: some View {
        VStack(spacing: 12) {
            HStack {
                Circle()
                    .fill(statusColor)
                    .frame(width: 10, height: 10)
                    .overlay(
                        Circle()
                            .stroke(statusColor.opacity(0.3), lineWidth: 4)
                            .scaleEffect(status == .advertising ? 1.4 : 1)
                            .animation(
                                status == .advertising
                                    ? .easeInOut(duration: 1).repeatForever(autoreverses: true)
                                    : .default,
                                value: status
                            )
                    )

                Text(status.label)
                    .font(.headline)

                Spacer()

                if tasksHandled > 0 {
                    Label("\(tasksHandled)", systemImage: "checkmark.circle.fill")
                        .font(.subheadline)
                        .foregroundStyle(.green)
                }
            }

            if let addr = masterAddress {
                HStack {
                    Image(systemName: "desktopcomputer")
                        .foregroundStyle(.secondary)
                    Text(addr)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                }
            }
        }
        .padding(.vertical, 4)
    }

    private var statusColor: Color {
        switch status {
        case .stopped:     return .gray
        case .advertising: return .yellow
        case .connected:   return .green
        case .processing:  return .blue
        }
    }
}
