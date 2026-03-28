/*
 * SwarmView.swift — Flash-Swarm worker settings and status tab
 *
 * Lets the user configure the network key, start/stop the swarm worker,
 * and see live connection status + task stats.
 *
 * The view requires:
 *   @Environment(FlashMoEEngine.self) — the loaded inference engine
 *   @Environment(SwarmWorker.self)    — the swarm worker instance
 */

import SwiftUI

// MARK: - SwarmView

struct SwarmView: View {
    @Environment(FlashMoEEngine.self) private var engine
    @Environment(SwarmWorker.self)    private var swarm

    // Editable fields (only active when worker is stopped)
    @State private var networkKey: String   = "flashswarm"
    @State private var expertPort: String   = "10128"
    @State private var discoveryPort: String = "10127"
    @State private var showAdvanced = false

    var body: some View {
        NavigationStack {
            List {
                // ── Status card ──────────────────────────────────────────
                Section {
                    StatusCard(status: swarm.status,
                               tasksHandled: swarm.tasksHandled,
                               masterAddress: swarm.masterAddress)
                }

                // ── Toggle ───────────────────────────────────────────────
                Section {
                    if swarm.status == .stopped {
                        startButton
                    } else {
                        stopButton
                    }
                }
                .disabled(engine.state != .ready && engine.state != .generating)

                // ── Network settings (editable when stopped) ─────────────
                Section(header: Text("Network")) {
                    HStack {
                        Label("Key", systemImage: "key.fill")
                        Spacer()
                        TextField("flashswarm", text: $networkKey)
                            .multilineTextAlignment(.trailing)
                            .disabled(swarm.status.isActive)
                            .foregroundStyle(swarm.status.isActive ? .secondary : .primary)
                    }
                }

                // ── Advanced ─────────────────────────────────────────────
                Section {
                    DisclosureGroup("Advanced", isExpanded: $showAdvanced) {
                        portRow("Handshake port", binding: $expertPort)
                        portRow("Discovery port", binding: $discoveryPort)
                        infoRow(
                            "Task server port",
                            value: taskPortLabel
                        )
                    }
                }

                // ── Error ────────────────────────────────────────────────
                if let err = swarm.lastError {
                    Section {
                        Label(err, systemImage: "exclamationmark.triangle.fill")
                            .foregroundStyle(.red)
                            .font(.footnote)
                    }
                }

                // ── Requirements hint ─────────────────────────────────────
                if engine.state == .idle || engine.state == .loading {
                    Section {
                        Label("Load a model first to join the swarm.",
                              systemImage: "info.circle")
                            .foregroundStyle(.secondary)
                            .font(.footnote)
                    }
                }

                // ── How it works ─────────────────────────────────────────
                Section(header: Text("How it works")) {
                    VStack(alignment: .leading, spacing: 8) {
                        BulletRow(icon: "wifi",
                                  text: "Your iPhone advertises on the LAN via UDP broadcast.")
                        BulletRow(icon: "network",
                                  text: "A Mac router (flashswarm serve --mode router) discovers it.")
                        BulletRow(icon: "cpu",
                                  text: "Inference requests are dispatched here and processed locally using Metal.")
                        BulletRow(icon: "lock.shield",
                                  text: "Authentication uses HMAC-SHA256 with the shared network key.")
                    }
                    .padding(.vertical, 4)
                }
            }
            .navigationTitle("Swarm Worker")
            .onAppear {
                // Sync fields from worker's current settings.
                networkKey    = swarm.networkKey
                expertPort    = String(swarm.expertPort)
                discoveryPort = String(swarm.discoveryPort)
            }
        }
    }

    // MARK: - Subviews

    private var startButton: some View {
        Button {
            applySettings()
            let path = engine.loadedModelPath ?? ""
            swarm.start(modelPath: path)
        } label: {
            Label("Join Swarm", systemImage: "antenna.radiowaves.left.and.right")
                .frame(maxWidth: .infinity)
        }
        .buttonStyle(.borderedProminent)
        .controlSize(.large)
    }

    private var stopButton: some View {
        Button(role: .destructive) {
            swarm.stop()
        } label: {
            Label("Leave Swarm", systemImage: "antenna.radiowaves.left.and.right.slash")
                .frame(maxWidth: .infinity)
        }
        .buttonStyle(.bordered)
        .controlSize(.large)
    }

    private func portRow(_ label: String, binding: Binding<String>) -> some View {
        HStack {
            Text(label)
            Spacer()
            TextField("port", text: binding)
                .multilineTextAlignment(.trailing)
                .keyboardType(.numberPad)
                .frame(width: 80)
                .disabled(swarm.status.isActive)
                .foregroundStyle(swarm.status.isActive ? .secondary : .primary)
        }
    }

    private func infoRow(_ label: String, value: String) -> some View {
        HStack {
            Text(label)
            Spacer()
            Text(value).foregroundStyle(.secondary)
        }
    }

    private var taskPortLabel: String {
        let base = UInt16(expertPort) ?? 10128
        return String(base + 3)
    }

    private func applySettings() {
        swarm.networkKey     = networkKey
        swarm.expertPort     = UInt16(expertPort) ?? 10128
        swarm.discoveryPort  = UInt16(discoveryPort) ?? 10127
    }
}

// MARK: - StatusCard

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

// MARK: - BulletRow

private struct BulletRow: View {
    let icon: String
    let text: String

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: icon)
                .foregroundStyle(.tint)
                .frame(width: 20)
            Text(text)
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
    }
}
