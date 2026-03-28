/*
 * FlashMoEApp.swift — Main app entry point
 *
 * Flash-MoE: Run massive MoE models on iPhone and Mac.
 *
 * Injects three environment objects into the view hierarchy:
 *   FlashMoEEngine — Metal inference engine
 *   SwarmWorker    — Flash-Swarm cluster worker
 *   RouterManager  — Flash-Swarm cluster router/coordinator
 *
 * On macOS, a MenuBarExtra provides quick cluster status in the menu bar.
 */

import SwiftUI

@main
struct FlashMoEApp: App {
#if os(iOS)
    @UIApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
#endif

    // Shared state — accessible by both WindowGroup and MenuBarExtra.
    @State private var engine = FlashMoEEngine()
    @State private var swarm: SwarmWorker
    @State private var router = RouterManager()

    init() {
        let eng = FlashMoEEngine()
        _engine = State(wrappedValue: eng)
        _swarm = State(wrappedValue: SwarmWorker(engine: eng))
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(engine)
                .environment(swarm)
                .environment(router)
#if os(macOS)
                .frame(minWidth: 900, minHeight: 600)
#endif
        }

#if os(macOS)
        MenuBarExtra {
            MenuBarClusterView(router: router, engine: engine)
        } label: {
            Image(systemName: menuBarIcon)
        }
        .menuBarExtraStyle(.menu)
#endif
    }

#if os(macOS)
    private var menuBarIcon: String {
        if router.isRunning {
            return router.workerCount > 0
                ? "antenna.radiowaves.left.and.right"
                : "antenna.radiowaves.left.and.right.circle"
        }
        return "antenna.radiowaves.left.and.right.slash"
    }
#endif
}

// MARK: - Menu bar dropdown (macOS only)

#if os(macOS)
struct MenuBarClusterView: View {
    let router: RouterManager
    let engine: FlashMoEEngine

    var body: some View {
        if router.isRunning {
            Text("Router: Running")
                .font(.headline)
            Divider()
            Text("Workers: \(router.workerCount)")
            Text("Alive: \(router.workersAlive)")
            if router.workersDeparted > 0 {
                Text("Departed: \(router.workersDeparted)")
            }
            Text("Experts: \(router.totalExperts)")
            Text("Unassigned: \(router.unassignedExperts)")
            Text("Uptime: \(formatUptime(router.uptimeSecs))")
            Divider()
            ForEach(router.workers) { w in
                Text("\(w.displayName) (\(w.active_tasks ?? 0) active)")
            }
        } else {
            Text("Router: Stopped")
                .foregroundStyle(.secondary)
        }

        Divider()

        Button("Open Flash-MoE") {
            NSApplication.shared.activate(ignoringOtherApps: true)
            if let window = NSApplication.shared.windows.first {
                window.makeKeyAndOrderFront(nil)
            }
        }
        .keyboardShortcut("o")

        Button("Quit") {
            NSApplication.shared.terminate(nil)
        }
        .keyboardShortcut("q")
    }

    private func formatUptime(_ secs: Int) -> String {
        if secs < 60 { return "\(secs)s" }
        if secs < 3600 { return "\(secs / 60)m" }
        return "\(secs / 3600)h \((secs % 3600) / 60)m"
    }
}
#endif
