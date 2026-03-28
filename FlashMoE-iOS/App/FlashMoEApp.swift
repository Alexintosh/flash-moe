/*
 * FlashMoEApp.swift — Main app entry point
 *
 * Flash-MoE iOS: Run massive MoE models on iPhone.
 * Targets iOS 18+ (iPhone 15 Pro and later with 8GB+ RAM).
 *
 * Injects two environment objects into the view hierarchy:
 *   FlashMoEEngine — Metal inference engine
 *   SwarmWorker    — Flash-Swarm cluster worker
 */

import SwiftUI

@main
struct FlashMoEApp: App {
#if os(iOS)
    @UIApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
#endif
    @State private var engine: FlashMoEEngine = .init()

    var body: some Scene {
        WindowGroup {
            RootView(engine: engine)
#if os(macOS)
                .frame(minWidth: 900, minHeight: 600)
#endif
        }
    }
}

/// Intermediate view that creates SwarmWorker after engine is available.
/// Using a separate view ensures SwarmWorker is initialised once per session.
private struct RootView: View {
    let engine: FlashMoEEngine
    @State private var swarm: SwarmWorker

    init(engine: FlashMoEEngine) {
        self.engine = engine
        _swarm = State(wrappedValue: SwarmWorker(engine: engine))
    }

    var body: some View {
        ContentView()
            .environment(engine)
            .environment(swarm)
    }
}
