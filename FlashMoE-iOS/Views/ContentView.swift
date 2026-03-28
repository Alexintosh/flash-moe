/*
 * ContentView.swift — Root navigation view
 *
 * Tab bar with Chat, Models, and Swarm tabs.
 * The Swarm tab is always visible; joining the cluster requires a loaded model.
 */

import SwiftUI

struct ContentView: View {
    @Environment(FlashMoEEngine.self) private var engine
    @Environment(SwarmWorker.self)    private var swarm

    var body: some View {
        TabView {
            // ── Chat ──────────────────────────────────────────────────────
            Tab("Chat", systemImage: "bubble.left.and.bubble.right") {
                NavigationStack {
                    switch engine.state {
                    case .idle, .loading, .error:
                        ModelListView()
                    case .ready, .generating:
                        ChatView()
                    }
                }
            }

            // ── Models ────────────────────────────────────────────────────
            Tab("Models", systemImage: "cpu") {
                ModelListView()
            }

            // ── Swarm ─────────────────────────────────────────────────────
            Tab("Swarm", systemImage: "antenna.radiowaves.left.and.right") {
                SwarmView()
            }
            .badge(swarm.status == .processing ? "●" : nil)
        }
    }
}
