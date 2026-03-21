/*
 * FlashMoEApp.swift — Main app entry point
 *
 * Flash-MoE iOS: Run massive MoE models on iPhone.
 * Targets iOS 17+ (iPhone 15 Pro with 8GB RAM).
 */

import SwiftUI

@main
struct FlashMoEApp: App {
    @State private var engine = FlashMoEEngine()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(engine)
        }
    }
}
