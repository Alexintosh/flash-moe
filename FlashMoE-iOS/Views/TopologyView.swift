/*
 * TopologyView.swift — Radial network topology visualization
 *
 * Shows the cluster as a radial graph:
 * - Master node (this Mac) in the center
 * - Worker nodes arranged in a circle around it
 * - Lines connecting each worker to the master
 */

import SwiftUI

// MARK: - Node model for layout

struct TopologyNode: Identifiable {
    let id: String
    let label: String
    let detail: String
    let isMaster: Bool
    let isActive: Bool
    var position: CGPoint = .zero
}

// MARK: - TopologyView

struct TopologyView: View {
    let workers: [ClusterWorkerInfo]
    let totalExperts: Int
    let isRunning: Bool

    @State private var animationPhase: CGFloat = 0

    var body: some View {
        GeometryReader { geo in
            let size = geo.size
            let center = CGPoint(x: size.width / 2, y: size.height / 2)
            let radius = min(size.width, size.height) * 0.35
            let nodes = layoutNodes(center: center, radius: radius)

            Canvas { context, canvasSize in
                // Draw connection lines from each worker to master.
                let masterPos = center
                for node in nodes where !node.isMaster {
                    var path = Path()
                    path.move(to: masterPos)
                    path.addLine(to: node.position)
                    context.stroke(
                        path,
                        with: .color(node.isActive ? .green.opacity(0.5) : .gray.opacity(0.3)),
                        lineWidth: 2
                    )
                }

                // Draw nodes.
                for node in nodes {
                    let nodeRadius: CGFloat = node.isMaster ? 28 : 22
                    let rect = CGRect(
                        x: node.position.x - nodeRadius,
                        y: node.position.y - nodeRadius,
                        width: nodeRadius * 2,
                        height: nodeRadius * 2
                    )

                    // Filled circle.
                    let fillColor: Color = node.isMaster
                        ? .blue
                        : (node.isActive ? .green : .gray)
                    context.fill(Circle().path(in: rect), with: .color(fillColor.opacity(0.8)))

                    // Border.
                    context.stroke(
                        Circle().path(in: rect),
                        with: .color(fillColor),
                        lineWidth: 2
                    )

                    // Pulse ring for active workers.
                    if node.isActive && !node.isMaster {
                        let pulseRadius = nodeRadius + 4 + animationPhase * 6
                        let pulseRect = CGRect(
                            x: node.position.x - pulseRadius,
                            y: node.position.y - pulseRadius,
                            width: pulseRadius * 2,
                            height: pulseRadius * 2
                        )
                        context.stroke(
                            Circle().path(in: pulseRect),
                            with: .color(.green.opacity(0.3 * (1 - animationPhase))),
                            lineWidth: 1.5
                        )
                    }
                }
            }

            // Overlay labels on top of canvas.
            ForEach(nodes) { node in
                VStack(spacing: 1) {
                    Text(node.label)
                        .font(.system(size: 10, weight: .semibold, design: .rounded))
                        .foregroundStyle(.white)
                    Text(node.detail)
                        .font(.system(size: 8, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.7))
                }
                .position(x: node.position.x, y: node.position.y + (node.isMaster ? 42 : 34))
            }
        }
        .onAppear {
            withAnimation(.easeInOut(duration: 1.5).repeatForever(autoreverses: true)) {
                animationPhase = 1
            }
        }
    }

    // MARK: - Layout

    private func layoutNodes(center: CGPoint, radius: CGFloat) -> [TopologyNode] {
        var nodes: [TopologyNode] = []

        // Master node at center.
        #if os(macOS)
        let hostName = Host.current().localizedName ?? "This Mac"
        #else
        let hostName = UIDevice.current.name
        #endif
        nodes.append(TopologyNode(
            id: "master",
            label: hostName,
            detail: "\(totalExperts) experts",
            isMaster: true,
            isActive: isRunning,
            position: center
        ))

        // Worker nodes in a circle.
        let count = workers.count
        guard count > 0 else { return nodes }

        for (i, worker) in workers.enumerated() {
            let angle = (2 * .pi / Double(count)) * Double(i) - .pi / 2
            let x = center.x + radius * cos(angle)
            let y = center.y + radius * sin(angle)

            nodes.append(TopologyNode(
                id: worker.id,
                label: worker.displayName,
                detail: worker.capabilities?.first ?? "worker",
                isMaster: false,
                isActive: true,
                position: CGPoint(x: x, y: y)
            ))
        }

        return nodes
    }
}
