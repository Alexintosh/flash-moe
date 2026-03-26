/*
 * BenchmarkView.swift -- Built-in benchmark mode for Flash-MoE iOS
 *
 * Runs a test matrix of prompts x settings configurations,
 * logs structured results (tab-separated), and provides
 * copy-to-clipboard for analysis.
 */

import SwiftUI

// MARK: - Benchmark Data Types

struct BenchmarkPrompt: Identifiable {
    let id: String
    let label: String
    let text: String

    static let builtIn: [BenchmarkPrompt] = [
        BenchmarkPrompt(id: "short", label: "Short", text: "Hi"),
        BenchmarkPrompt(id: "medium", label: "Medium", text: "What is an SSD and how does it work?"),
        BenchmarkPrompt(id: "long", label: "Long", text: "Explain the differences between CPU, GPU, and TPU architectures. Compare their memory hierarchies, parallelism models, and ideal workloads. Include specific examples."),
        BenchmarkPrompt(id: "json", label: "JSON", text: "Respond with a JSON object containing: name, age, hobbies (array of 3), and address (nested object with street, city, country)"),
        BenchmarkPrompt(id: "code", label: "Code", text: "Write a Python function that implements binary search on a sorted array. Include docstring and type hints."),
    ]
}

struct BenchmarkConfig: Identifiable {
    let id: String
    let name: String
    let activeExpertsK: Int
    let cmdMerge: Bool
    let fusedAttention: Bool
    let cacheIOSplit: Int
    let fp16Accumulation: Bool

    static let builtIn: [BenchmarkConfig] = [
        BenchmarkConfig(id: "baseline", name: "Baseline K=8", activeExpertsK: 8, cmdMerge: false, fusedAttention: false, cacheIOSplit: 1, fp16Accumulation: false),
        BenchmarkConfig(id: "k4_merge", name: "K=4+Merge", activeExpertsK: 4, cmdMerge: true, fusedAttention: false, cacheIOSplit: 1, fp16Accumulation: false),
        BenchmarkConfig(id: "k4_merge_fused", name: "K=4+Merge+Fused F2", activeExpertsK: 4, cmdMerge: true, fusedAttention: true, cacheIOSplit: 2, fp16Accumulation: false),
        BenchmarkConfig(id: "k4_merge_fused_fp16", name: "K=4+Merge+Fused+FP16", activeExpertsK: 4, cmdMerge: true, fusedAttention: true, cacheIOSplit: 2, fp16Accumulation: true),
        BenchmarkConfig(id: "k3_merge_fused_fp16", name: "K=3+Merge+Fused+FP16", activeExpertsK: 3, cmdMerge: true, fusedAttention: true, cacheIOSplit: 2, fp16Accumulation: true),
    ]
}

struct BenchmarkResult: Identifiable {
    let id = UUID()
    let configName: String
    let promptLabel: String
    let promptText: String
    let prefillTokens: Int
    let ttftMs: Double
    let tokensGenerated: Int
    let decodeTokPerSec: Double   // actual decode speed (tokens after first / decode time)
    let engineTokPerSec: Double   // engine-reported tok/s
    let totalMs: Double
    let outputSnippet: String
    let quality: String
    let thermalState: String

    var tsvLine: String {
        let truncatedPrompt = promptText.count > 50
            ? String(promptText.prefix(50)) + "..."
            : promptText
        let cleanSnippet = outputSnippet
            .replacingOccurrences(of: "\t", with: " ")
            .replacingOccurrences(of: "\n", with: " ")
        return "\(configName)\t\(truncatedPrompt)\t\(prefillTokens)\t\(String(format: "%.0f", ttftMs))\t\(tokensGenerated)\t\(String(format: "%.1f", decodeTokPerSec))\t\(String(format: "%.1f", engineTokPerSec))\t\(String(format: "%.0f", totalMs))\t\(thermalState)\t\(quality)\t\(cleanSnippet)"
    }
}

// MARK: - Benchmark Runner (Observable)

@Observable @MainActor
final class BenchmarkRunner {
    var isRunning = false
    var isCancelled = false
    var currentConfigIndex = 0
    var currentPromptIndex = 0
    var totalConfigs: Int = BenchmarkConfig.builtIn.count
    var totalPrompts: Int = BenchmarkPrompt.builtIn.count
    var results: [BenchmarkResult] = []
    var statusMessage = "Ready"
    var progressDetail = ""

    let maxTokensPerRun = 50
    let cooldownBetweenPrompts: UInt64 = 5_000_000_000  // 5 seconds
    let cooldownBetweenConfigs: UInt64 = 10_000_000_000 // 10 seconds

    func cancel() {
        isCancelled = true
        statusMessage = "Cancelling..."
    }

    private func thermalStateString() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    private func waitForCooldown(duration: UInt64, label: String) async {
        let thermal = thermalStateString()
        if thermal == "serious" || thermal == "critical" {
            progressDetail = "Thermal \(thermal) — cooling down (30s)..."
            try? await Task.sleep(nanoseconds: 30_000_000_000)
        } else {
            progressDetail = "\(label) cooldown (\(duration / 1_000_000_000)s)..."
            try? await Task.sleep(nanoseconds: duration)
        }
    }

    func run(engine: FlashMoEEngine, modelPath: String,
             configIds: Set<String>? = nil, promptIds: Set<String>? = nil) async {
        isRunning = true
        isCancelled = false
        results = []
        currentConfigIndex = 0
        currentPromptIndex = 0

        let configs = configIds != nil
            ? BenchmarkConfig.builtIn.filter { configIds!.contains($0.id) }
            : BenchmarkConfig.builtIn
        let prompts = promptIds != nil
            ? BenchmarkPrompt.builtIn.filter { promptIds!.contains($0.id) }
            : BenchmarkPrompt.builtIn

        totalConfigs = configs.count
        totalPrompts = prompts.count
        statusMessage = "Starting benchmark (\(configs.count)×\(prompts.count))..."

        for (ci, config) in configs.enumerated() {
            if isCancelled { break }

            currentConfigIndex = ci
            statusMessage = "Config \(ci + 1)/\(configs.count): \(config.name)"
            progressDetail = "Applying settings..."

            // Apply config by setting C globals directly — NO model reload needed.
            // This keeps the fullScreenCover alive and avoids 5-10s reload per config.
            engine.applyBenchmarkConfig(
                activeExpertsK: config.activeExpertsK,
                cmdMerge: config.cmdMerge,
                fusedAttention: config.fusedAttention,
                cacheIOSplit: config.cacheIOSplit,
                fp16Accumulation: config.fp16Accumulation
            )

            // Run each prompt with this config
            for (pi, prompt) in prompts.enumerated() {
                if isCancelled { break }

                currentPromptIndex = pi
                statusMessage = "Config \(ci + 1)/\(configs.count), Prompt \(pi + 1)/\(prompts.count)"
                progressDetail = "\(config.name) | \(prompt.label)"

                // Reset conversation state between prompts
                engine.reset()

                let result = await runSingleBenchmark(
                    engine: engine,
                    config: config,
                    prompt: prompt
                )

                results.append(result)

                // Cooldown between prompts
                if pi < prompts.count - 1 && !isCancelled {
                    await waitForCooldown(duration: cooldownBetweenPrompts, label: "Prompt")
                }
            }

            // Cooldown between configs
            if ci < configs.count - 1 && !isCancelled {
                await waitForCooldown(duration: cooldownBetweenConfigs, label: "Config")
            }
        }

        isRunning = false
        statusMessage = isCancelled ? "Cancelled" : "Complete (\(results.count) runs)"
        progressDetail = ""
    }

    private func runSingleBenchmark(
        engine: FlashMoEEngine,
        config: BenchmarkConfig,
        prompt: BenchmarkPrompt
    ) async -> BenchmarkResult {
        // Build Qwen chat template (no thinking)
        let formatted = "<|im_start|>system\nYou are a helpful assistant. /no_think<|im_end|>\n<|im_start|>user\n\(prompt.text)<|im_end|>\n<|im_start|>assistant\n<think>\n"

        let thermal = thermalStateString()
        let startTime = CFAbsoluteTimeGetCurrent()
        var firstTokenTime: Double = 0
        var output = ""
        var tokenCount = 0
        var lastEngTokPerSec: Double = 0
        var gotFirstToken = false
        var inThink = false

        let stream = engine.generate(prompt: formatted, maxTokens: maxTokensPerRun)

        for await token in stream {
            if !gotFirstToken {
                firstTokenTime = CFAbsoluteTimeGetCurrent()
                gotFirstToken = true
            }
            tokenCount += 1
            lastEngTokPerSec = token.tokensPerSecond

            // Clean special tokens + strip thinking
            var clean = token.text
                .replacingOccurrences(of: "<|im_end|>", with: "")
                .replacingOccurrences(of: "<|im_start|>", with: "")
                .replacingOccurrences(of: "<|endoftext|>", with: "")
            if clean.contains("<think>") { inThink = true; clean = clean.replacingOccurrences(of: "<think>", with: "") }
            if clean.contains("</think>") { inThink = false; clean = clean.replacingOccurrences(of: "</think>", with: "") }
            if !inThink { output += clean }
        }

        let endTime = CFAbsoluteTimeGetCurrent()
        let totalMs = (endTime - startTime) * 1000
        let ttftMs = gotFirstToken ? (firstTokenTime - startTime) * 1000 : 0

        // Calculate actual decode tok/s (exclude first token = prefill)
        let decodeTokens = max(tokenCount - 1, 0)
        let decodeTimeMs = gotFirstToken ? (endTime - firstTokenTime) * 1000 : 0
        let decodeTokPerSec = decodeTimeMs > 0 ? Double(decodeTokens) / (decodeTimeMs / 1000.0) : 0

        // Quality check
        let hasGibberish = output.contains("!!!!") || output.contains("????")
        let isEmpty = output.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        let quality = hasGibberish ? "GIBBERISH" : isEmpty ? "EMPTY" : "OK"

        // Estimate prefill tokens from TTFT log (rough)
        let wordCount = prompt.text.split(separator: " ").count
        let estimatedPrefillTokens = wordCount + 15  // +15 for chat template + system prompt tokens

        let snippet = String(output.prefix(100))

        return BenchmarkResult(
            configName: config.name,
            promptLabel: prompt.label,
            promptText: prompt.text,
            prefillTokens: estimatedPrefillTokens,
            ttftMs: ttftMs,
            tokensGenerated: tokenCount,
            decodeTokPerSec: decodeTokPerSec,
            engineTokPerSec: lastEngTokPerSec,
            totalMs: totalMs,
            outputSnippet: snippet,
            quality: quality,
            thermalState: thermal
        )
    }

    var fullTSV: String {
        var lines = "Config\tPrompt\tPrefillToks\tTTFT_ms\tGenToks\tDecodeTok/s\tEngTok/s\tTotalMs\tThermal\tQuality\tOutput\n"
        for r in results {
            lines += r.tsvLine + "\n"
        }
        return lines
    }
}

// MARK: - Benchmark View

struct BenchmarkView: View {
    @Environment(FlashMoEEngine.self) private var engine
    let modelPath: String

    @State private var runner = BenchmarkRunner()
    @State private var selectedConfigs: Set<String> = Set(BenchmarkConfig.builtIn.map(\.id))
    @State private var selectedPrompts: Set<String> = Set(BenchmarkPrompt.builtIn.map(\.id))
    @State private var showSetup = true

    var body: some View {
        VStack(spacing: 0) {
            // Status header
            statusHeader

            Divider()

            // Results
            if runner.results.isEmpty && !runner.isRunning {
                emptyState
            } else {
                resultsList
            }
        }
        .navigationTitle("Benchmark")
#if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
#endif
        .toolbar {
            ToolbarItem(placement: .confirmationAction) {
                if runner.isRunning {
                    Button("Stop") {
                        engine.cancel()
                        runner.cancel()
                    }
                    .foregroundStyle(.red)
                } else {
                    Button("Copy Results") {
                        copyResults()
                    }
                    .disabled(runner.results.isEmpty)
                }
            }
        }
    }

    // MARK: - Status Header

    private var statusHeader: some View {
        VStack(spacing: 12) {
            if runner.isRunning {
                VStack(spacing: 6) {
                    ProgressView(
                        value: Double(runner.currentConfigIndex * runner.totalPrompts + runner.currentPromptIndex),
                        total: Double(runner.totalConfigs * runner.totalPrompts)
                    )

                    Text(runner.statusMessage)
                        .font(.headline)

                    if !runner.progressDetail.isEmpty {
                        Text(runner.progressDetail)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .padding()
            } else {
                VStack(spacing: 8) {
                    Text(runner.statusMessage)
                        .font(.headline)

                    // Toggle setup panel
                    Button {
                        withAnimation { showSetup.toggle() }
                    } label: {
                        HStack {
                            Text("\(selectedConfigs.count) configs × \(selectedPrompts.count) prompts = \(selectedConfigs.count * selectedPrompts.count) runs")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Image(systemName: showSetup ? "chevron.up" : "chevron.down")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                    .buttonStyle(.plain)

                    if showSetup {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Configs").font(.caption).fontWeight(.bold)
                            ForEach(BenchmarkConfig.builtIn) { cfg in
                                Toggle(cfg.name, isOn: Binding(
                                    get: { selectedConfigs.contains(cfg.id) },
                                    set: { if $0 { selectedConfigs.insert(cfg.id) } else { selectedConfigs.remove(cfg.id) } }
                                ))
                                .font(.caption)
                                .toggleStyle(.switch)
                            }

                            Divider()

                            Text("Prompts").font(.caption).fontWeight(.bold)
                            ForEach(BenchmarkPrompt.builtIn) { p in
                                Toggle(p.label, isOn: Binding(
                                    get: { selectedPrompts.contains(p.id) },
                                    set: { if $0 { selectedPrompts.insert(p.id) } else { selectedPrompts.remove(p.id) } }
                                ))
                                .font(.caption)
                                .toggleStyle(.switch)
                            }
                        }
                        .padding(.horizontal)
                    }

                    Button {
                        showSetup = false
                        Task {
                            await runner.run(engine: engine, modelPath: modelPath,
                                           configIds: selectedConfigs, promptIds: selectedPrompts)
                        }
                    } label: {
                        Label("Run Benchmark", systemImage: "play.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.orange)
                    .disabled((engine.state != .ready && !runner.isRunning) || selectedConfigs.isEmpty || selectedPrompts.isEmpty)
                }
                .padding()
            }
        }
        .background(.ultraThinMaterial)
    }

    // MARK: - Empty State

    private var emptyState: some View {
        VStack(spacing: 16) {
            Spacer()
            Image(systemName: "gauge.with.dots.needle.50percent")
                .font(.system(size: 48))
                .foregroundStyle(.secondary)
            Text("Tap Run Benchmark to start")
                .font(.headline)
                .foregroundStyle(.secondary)
            Text("Tests 5 prompts across 5 configurations.\nModel reloads between config changes.")
                .font(.caption)
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
            Spacer()
        }
        .padding()
    }

    // MARK: - Results List

    private var resultsList: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 2) {
                // Header row
                Text("Config\tPrompt\tPrefill\tTTFT\tToks\ttok/s\tTotal\tQual\tOutput")
                    .font(.system(.caption2, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 8)
                    .padding(.top, 8)

                ForEach(runner.results) { result in
                    resultRow(result)
                }
            }
        }
    }

    private func resultRow(_ result: BenchmarkResult) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 8) {
                Text(result.configName)
                    .font(.system(.caption, design: .monospaced))
                    .fontWeight(.semibold)
                    .lineLimit(1)

                Text(result.promptLabel)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)

                Spacer()

                Text(result.quality)
                    .font(.system(.caption2, design: .monospaced))
                    .padding(.horizontal, 4)
                    .padding(.vertical, 1)
                    .background(result.quality == "OK" ? Color.green.opacity(0.2) : Color.red.opacity(0.2))
                    .clipShape(RoundedRectangle(cornerRadius: 3))
            }

            HStack(spacing: 12) {
                metricLabel("TTFT", String(format: "%.0fms", result.ttftMs))
                metricLabel("Toks", "\(result.tokensGenerated)")
                metricLabel("dec tok/s", String(format: "%.1f", result.decodeTokPerSec))
                metricLabel("eng tok/s", String(format: "%.1f", result.engineTokPerSec))
                metricLabel("Total", String(format: "%.0fms", result.totalMs))
            }

            if !result.outputSnippet.isEmpty {
                Text(result.outputSnippet)
                    .font(.system(.caption2, design: .monospaced))
                    .foregroundStyle(.tertiary)
                    .lineLimit(2)
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(
            runner.results.firstIndex(where: { $0.id == result.id }).map { $0 % 2 == 0 } == true
            ? Color.clear : Color.secondary.opacity(0.05)
        )
    }

    private func metricLabel(_ label: String, _ value: String) -> some View {
        HStack(spacing: 2) {
            Text(label)
                .font(.system(.caption2, design: .monospaced))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.system(.caption2, design: .monospaced))
                .fontWeight(.medium)
        }
    }

    // MARK: - Actions

    private func copyResults() {
#if os(iOS)
        UIPasteboard.general.string = runner.fullTSV
#elseif os(macOS)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(runner.fullTSV, forType: .string)
#endif
    }
}
