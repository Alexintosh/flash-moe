/*
 * ChatView.swift — Chat interface for Flash-MoE inference
 *
 * Streaming token display, stats overlay, conversation history.
 */

import SwiftUI

// MARK: - Chat Message Model

struct ChatMessage: Identifiable {
    let id = UUID()
    let role: Role
    var text: String
    let timestamp: Date

    enum Role {
        case user
        case assistant
    }
}

// MARK: - Chat View

struct ChatView: View {
    @Environment(FlashMoEEngine.self) private var engine
    @State private var messages: [ChatMessage] = []
    @State private var inputText = ""
    @State private var isGenerating = false
    @State private var showStats = false
    @State private var showModelInfo = false
    @FocusState private var inputFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            // Messages
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        ForEach(messages) { message in
                            MessageBubble(message: message)
                                .id(message.id)
                        }
                    }
                    .padding()
                }
                .onChange(of: messages.count) {
                    if let last = messages.last {
                        withAnimation {
                            proxy.scrollTo(last.id, anchor: .bottom)
                        }
                    }
                }
            }

            // Stats bar
            if isGenerating || engine.tokensGenerated > 0 {
                StatsBar(
                    tokensPerSecond: engine.tokensPerSecond,
                    tokensGenerated: engine.tokensGenerated,
                    isGenerating: isGenerating
                )
            }

            Divider()

            // Input bar
            HStack(spacing: 12) {
                TextField("Message...", text: $inputText, axis: .vertical)
                    .textFieldStyle(.plain)
                    .lineLimit(1...5)
                    .focused($inputFocused)
                    .onSubmit { sendMessage() }
                    .disabled(isGenerating)

                if isGenerating {
                    Button(action: { engine.cancel() }) {
                        Image(systemName: "stop.circle.fill")
                            .font(.title2)
                            .foregroundStyle(.red)
                    }
                } else {
                    Button(action: sendMessage) {
                        Image(systemName: "arrow.up.circle.fill")
                            .font(.title2)
                            .foregroundStyle(inputText.isEmpty ? .gray : .blue)
                    }
                    .disabled(inputText.isEmpty)
                }
            }
            .padding(.horizontal)
            .padding(.vertical, 8)
        }
        .navigationTitle("Flash-MoE")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarLeading) {
                Button(action: { showModelInfo = true }) {
                    Image(systemName: "cpu")
                }
            }
            ToolbarItem(placement: .topBarTrailing) {
                Menu {
                    Button("New Chat", systemImage: "plus.message") {
                        messages.removeAll()
                        engine.reset()
                    }
                    Button("Show Stats", systemImage: "chart.bar") {
                        showStats.toggle()
                    }
                } label: {
                    Image(systemName: "ellipsis.circle")
                }
            }
        }
        .sheet(isPresented: $showModelInfo) {
            ModelInfoSheet(info: engine.modelInfo)
        }
    }

    private func sendMessage() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }

        inputText = ""
        let userMessage = ChatMessage(role: .user, text: text, timestamp: Date())
        messages.append(userMessage)

        // Start generation
        isGenerating = true
        let assistantMessage = ChatMessage(role: .assistant, text: "", timestamp: Date())
        messages.append(assistantMessage)
        let assistantIndex = messages.count - 1

        Task {
            let stream = engine.generate(prompt: text, maxTokens: 500)
            for await token in stream {
                messages[assistantIndex].text += token.text
            }
            isGenerating = false
        }
    }
}

// MARK: - Message Bubble

struct MessageBubble: View {
    let message: ChatMessage

    var body: some View {
        HStack {
            if message.role == .user { Spacer(minLength: 60) }

            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 4) {
                Text(message.text)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(message.role == .user ? Color.blue : Color(.systemGray5))
                    .foregroundStyle(message.role == .user ? .white : .primary)
                    .clipShape(RoundedRectangle(cornerRadius: 18))

                if message.text.isEmpty {
                    ProgressView()
                        .padding(.leading, 8)
                }
            }

            if message.role == .assistant { Spacer(minLength: 60) }
        }
    }
}

// MARK: - Stats Bar

struct StatsBar: View {
    let tokensPerSecond: Double
    let tokensGenerated: Int
    let isGenerating: Bool

    var body: some View {
        HStack(spacing: 16) {
            Label(String(format: "%.1f tok/s", tokensPerSecond), systemImage: "speedometer")
                .font(.caption)
                .foregroundStyle(.secondary)

            Label("\(tokensGenerated) tokens", systemImage: "number")
                .font(.caption)
                .foregroundStyle(.secondary)

            Spacer()

            if isGenerating {
                ProgressView()
                    .scaleEffect(0.7)
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 6)
        .background(.ultraThinMaterial)
    }
}

// MARK: - Model Info Sheet

struct ModelInfoSheet: View {
    let info: ModelInfo?
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            if let info {
                List {
                    Section("Architecture") {
                        InfoRow(label: "Layers", value: "\(info.numLayers)")
                        InfoRow(label: "Experts", value: "\(info.numExperts) (K=\(info.activeExpertsK))")
                        InfoRow(label: "Hidden Dim", value: "\(info.hiddenDim)")
                        InfoRow(label: "Vocab Size", value: "\(info.vocabSize)")
                    }
                    Section("Storage") {
                        InfoRow(label: "Weights", value: String(format: "%.1f MB", info.weightFileMB))
                        InfoRow(label: "Experts", value: String(format: "%.1f MB", info.expertFileMB))
                        InfoRow(label: "Total", value: String(format: "%.1f GB", info.totalSizeMB / 1024))
                    }
                }
                .navigationTitle("Model Info")
            } else {
                Text("No model loaded")
            }
        }
        .presentationDetents([.medium])
    }
}

struct InfoRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack {
            Text(label)
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .fontDesign(.monospaced)
        }
    }
}
