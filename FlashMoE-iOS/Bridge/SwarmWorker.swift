/*
 * SwarmWorker.swift — Flash-Swarm worker integration for the iOS app
 *
 * Wraps the flashswarm_ios C API (from the Rust static library) and bridges
 * incoming inference tasks to the local FlashMoEEngine.
 *
 * Lifecycle:
 *   1. Create SwarmWorker with a reference to the loaded FlashMoEEngine.
 *   2. Call configure(networkKey:modelPath:) to set connection parameters.
 *   3. Call start() to begin advertising and accepting cluster tasks.
 *   4. Call stop() before the app exits or the engine is unloaded.
 *
 * Thread model:
 *   - All @Observable property mutations happen on the MainActor.
 *   - Inference callbacks arrive on a background thread from the Rust runtime.
 *   - flashmoe_generate is called synchronously on the background thread;
 *     the engine queue serializes it with any local user inference.
 */

import Foundation
import Observation

// MARK: - Swarm connection status

enum SwarmStatus: Int32, Equatable {
    case stopped      = 0
    case advertising  = 1
    case connected    = 2
    case processing   = 3

    var label: String {
        switch self {
        case .stopped:     return "Stopped"
        case .advertising: return "Looking for router…"
        case .connected:   return "Connected"
        case .processing:  return "Processing task"
        }
    }

    var isActive: Bool { self != .stopped }
}

// MARK: - Chat request types (matches the Rust api::ChatCompletionRequest)

private struct SwarmChatMessage: Codable {
    let role: String
    let content: String
}

private struct SwarmChatRequest: Codable {
    let messages: [SwarmChatMessage]
    let max_tokens: Int?
}

// MARK: - SwarmWorker

@Observable
final class SwarmWorker: @unchecked Sendable {
    // State visible to SwiftUI
    private(set) var status: SwarmStatus = .stopped
    private(set) var tasksHandled: Int = 0
    private(set) var lastError: String? = nil
    private(set) var masterAddress: String? = nil

    // Settings (read by the start() path)
    var networkKey: String = "flashswarm"
    var expertPort: UInt16 = 10128
    var discoveryPort: UInt16 = 10127

    // The inference engine (weak to avoid retain cycle)
    private weak var engine: FlashMoEEngine?

    // Serialises flashmoe_generate calls from swarm tasks.
    // Using the engine's own queue ensures we don't double-enter the GPU.
    private let inferenceQueue = DispatchQueue(
        label: "com.flashmoe.swarm-inference",
        qos: .userInitiated
    )

    init(engine: FlashMoEEngine) {
        self.engine = engine
        registerCallbacks()
    }

    deinit {
        // Clear callbacks so Rust doesn't call back into a dead object.
        flashswarm_ios_set_task_handler(nil, nil)
        flashswarm_ios_set_status_callback(nil, nil)
    }

    // MARK: - Callback registration

    private func registerCallbacks() {
        // We pass `self` as an unretained pointer. The deinit clears the
        // callbacks so Rust will never call back after deallocation.
        let selfPtr = Unmanaged.passUnretained(self).toOpaque()

        // Task callback — called on Rust's spawn_blocking thread.
        flashswarm_ios_set_task_handler(
            { requestJson, responseOut, userdata -> Int32 in
                guard let userdata, let requestJson, let responseOut else {
                    return -1
                }
                let worker = Unmanaged<SwarmWorker>
                    .fromOpaque(userdata)
                    .takeUnretainedValue()
                return worker.handleIncomingTask(
                    requestJson: requestJson,
                    responseOut: responseOut
                )
            },
            selfPtr
        )

        // Status callback — called on Rust's background thread.
        flashswarm_ios_set_status_callback(
            { statusCode, infoJson, userdata in
                guard let userdata else { return }
                let worker = Unmanaged<SwarmWorker>
                    .fromOpaque(userdata)
                    .takeUnretainedValue()
                let info = infoJson.map { String(cString: $0) }
                worker.handleStatusChange(code: statusCode, info: info)
            },
            selfPtr
        )
    }

    // MARK: - Lifecycle

    /// Start advertising and accepting cluster tasks.
    ///
    /// `modelPath` should match the path passed to `FlashMoEEngine.loadModel`.
    func start(modelPath: String) {
        guard status == .stopped else { return }

        let keyPtr = (networkKey as NSString).utf8String!
        let pathPtr = (modelPath as NSString).utf8String!

        let rc = flashswarm_ios_init(pathPtr, keyPtr, expertPort, discoveryPort)
        if rc != 0 {
            setError("Failed to initialise swarm worker (rc=\(rc))")
            return
        }

        let startRc = flashswarm_ios_start()
        if startRc != 0 {
            setError("Failed to start swarm worker (rc=\(startRc))")
            return
        }

        setStatus(.advertising)
        lastError = nil
    }

    /// Stop the worker and disconnect from the cluster.
    func stop() {
        guard status != .stopped else { return }
        let _ = flashswarm_ios_stop()
        setStatus(.stopped)
    }

    // MARK: - Internal: task handling (called on Rust background thread)

    /// Synchronous inference handler. Returns 0 on success, -1 on failure.
    ///
    /// - Parameter requestJson: null-terminated JSON `ChatCompletionRequest`
    /// - Parameter responseOut: set to a malloc-allocated JSON response string
    private func handleIncomingTask(
        requestJson: UnsafePointer<CChar>,
        responseOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>
    ) -> Int32 {
        guard let engine else {
            return writeError("engine not available", into: responseOut)
        }

        // Parse request.
        let jsonString = String(cString: requestJson)
        guard
            let jsonData = jsonString.data(using: .utf8),
            let req = try? JSONDecoder().decode(SwarmChatRequest.self, from: jsonData)
        else {
            return writeError("failed to parse ChatCompletionRequest", into: responseOut)
        }

        let prompt = formatQwenChat(req.messages)
        let maxTokens = req.max_tokens ?? 256

        // Run inference synchronously. We block here intentionally — Rust's
        // spawn_blocking provides a dedicated OS thread for this call.
        var generatedTokens: [String] = []
        let generateResult = inferenceQueue.sync { () -> Int32 in
            engine.generateSync(prompt: prompt, maxTokens: Int32(maxTokens)) { token in
                generatedTokens.append(token)
            }
        }

        if generateResult < 0 {
            return writeError("flashmoe_generate failed (\(generateResult))", into: responseOut)
        }

        let rawText = generatedTokens.joined()
        let cleaned = cleanAssistantOutput(rawText)

        let responseJson = #"{"role":"assistant","content":"\#(jsonEscape(cleaned))"}"#

        // Allocate the response string with malloc so Rust can call free() on it.
        guard let cStr = (responseJson as NSString).utf8String else {
            return writeError("failed to encode response", into: responseOut)
        }
        let len = strlen(cStr) + 1
        let buf = malloc(len)!
        memcpy(buf, cStr, len)
        responseOut.pointee = buf.assumingMemoryBound(to: CChar.self)

        // Update stats on main thread (non-blocking).
        DispatchQueue.main.async { [weak self] in
            self?.tasksHandled += 1
        }

        return 0
    }

    // MARK: - Internal: status change (called on Rust background thread)

    private func handleStatusChange(code: Int32, info: String?) {
        let newStatus = SwarmStatus(rawValue: code) ?? .stopped
        DispatchQueue.main.async { [weak self] in
            self?.status = newStatus
            // Extract master address from info JSON if present.
            if let info, let data = info.data(using: .utf8),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let addr = json["master"] as? String
            {
                self?.masterAddress = addr
            }
        }
    }

    // MARK: - Helpers

    private func setStatus(_ s: SwarmStatus) {
        DispatchQueue.main.async { [weak self] in self?.status = s }
    }

    private func setError(_ msg: String) {
        DispatchQueue.main.async { [weak self] in self?.lastError = msg }
    }

    private func writeError(
        _ message: String,
        into responseOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>
    ) -> Int32 {
        let errJson = #"{"error":"\#(jsonEscape(message))"}"#
        if let cStr = (errJson as NSString).utf8String {
            let len = strlen(cStr) + 1
            let buf = malloc(len)!
            memcpy(buf, cStr, len)
            responseOut.pointee = buf.assumingMemoryBound(to: CChar.self)
        }
        return -1
    }

    /// Format messages into Qwen3 chat template.
    /// Must match `format_qwen_chat` in flashswarm-cli/src/api.rs.
    private func formatQwenChat(_ messages: [SwarmChatMessage]) -> String {
        var prompt = ""
        for msg in messages {
            prompt += "<|im_start|>\(msg.role)\n\(msg.content)<|im_end|>\n"
        }
        prompt += "<|im_start|>assistant\n"
        return prompt
    }

    /// Strip Qwen special tokens from generated output.
    private func cleanAssistantOutput(_ text: String) -> String {
        var cleaned = text
        // Remove end-of-turn token and anything after it.
        if let range = cleaned.range(of: "<|im_end|>") {
            cleaned = String(cleaned[..<range.lowerBound])
        }
        // Remove thinking block if present.
        if let start = cleaned.range(of: "<think>"),
           let end = cleaned.range(of: "</think>") {
            cleaned.removeSubrange(start.lowerBound...end.upperBound)
        }
        return cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Escape a string for inclusion in a JSON string literal.
    private func jsonEscape(_ s: String) -> String {
        s
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n")
            .replacingOccurrences(of: "\r", with: "\\r")
            .replacingOccurrences(of: "\t", with: "\\t")
    }
}

// MARK: - FlashMoEEngine synchronous generation extension

extension FlashMoEEngine {
    /// Generate text synchronously, calling `onToken` for each token.
    ///
    /// Runs on the caller's thread — call from a background thread only.
    /// Serialises with internal engine state via the engine's dispatch queue.
    ///
    /// - Returns: token count on success (≥0), -1 on error.
    @discardableResult
    func generateSync(
        prompt: String,
        maxTokens: Int32,
        onToken: @escaping (String) -> Void
    ) -> Int32 {
        // Capture the context synchronously on the engine queue to avoid
        // racing with loadModel/unloadModel.
        var capturedCtx: OpaquePointer? = nil
        engineQueue.sync {
            capturedCtx = self.context
        }

        guard let ctx = capturedCtx else { return -1 }

        // Collect tokens through a heap-allocated box so the C callback can
        // write into it without capturing a Swift closure.
        let box = OnTokenBox(onToken)

        let result = flashmoe_generate(
            ctx,
            prompt,
            maxTokens,
            { tokenText, _, _, _, userData -> Int32 in
                guard let userData, let tokenText else { return 0 }
                let b = Unmanaged<OnTokenBox>.fromOpaque(userData).takeUnretainedValue()
                b.onToken(String(cString: tokenText))
                return 0
            },
            Unmanaged.passUnretained(box).toOpaque()
        )

        return result
    }
}

// Helper class to carry the Swift closure through a C void* callback.
private final class OnTokenBox {
    let onToken: (String) -> Void
    init(_ fn: @escaping (String) -> Void) { self.onToken = fn }
}
