import Foundation
import MLX
import MLXRandom
import MLXLMCommon
import MLXLLM
import BenchmarkHelpers
import DonkeyArtifacts

fileprivate func slog(_ s: String) {
    FileHandle.standardError.write(Data("\(s)\n".utf8))
}

/// Sample a token from logits with temperature via the Gumbel-max trick.
/// argmax(logits/T + Gumbel(0,1)) ~ Categorical(softmax(logits/T)).
/// Faster than MLXRandom.categorical because it's a single argMax (already
/// well-optimized) plus a small noise draw, both on-GPU.
/// At temp=0, behaves as pure argmax. top_p arg accepted for forward
/// compatibility but not yet implemented.
fileprivate func sampleTempTopP(_ logits: MLXArray, temperature: Float, topP: Float) -> Int {
    if temperature <= 0.0 {
        return argMax(logits, axis: -1).item(Int.self)
    }
    let scaled = logits / temperature
    let noise = MLXRandom.gumbel(scaled.shape, type: Float.self)
    return argMax(scaled + noise, axis: -1).item(Int.self)
}


fileprivate func appendData(_ data: Data, to url: URL) throws {
    if FileManager.default.fileExists(atPath: url.path) {
        let fh = try FileHandle(forWritingTo: url)
        defer { try? fh.close() }
        try fh.seekToEnd()
        try fh.write(contentsOf: data)
    } else {
        try data.write(to: url)
    }
}

fileprivate func readInt32Bin(_ url: URL) throws -> [Int32] {
    let data = try Data(contentsOf: url)
    let count = data.count / MemoryLayout<Int32>.size
    var out = [Int32](repeating: 0, count: count)
    out.withUnsafeMutableBytes { buf in
        _ = data.copyBytes(to: buf)
    }
    return out
}

/// Phase 5.3: batch trunk self-generation over a tokenized prompt corpus.
///
/// Reads <prompts_dir>/prompts.bin + offsets.bin (produced by prepare_*.py),
/// iterates each prompt with a FRESH KV cache (no cross-prompt contamination),
/// and appends (lastHiddenState, fed_token) per position to the standard
/// trace format under <out_dir>. Greedy decode.
///
/// Usage: trace-selfgen-batch <prompts_dir> <out_dir> [maxNewPerPrompt=128]
@main
struct TraceSelfGenBatch {
    static func main() async {
        let args = CommandLine.arguments
        guard args.count >= 3 else {
            slog("usage: trace-selfgen-batch <prompts_dir> <out_dir> [maxNew=128] [temp=0.8] [topP=0.95]")
            exit(2)
        }
        let promptsDir = URL(fileURLWithPath: args[1])
        let outDir     = URL(fileURLWithPath: args[2])
        let maxNew     = args.count > 3 ? (Int(args[3]) ?? 128) : 128
        let temp       = args.count > 4 ? (Float(args[4]) ?? 0.8) : 0.8
        let topP       = args.count > 5 ? (Float(args[5]) ?? 0.95) : 0.95
        slog("[batch] sampler: temp=\(temp) topP=\(topP)")

        let modelPath = ProcessInfo.processInfo.environment["MIMO_PATH"]
            ?? "/Volumes/TB5/llm/MiMo-V2-Flash-JANG_4M"
        let trunkDir = URL(fileURLWithPath: modelPath)

        let promptsAll: [Int32]
        let offsets: [Int32]
        do {
            promptsAll = try readInt32Bin(promptsDir.appendingPathComponent("prompts.bin"))
            offsets    = try readInt32Bin(promptsDir.appendingPathComponent("offsets.bin"))
        } catch {
            slog("[batch] FAIL: reading prompt bins: \(error)"); exit(3)
        }
        let nPrompts = offsets.count - 1
        guard nPrompts > 0 else { slog("[batch] FAIL: empty prompts"); exit(4) }
        slog("[batch] \(nPrompts) prompts, \(promptsAll.count) prompt tokens total, maxNew=\(maxNew)")

        do {
            try FileManager.default.createDirectory(
                at: outDir, withIntermediateDirectories: true)
        } catch {
            slog("[batch] FAIL: mkdir: \(error)"); exit(5)
        }

        let trunkHash: String
        do {
            trunkHash = try TrunkHash.compute(trunkDir: trunkDir)
        } catch {
            slog("[batch] FAIL: hash: \(error)"); exit(6)
        }

        slog("[batch] loading MiMo from \(modelPath)")
        do {
            let container = try await LLMModelFactory.shared.loadContainer(
                from: trunkDir, using: NoOpTokenizerLoader())

            await container.perform { ctx in
                guard let model = ctx.model as? MiMoV2FlashModel else {
                    slog("[batch] FAIL: not MiMoV2FlashModel"); exit(7)
                }

                let hiddenURL = outDir.appendingPathComponent("lastHiddenState.bin")
                let tokensURL = outDir.appendingPathComponent("tokens.bin")
                let promptIdxURL = outDir.appendingPathComponent("prompt_idx.bin")
                let metaURL   = outDir.appendingPathComponent("meta.json")

                var totalPositions = 0
                let runStart = Date()

                for pi in 0..<nPrompts {
                    let s = Int(offsets[pi])
                    let e = Int(offsets[pi + 1])
                    let promptTokens = Array(promptsAll[s..<e])
                    if promptTokens.isEmpty { continue }

                    let params = GenerateParameters()
                    let cache = model.newCache(parameters: params)

                    var hiddenFlat: [Float] = []
                    var tokensCaptured: [Int32] = []
                    hiddenFlat.reserveCapacity((promptTokens.count + maxNew) * 4096)
                    tokensCaptured.reserveCapacity(promptTokens.count + maxNew)

                    // Prefill (Convention 2: hidden[i] paired with the token it predicts)
                    //
                    // We feed promptTokens[0..T-1] and get T hidden states. Each hidden[i]
                    // predicts the token at the next position. So:
                    //   row 0:     (hidden[0], promptTokens[1])
                    //   row 1:     (hidden[1], promptTokens[2])
                    //   ...
                    //   row T-2:   (hidden[T-2], promptTokens[T-1])
                    //   row T-1:   (hidden[T-1], firstGeneratedToken)  -- filled by first decode step
                    //
                    // We park hidden[T-1] without a label, and the first decode iteration
                    // appends its argmax as the label retroactively.
                    var nextTok: Int
                    var pendingLastPrefillHidden: [Float] = []
                    do {
                        let arr = MLXArray(promptTokens).reshaped([1, promptTokens.count])
                        let result = model(LMInput.Text(tokens: arr), cache: cache, state: nil)
                        guard let st = result.state, let lh = st.lastHiddenState else {
                            slog("[batch] FAIL p=\(pi): prefill hidden nil"); exit(8)
                        }
                        let lhFlat: [Float] = lh.asType(.float32).asArray(Float.self)
                        let expected = promptTokens.count * 4096
                        guard lhFlat.count == expected else {
                            slog("[batch] FAIL p=\(pi): prefill count \(lhFlat.count) != \(expected)")
                            exit(9)
                        }
                        // CHANGED: do NOT write prefill positions to trace.
                        // Donkey only operates during generation at inference time, so
                        // training on prefill positions wastes capacity on a regime that's
                        // never on. We still RUN the prefill (needed for KV cache + first
                        // nextTok), but discard all but the final hidden state.
                        let T = promptTokens.count
                        // Park hidden[T-1] for the upcoming decode loop to label
                        // with whatever token gets generated first.
                        pendingLastPrefillHidden = Array(lhFlat[((T-1) * 4096)..<(T * 4096)])
                        nextTok = sampleTempTopP(result.logits[0, -1, 0...], temperature: temp, topP: topP)
                    }

                    // Decode (Convention 2: hidden labeled by what IT predicts, not what was fed)
                    //
                    // We maintain `prevHidden`: the hidden state from the previous forward,
                    // waiting for a label. Each iteration:
                    //   1. Feed nextTok, get (newHidden, newLogits)
                    //   2. thisArgmax = argmax(newLogits) -- this is what newHidden predicts
                    //      AND it's also what prevHidden's *successor input* would be in a
                    //      teacher-forced setup. But what prevHidden actually predicts is
                    //      exactly what we fed THIS step (nextTok).
                    //   3. Write (prevHidden, nextTok)  -- prevHidden's label is THE token we fed
                    //   4. Set prevHidden = newHidden
                    //   5. nextTok = thisArgmax (greedy decode)
                    //
                    // At loop start, prevHidden = pendingLastPrefillHidden (the final
                    // prefill hidden — its label is the first generated token, which IS
                    // a generation-time prediction).
                    // The very last decode step's newHidden has no successor → discarded.
                    // Net positions written: number of generated tokens (1 + steps until EOS).
                    // Decode UNTIL a STOP token, with a safety cap.
                    // STOP_TOKENS: <|im_end|>=151645 (chat-end), <|endoftext|>=151643 (raw EOS)
                    let STOP_TOKENS: Set<Int> = [151645, 151643]
                    // No cap: generate until EOS. The huge safety net (200K)
                    // exists only to prevent infinite loops in case of catastrophic
                    // bugs — a real MiMo response should be well under 100K tokens.
                    let MAX_GEN_CAP = 32768
                    var prevHidden = pendingLastPrefillHidden
                    var step = 0
                    var hitStop = false
                    while step < MAX_GEN_CAP {
                        let arr = MLXArray([Int32(nextTok)]).reshaped([1, 1])
                        let result = model(LMInput.Text(tokens: arr), cache: cache, state: nil)
                        guard let st = result.state, let lh = st.lastHiddenState else {
                            slog("[batch] FAIL p=\(pi) step \(step): hidden nil"); exit(10)
                        }
                        let lhFlat: [Float] = lh.asType(.float32).asArray(Float.self)
                        guard lhFlat.count == 4096 else {
                            slog("[batch] FAIL p=\(pi) step \(step): hidden size \(lhFlat.count)")
                            exit(11)
                        }
                        // prevHidden predicts the token we just fed (nextTok). Always write
                        // INCLUDING the stop token (model learns "about to end" patterns).
                        hiddenFlat.append(contentsOf: prevHidden)
                        tokensCaptured.append(Int32(nextTok))
                        slog("[tok] \(nextTok)")
                        prevHidden = lhFlat
                        step += 1
                        if STOP_TOKENS.contains(nextTok) { hitStop = true; break }
                        nextTok = sampleTempTopP(result.logits[0, -1, 0...], temperature: temp, topP: topP)
                    }
                    // prevHidden (the final forward's output) has no successor; discard.
                    if !hitStop {
                        slog("[batch] p=\(pi) ABORTED at safety cap=\(MAX_GEN_CAP) — investigate!")
                    }

                    let positions = tokensCaptured.count
                    let promptIdxStream = [Int32](repeating: Int32(pi), count: positions)

                    let hData = hiddenFlat.withUnsafeBufferPointer { Data(buffer: $0) }
                    let tData = tokensCaptured.withUnsafeBufferPointer { Data(buffer: $0) }
                    let pData = promptIdxStream.withUnsafeBufferPointer { Data(buffer: $0) }
                    do {
                        try appendData(hData, to: hiddenURL)
                        try appendData(tData, to: tokensURL)
                        try appendData(pData, to: promptIdxURL)
                    } catch {
                        slog("[batch] FAIL p=\(pi): append: \(error)"); exit(12)
                    }
                    totalPositions += positions

                    if pi < 3 || pi % 16 == 0 || pi == nPrompts - 1 {
                        let elapsed = Date().timeIntervalSince(runStart)
                        let rate = Double(totalPositions) / max(elapsed, 0.001)
                        slog(String(format: "[batch] p=%d/%d  pos+=%d  total=%d  %.1f tok/s  elapsed=%.1fs",
                            pi + 1, nPrompts, positions, totalPositions, rate, elapsed))
                    }

                    // Free per-prompt MLX state aggressively
                    Memory.clearCache()
                }

                let meta: [String: Any] = [
                        "convention": "self-aligned: hidden[i] predicts tokens[i] (GENERATION-ONLY, no prefill)",
                        "prediction_offset": 0,
                        "normalization_applied": false,
                        "hidden_layer": "pre-final-rmsnorm",
                    "hidden_dim": 4096,
                    "hidden_dtype": "float32",
                    "token_dtype": "int32",
                    "total_positions": totalPositions,
                    "prompt_count": nPrompts,
                    "max_new_per_prompt_cap": maxNew,
                    "max_gen_cap_used": max(maxNew, 4096),
                    "generation_stop_tokens": [151643, 151645],
                    "trunk_hash": trunkHash,
                    "source": "self-generation batch (greedy, KV-cached)",
                    "prompts_dir": promptsDir.path,
                ]
                do {
                    let metaData = try JSONSerialization.data(
                        withJSONObject: meta, options: [.prettyPrinted, .sortedKeys])
                    try metaData.write(to: metaURL)
                } catch {
                    slog("[batch] FAIL: meta: \(error)"); exit(13)
                }
                slog("[batch] OK  totalPositions=\(totalPositions)")
            }
        } catch {
            slog("[batch] error: \(error)"); exit(99)
        }
    }
}
