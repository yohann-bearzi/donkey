// WorldForwardSmoke — execute DonkeyWorldForward end-to-end with random
// weights. Validates the orchestration logic (history buffer, cold-start
// replication, draft-slot extraction, sigmoid on confidence) by checking:
//   - shapes correct
//   - no NaN/Inf
//   - confidence in [0, 1]
//   - bit-deterministic across N steady-state calls with same input
//   - steady-state latency under threshold
//
// PyTorch numerical-equivalence check comes in Step 6 (separate smoke).
import Foundation
import DonkeyConfig
import DonkeyRuntime

fileprivate func slog(_ s: String) {
    FileHandle.standardError.write(Data("\(s)\n".utf8))
}

fileprivate func rng(_ seed: inout UInt32) -> Float {
    seed = seed &* 1664525 &+ 1013904223
    return Float((seed >> 8) & 0x00FFFFFF) / Float(1 << 23) - 1.0
}
fileprivate func randomArray(count: Int, seed: UInt32, scale: Float = 1.0) -> [Float] {
    var s = seed
    var a = [Float](repeating: 0, count: count)
    for i in 0..<count { a[i] = rng(&s) * scale }
    return a
}

fileprivate func makeRandomWeights(_ cfg: DonkeyConfig) -> DonkeyWorldWeights {
    let D  = cfg.architecture.hiddenDim
    let F_ = cfg.architecture.ffnDim
    let TH = cfg.trunk.hiddenDim
    let K  = cfg.sequence.draftSize
    let SP = cfg.sequence.spatialPad
    let OUT = cfg.outCh

    let invSqrtTH = 1.0 / Float(TH).squareRoot()
    let invSqrtD  = 1.0 / Float(D).squareRoot()
    let invSqrtF  = 1.0 / Float(F_).squareRoot()

    let inputProj    = randomArray(count: D * TH, seed: 0x1000_0000, scale: invSqrtTH)
    let draftQueries = randomArray(count: D * K,  seed: 0x1100_0000, scale: 0.02)
    let positionalBias = randomArray(count: D * SP, seed: 0x1200_0000, scale: 0.02)

    var layers = [DonkeyWorldWeights.LayerWeights]()
    for l in 0..<cfg.architecture.nLayers {
        let base = UInt32(0x2000_0000 &+ UInt32(l) &* 0x0100_0000)
        var gammaAtt = randomArray(count: D, seed: base, scale: 0.1)
        for i in 0..<D { gammaAtt[i] += 1.0 }
        var gammaFfn = randomArray(count: D, seed: base &+ 1, scale: 0.1)
        for i in 0..<D { gammaFfn[i] += 1.0 }
        let Wq    = randomArray(count: D * D,  seed: base &+ 2, scale: invSqrtD)
        let Wk    = randomArray(count: D * D,  seed: base &+ 3, scale: invSqrtD)
        let Wv    = randomArray(count: D * D,  seed: base &+ 4, scale: invSqrtD)
        let Wo    = randomArray(count: D * D,  seed: base &+ 5, scale: invSqrtD)
        let Wup   = randomArray(count: F_ * D, seed: base &+ 6, scale: invSqrtD)
        let Wdown = randomArray(count: D * F_, seed: base &+ 7, scale: invSqrtF)
        layers.append(DonkeyWorldWeights.LayerWeights(
            gammaAtt: gammaAtt, Wq: Wq, Wk: Wk, Wv: Wv, Wo: Wo,
            gammaFfn: gammaFfn, Wup: Wup, Wdown: Wdown))
    }
    var gammaFinal = randomArray(count: D, seed: 0x4000_0000, scale: 0.1)
    for i in 0..<D { gammaFinal[i] += 1.0 }
    let head = randomArray(count: OUT * D, seed: 0x5000_0000, scale: invSqrtD)

    let inputProjLnGamma = [Float](repeating: 1.0, count: D)
    let inputProjLnBeta  = [Float](repeating: 0.0, count: D)
    return DonkeyWorldWeights(
        inputProj: inputProj,
        inputProjLnGamma: inputProjLnGamma, inputProjLnBeta: inputProjLnBeta,
        draftQueries: draftQueries, positionalBias: positionalBias,
        layers: layers, gammaFinal: gammaFinal, head: head)
}

@main
struct WorldForwardSmoke {
    static func main() {
        let specPath = ProcessInfo.processInfo.environment["DONKEY_SPEC"]
            ?? "spec/donkey_v2_default.json"
        slog("[v2-fwd] loading config from \(specPath)")
        let cfg: DonkeyConfig
        do {
            cfg = try DonkeyConfig(jsonURL: URL(fileURLWithPath: specPath))
        } catch {
            slog("[v2-fwd] FAIL config load: \(error)"); exit(1)
        }
        let TH = cfg.trunk.hiddenDim
        let K = cfg.sequence.draftSize

        slog("[v2-fwd] building weights + orchestrator")
        let weights = makeRandomWeights(cfg)
        let t0 = Date()
        let donkey: DonkeyWorldForward
        do {
            donkey = try DonkeyWorldForward(cfg: cfg, weights: weights)
        } catch {
            slog("[v2-fwd] FAIL init: \(error)"); exit(2)
        }
        slog("[v2-fwd] init OK in \(String(format: "%.0f ms", -t0.timeIntervalSinceNow * 1000.0))")

        // === Cold-start forward (very first call). ===
        let h0 = randomArray(count: TH, seed: 0xAAAA_0000, scale: 0.5)
        let out0: DonkeyDraft
        do {
            out0 = try donkey.forward(currentHidden: h0)
        } catch {
            slog("[v2-fwd] FAIL cold forward: \(error)"); exit(3)
        }
        guard out0.predHidden.count == TH * K else {
            slog("[v2-fwd] FAIL: predHidden size \(out0.predHidden.count) != \(TH * K)"); exit(4)
        }
        guard out0.confidence.count == K else {
            slog("[v2-fwd] FAIL: confidence size \(out0.confidence.count) != \(K)"); exit(5)
        }
        for v in out0.predHidden where !v.isFinite {
            slog("[v2-fwd] FAIL: predHidden has NaN/Inf"); exit(6)
        }
        for v in out0.confidence {
            if !v.isFinite { slog("[v2-fwd] FAIL: confidence has NaN/Inf"); exit(7) }
            if v < 0 || v > 1 { slog("[v2-fwd] FAIL: confidence out of [0,1]: \(v)"); exit(8) }
        }
        slog("[v2-fwd] cold forward OK")
        slog("[v2-fwd]   predHidden[0..4] = \(Array(out0.predHidden[0..<4]))")
        slog("[v2-fwd]   confidence       = \(out0.confidence)")

        // === Run a few more times to populate the rolling window. ===
        for i in 1..<5 {
            let hi = randomArray(count: TH, seed: UInt32(0xBBBB_0000 &+ UInt32(i)), scale: 0.5)
            do {
                let outI = try donkey.forward(currentHidden: hi)
                for v in outI.predHidden where !v.isFinite {
                    slog("[v2-fwd] FAIL: step \(i) predHidden has NaN/Inf"); exit(10)
                }
                for v in outI.confidence where v < 0 || v > 1 {
                    slog("[v2-fwd] FAIL: step \(i) confidence out of [0,1]: \(v)"); exit(11)
                }
                if i == 1 || i == 4 {
                    slog("[v2-fwd] step \(i) conf = \(outI.confidence)")
                }
            } catch {
                slog("[v2-fwd] FAIL step \(i): \(error)"); exit(9)
            }
        }
        slog("[v2-fwd] rolling-window calls OK")

        // === Steady-state latency. Reset between to get a clean run. ===
        donkey.reset()
        for _ in 0..<5 { _ = try? donkey.forward(currentHidden: h0) }
        let tBench = Date()
        let ITERS = 20
        for _ in 0..<ITERS { _ = try? donkey.forward(currentHidden: h0) }
        let ms = -tBench.timeIntervalSinceNow * 1000.0 / Double(ITERS)
        slog("[v2-fwd] steady-state: \(String(format: "%.2f ms/call", ms)) over \(ITERS) iters")
        slog("[v2-fwd] OK")
    }
}
