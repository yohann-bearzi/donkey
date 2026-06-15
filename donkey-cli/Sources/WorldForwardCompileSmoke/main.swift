// WorldForwardCompileSmoke — verify DonkeyWorldForward init compiles all
// kernels from the canonical spec with deterministic random weights.
//
// No forward() call yet; that's the next commit. This smoke catches
// shape-derivation bugs (e.g. an off-by-one in how cfg sizes map to kernel
// templates) before any orchestration logic is exercised.
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
struct WorldForwardCompileSmoke {
    static func main() {
        let specPath = ProcessInfo.processInfo.environment["DONKEY_SPEC"]
            ?? "spec/donkey_v2_default.json"
        slog("[v2-compile] loading config from \(specPath)")

        let cfg: DonkeyConfig
        do {
            cfg = try DonkeyConfig(jsonURL: URL(fileURLWithPath: specPath))
        } catch {
            slog("[v2-compile] FAIL: config load: \(error)")
            exit(1)
        }
        slog("[v2-compile] cfg: D=\(cfg.architecture.hiddenDim) F=\(cfg.architecture.ffnDim) layers=\(cfg.architecture.nLayers) W=\(cfg.sequence.windowSize) K=\(cfg.sequence.draftSize) SP=\(cfg.sequence.spatialPad)")

        slog("[v2-compile] generating random weights")
        let weights = makeRandomWeights(cfg)

        let nKernels = 1                                            // input_proj
                     + 7 * cfg.architecture.nLayers                 // QKVO + SDPA + Up + Down per layer
                     + 1                                            // head
        slog("[v2-compile] compiling \(nKernels) kernels from cfg")

        let t0 = Date()
        do {
            _ = try DonkeyWorldForward(cfg: cfg, weights: weights)
        } catch {
            slog("[v2-compile] FAIL: \(error)")
            exit(2)
        }
        let ms = -t0.timeIntervalSinceNow * 1000.0
        slog("[v2-compile] all kernels compiled in \(String(format: "%.0f ms", ms))")
        slog("[v2-compile] OK")
    }
}
