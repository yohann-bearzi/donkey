// WorldDumpSmoke — Phase 4.5: write Swift orchestrator's weights + outputs
// for Python cross-validation.
//
// usage: world-dump-smoke <dump_dir>
//
// Writes:
//   inputProj.bin           [D, TH]
//   draftQueries.bin        [D, K]
//   positionalBias.bin      [D, SP]
//   layer_{L}_gammaAtt.bin  [D]
//   layer_{L}_Wq.bin        [D, D]
//   layer_{L}_Wk.bin        [D, D]
//   layer_{L}_Wv.bin        [D, D]
//   layer_{L}_Wo.bin        [D, D]
//   layer_{L}_gammaFfn.bin  [D]
//   layer_{L}_Wup.bin       [F, D]
//   layer_{L}_Wdown.bin     [D, F]
//   gammaFinal.bin          [D]
//   head.bin                [OUT, D]
//   h_first.bin             [TH]  -- the cold-start trunk hidden
//   swift_pred_hidden.bin   [TH * K]
//   swift_confidence.bin    [K]
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

fileprivate func writeFloats(_ a: [Float], to path: String) throws {
    let data = a.withUnsafeBufferPointer { Data(buffer: $0) }
    try data.write(to: URL(fileURLWithPath: path))
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

    let inputProj    = randomArray(count: D * TH,  seed: 0x1000_0000, scale: invSqrtTH)
    let draftQueries = randomArray(count: D * K,   seed: 0x1100_0000, scale: 0.02)
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
struct WorldDumpSmoke {
    static func main() throws {
        let args = CommandLine.arguments
        guard args.count == 2 else {
            slog("usage: world-dump-smoke <dump_dir>")
            exit(1)
        }
        let dumpDir = args[1]
        try FileManager.default.createDirectory(atPath: dumpDir, withIntermediateDirectories: true)

        let specPath = ProcessInfo.processInfo.environment["DONKEY_SPEC"]
            ?? "spec/donkey_v2_default.json"
        slog("[dump] loading config from \(specPath)")
        let cfg = try DonkeyConfig(jsonURL: URL(fileURLWithPath: specPath))

        slog("[dump] generating weights")
        let w = makeRandomWeights(cfg)

        slog("[dump] writing weights to \(dumpDir)")
        try writeFloats(w.inputProj,      to: "\(dumpDir)/inputProj.bin")
        try writeFloats(w.inputProjLnGamma, to: "\(dumpDir)/inputProjLnGamma.bin")
        try writeFloats(w.inputProjLnBeta,  to: "\(dumpDir)/inputProjLnBeta.bin")
        try writeFloats(w.draftQueries,   to: "\(dumpDir)/draftQueries.bin")
        try writeFloats(w.positionalBias, to: "\(dumpDir)/positionalBias.bin")
        for (l, ly) in w.layers.enumerated() {
            try writeFloats(ly.gammaAtt, to: "\(dumpDir)/layer_\(l)_gammaAtt.bin")
            try writeFloats(ly.Wq,       to: "\(dumpDir)/layer_\(l)_Wq.bin")
            try writeFloats(ly.Wk,       to: "\(dumpDir)/layer_\(l)_Wk.bin")
            try writeFloats(ly.Wv,       to: "\(dumpDir)/layer_\(l)_Wv.bin")
            try writeFloats(ly.Wo,       to: "\(dumpDir)/layer_\(l)_Wo.bin")
            try writeFloats(ly.gammaFfn, to: "\(dumpDir)/layer_\(l)_gammaFfn.bin")
            try writeFloats(ly.Wup,      to: "\(dumpDir)/layer_\(l)_Wup.bin")
            try writeFloats(ly.Wdown,    to: "\(dumpDir)/layer_\(l)_Wdown.bin")
        }
        try writeFloats(w.gammaFinal, to: "\(dumpDir)/gammaFinal.bin")
        try writeFloats(w.head,       to: "\(dumpDir)/head.bin")

        // Cold-start input (deterministic seed).
        let h0 = randomArray(count: cfg.trunk.hiddenDim, seed: 0xAAAA_0000, scale: 0.5)
        try writeFloats(h0, to: "\(dumpDir)/h_first.bin")

        slog("[dump] compiling orchestrator")
        let donkey = try DonkeyWorldForward(cfg: cfg, weights: w)

        slog("[dump] running Swift forward (cold start)")
        let out = try donkey.forward(currentHidden: h0)
        slog("[dump]   predHidden[0..4] = \(Array(out.predHidden[0..<4]))")
        slog("[dump]   confidence       = \(out.confidence)")

        try writeFloats(out.predHidden, to: "\(dumpDir)/swift_pred_hidden.bin")
        try writeFloats(out.confidence, to: "\(dumpDir)/swift_confidence.bin")
        slog("[dump] done")
    }
}
