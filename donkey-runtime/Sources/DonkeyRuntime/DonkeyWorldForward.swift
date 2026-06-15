// DonkeyWorldForward — v2 world-model orchestrator on M3 Ultra ANE.
//
// Driven entirely by DonkeyConfig. No magic numbers. To run a parameter
// sweep: edit a JSON spec, instantiate with that config. All kernel shapes,
// buffer sizes, and orchestration loops derive from cfg.
//
// Architecture (matches donkey_world.py DonkeyWorldRef):
//
//   per call:
//     1. push currentHidden into rolling history buffer
//        (cold-start: replicate firstHidden across all W slots)
//     2. input_proj ANE: [trunk_hidden, SP] -> [hidden_dim, SP]
//        (positions W..SP-1 get overwritten by draft queries after)
//     3. CPU: overwrite slots W..W+K-1 with learned draftQueries
//     4. CPU: add learned positionalBias to all SP slots
//     5. for layer in 0..<nLayers:
//          RMSNorm(att) -> Q/K/V -> concat -> SDPA -> Wo -> residual
//          RMSNorm(ffn) -> up -> silu -> down -> residual
//     6. CPU: RMSNorm(final)
//     7. head ANE: [hidden_dim, SP] -> [trunk_hidden + out_conf, SP]
//     8. extract draft slots W..W+K-1; sigmoid on conf channel
//     9. return DonkeyDraft { predHidden [trunk_hidden, K], confidence [K] }
//
// Step 2 of Phase 4: this file implements init (compile kernels + allocate
// buffers from cfg). forward() will be added in the next commit.
import Foundation
import DonkeyANE
import DonkeyConfig
import DonkeyOps
import Accelerate

public enum DonkeyWorldError: Error, CustomStringConvertible {
    case shapeMismatch(String)
    case missingWeight(String)
    case compileFailed(String)
    public var description: String {
        switch self {
        case .shapeMismatch(let s): return "shape mismatch: \(s)"
        case .missingWeight(let s): return "missing weight: \(s)"
        case .compileFailed(let s): return "compile failed: \(s)"
        }
    }
}

// MARK: - Weight container, derived from DonkeyConfig

public struct DonkeyWorldWeights {
    public struct LayerWeights {
        public var gammaAtt: [Float]   // [hidden_dim]
        public var Wq, Wk, Wv, Wo: [Float]   // [hidden_dim, hidden_dim]
        public var gammaFfn: [Float]   // [hidden_dim]
        public var Wup: [Float]        // [ffn_dim, hidden_dim]
        public var Wdown: [Float]      // [hidden_dim, ffn_dim]
        public init(gammaAtt: [Float], Wq: [Float], Wk: [Float], Wv: [Float], Wo: [Float],
                    gammaFfn: [Float], Wup: [Float], Wdown: [Float]) {
            self.gammaAtt = gammaAtt; self.Wq = Wq; self.Wk = Wk; self.Wv = Wv; self.Wo = Wo
            self.gammaFfn = gammaFfn; self.Wup = Wup; self.Wdown = Wdown
        }
    }

    public var inputProj: [Float]            // [hidden_dim, trunk_hidden]
    public var inputProjLnGamma: [Float]     // [hidden_dim] LN after input_proj
    public var inputProjLnBeta: [Float]      // [hidden_dim]
    public var draftQueries: [Float]         // [hidden_dim, draft_size]
    public var positionalBias: [Float]       // [hidden_dim, spatial_pad]
    public var layers: [LayerWeights]        // length n_layers
    public var gammaFinal: [Float]           // [hidden_dim]
    public var head: [Float]                 // [trunk_hidden + out_conf_dim, hidden_dim]

    public init(inputProj: [Float],
                inputProjLnGamma: [Float], inputProjLnBeta: [Float],
                draftQueries: [Float], positionalBias: [Float],
                layers: [LayerWeights], gammaFinal: [Float], head: [Float]) {
        self.inputProj = inputProj
        self.inputProjLnGamma = inputProjLnGamma
        self.inputProjLnBeta  = inputProjLnBeta
        self.draftQueries = draftQueries
        self.positionalBias = positionalBias
        self.layers = layers
        self.gammaFinal = gammaFinal
        self.head = head
    }
}

// MARK: - Output type

public struct DonkeyDraft {
    public var predHidden: [Float]       // [trunk_hidden * draft_size]
    public var confidence: [Float]       // [draft_size]
    public init(predHidden: [Float], confidence: [Float]) {
        self.predHidden = predHidden
        self.confidence = confidence
    }
}

// MARK: - Orchestrator

public final class DonkeyWorldForward {
    public let cfg: DonkeyConfig

    // Kernels (compiled once at init from baked weights).
    private let kInputProj: ANEKernel
    private let inputProjLnGamma: [Float]
    private let inputProjLnBeta:  [Float]
    private let kQ: [ANEKernel]              // per layer
    private let kK: [ANEKernel]
    private let kV: [ANEKernel]
    private let kO: [ANEKernel]
    private let kSDPA: [ANEKernel]
    private let kFFNUp: [ANEKernel]
    private let kFFNDown: [ANEKernel]
    private let kHead: ANEKernel

    // CPU constants used by forward(): gammas, draft queries, positional bias.
    private let gammaAtt: [[Float]]          // [n_layers][hidden_dim]
    private let gammaFfn: [[Float]]
    private let gammaFinal: [Float]
    private let draftQueries: [Float]        // [hidden_dim * draft_size]
    private let positionalBias: [Float]      // [hidden_dim * spatial_pad]

    // Buffers (pre-allocated; forward() will use them).
    // Sizes documented in init.
    private var bufHistory: [Float]          // [trunk_hidden * spatial_pad] — pre-projection
    private var lnScratchMean: [Float]
    private var lnScratchVar:  [Float]
    private var bufHidden: [Float]           // [hidden_dim * SP]
    private var bufHidden2: [Float]          // [hidden_dim * SP] residual save
    private var bufNorm: [Float]
    private var bufQ: [Float]
    private var bufK: [Float]
    private var bufV: [Float]
    private var bufQKV: [Float]              // [3 * hidden_dim * SP]
    private var bufAttn: [Float]
    private var bufOO: [Float]
    private var bufHUp: [Float]              // [ffn_dim * SP]
    private var bufHSilu: [Float]
    private var bufHDown: [Float]
    private var bufHeadOut: [Float]          // [(trunk_hidden + out_conf) * SP]
    private var scratchRms: [Float]
    private var scratchSilu: [Float]

    // History state across calls.
    private var firstHidden: [Float]?
    private var realHistoryCount: Int = 0
    // bufHistory holds the W history slots + zero padding for the K+pad slots.
    // It is the input to input_proj.

    // MARK: - init

    public init(cfg: DonkeyConfig, weights: DonkeyWorldWeights) throws {
        self.cfg = cfg
        let D = cfg.architecture.hidden_dim
        let F_ = cfg.architecture.ffn_dim
        let TH = cfg.trunk.hidden_dim
        let SP = cfg.sequence.spatial_pad
        let W  = cfg.sequence.window_size
        let K  = cfg.sequence.draft_size
        let OUT = cfg.out_ch                 // trunk_hidden + out_conf_dim

        // === Validate weight shapes ===
        guard weights.inputProjLnGamma.count == D else {
            throw DonkeyWorldError.shapeMismatch(
                "inputProjLnGamma: expected \(D), got \(weights.inputProjLnGamma.count)")
        }
        guard weights.inputProjLnBeta.count == D else {
            throw DonkeyWorldError.shapeMismatch(
                "inputProjLnBeta: expected \(D), got \(weights.inputProjLnBeta.count)")
        }
        guard weights.inputProj.count == D * TH else {
            throw DonkeyWorldError.shapeMismatch(
                "input_proj: expected \(D * TH), got \(weights.inputProj.count)")
        }
        guard weights.draftQueries.count == D * K else {
            throw DonkeyWorldError.shapeMismatch(
                "draft_queries: expected \(D * K), got \(weights.draftQueries.count)")
        }
        guard weights.positionalBias.count == D * SP else {
            throw DonkeyWorldError.shapeMismatch(
                "positional_bias: expected \(D * SP), got \(weights.positionalBias.count)")
        }
        guard weights.layers.count == cfg.architecture.n_layers else {
            throw DonkeyWorldError.shapeMismatch(
                "layers: expected \(cfg.architecture.n_layers), got \(weights.layers.count)")
        }
        guard weights.gammaFinal.count == D else {
            throw DonkeyWorldError.shapeMismatch("gammaFinal: expected \(D)")
        }
        guard weights.head.count == OUT * D else {
            throw DonkeyWorldError.shapeMismatch(
                "head: expected \(OUT * D), got \(weights.head.count)")
        }

        // === Compile kernels ===
        try aneBridgeInit()

        kInputProj = try Self.compileLinear1x1(W: weights.inputProj, inCh: TH, outCh: D, sp: SP)
        self.inputProjLnGamma = weights.inputProjLnGamma
        self.inputProjLnBeta  = weights.inputProjLnBeta

        var kQs = [ANEKernel](), kKs = [ANEKernel](), kVs = [ANEKernel]()
        var kOs = [ANEKernel](), kSDPAs = [ANEKernel]()
        var kUps = [ANEKernel](), kDowns = [ANEKernel]()
        for l in 0..<cfg.architecture.n_layers {
            let lw = weights.layers[l]
            // Shape sanity per layer.
            guard lw.gammaAtt.count == D, lw.gammaFfn.count == D,
                  lw.Wq.count == D * D, lw.Wk.count == D * D, lw.Wv.count == D * D, lw.Wo.count == D * D,
                  lw.Wup.count == F_ * D, lw.Wdown.count == D * F_ else {
                throw DonkeyWorldError.shapeMismatch("layer \(l): a weight has wrong size")
            }
            kQs.append(try Self.compileLinear1x1(W: lw.Wq, inCh: D, outCh: D, sp: SP))
            kKs.append(try Self.compileLinear1x1(W: lw.Wk, inCh: D, outCh: D, sp: SP))
            kVs.append(try Self.compileLinear1x1(W: lw.Wv, inCh: D, outCh: D, sp: SP))
            kOs.append(try Self.compileLinear1x1(W: lw.Wo, inCh: D, outCh: D, sp: SP))
            kSDPAs.append(try Self.compileSDPA(cfg: cfg))
            kUps.append(try Self.compileLinear1x1(W: lw.Wup, inCh: D, outCh: F_, sp: SP))
            kDowns.append(try Self.compileLinear1x1(W: lw.Wdown, inCh: F_, outCh: D, sp: SP))
        }
        kQ = kQs; kK = kKs; kV = kVs; kO = kOs
        kSDPA = kSDPAs; kFFNUp = kUps; kFFNDown = kDowns
        kHead = try Self.compileLinear1x1(W: weights.head, inCh: D, outCh: OUT, sp: SP)

        // === Stash CPU constants ===
        gammaAtt = weights.layers.map { $0.gammaAtt }
        gammaFfn = weights.layers.map { $0.gammaFfn }
        gammaFinal = weights.gammaFinal
        draftQueries = weights.draftQueries
        positionalBias = weights.positionalBias

        // === Allocate buffers ===
        bufHistory = [Float](repeating: 0, count: TH * SP)
        bufHidden  = [Float](repeating: 0, count: D * SP)
        bufHidden2 = [Float](repeating: 0, count: D * SP)
        bufNorm    = [Float](repeating: 0, count: D * SP)
        bufQ       = [Float](repeating: 0, count: D * SP)
        bufK       = [Float](repeating: 0, count: D * SP)
        bufV       = [Float](repeating: 0, count: D * SP)
        bufQKV     = [Float](repeating: 0, count: 3 * D * SP)
        bufAttn    = [Float](repeating: 0, count: D * SP)
        bufOO      = [Float](repeating: 0, count: D * SP)
        bufHUp     = [Float](repeating: 0, count: F_ * SP)
        bufHSilu   = [Float](repeating: 0, count: F_ * SP)
        bufHDown   = [Float](repeating: 0, count: D * SP)
        bufHeadOut = [Float](repeating: 0, count: OUT * SP)
        scratchRms = [Float](repeating: 0, count: SP)
        lnScratchMean = [Float](repeating: 0, count: SP)
        lnScratchVar  = [Float](repeating: 0, count: SP)
        scratchSilu = [Float](repeating: 0, count: 4096)

        // History state initialised on first forward() call (cold-start logic
        // lives there). For now, just record we have no history.
        firstHidden = nil
        realHistoryCount = 0
        _ = W  // silence unused warning (used by forward() in next commit)
    }

    // MARK: - forward
    //
    // Per-call data flow documented at the top of the file.
    //
    // Side effects: pushes currentHidden into the rolling history buffer.
    // Caller must call reset() between independent sessions.
    // MARK: - forward
    //
    // Per-call data flow documented at the top of the file.
    //
    // This is the depth-1 entrypoint: one forward pass produces K parallel
    // predictions from the real current trunk hidden. cfg.architecture.rolloutDepth
    // is currently asserted to be 1; depth>1 (tree decoding via re-feeding
    // predicted hiddens) is a future addition that would live alongside this
    // method, not replace it.
    //
    // Side effects: pushes currentHidden into the rolling history buffer.
    // Caller must call reset() between independent sessions.
    public func forward(currentHidden: [Float]) throws -> DonkeyDraft {
        // Depth-1 only for now. Higher depths will be implemented as a
        // separate forwardSpeculative(predictedHidden:) entrypoint or as a
        // wrapping orchestrator that calls this method multiple times.
        precondition(cfg.architecture.rolloutDepth == 1,
                     "rollout_depth > 1 not implemented yet (got \(cfg.architecture.rolloutDepth))")
        let cfg = self.cfg
        let D   = cfg.architecture.hiddenDim
        let F_  = cfg.architecture.ffnDim
        let TH  = cfg.trunk.hiddenDim
        let SP  = cfg.sequence.spatialPad
        let W   = cfg.sequence.windowSize
        let K   = cfg.sequence.draftSize
        let OUT = cfg.outCh

        guard currentHidden.count == TH else {
            throw DonkeyWorldError.shapeMismatch(
                "currentHidden: expected \(TH), got \(currentHidden.count)")
        }

        // === 1. History buffer update (with cold-start replication) ===
        //
        // bufHistory layout: row-major [TH, SP]. Positions 0..W-1 hold the
        // W most recent trunk lastHiddenStates (with replication of
        // firstHidden when realHistoryCount < W). Positions W..SP-1 hold
        // zeros (overwritten by draftQueries after input_proj).
        if firstHidden == nil {
            // First-ever call this session: replicate currentHidden across all W slots.
            firstHidden = currentHidden
            for c in 0..<TH {
                let row = c * SP
                for s in 0..<W { bufHistory[row + s] = currentHidden[c] }
                for s in W..<SP { bufHistory[row + s] = 0 }
            }
            realHistoryCount = 1
        } else {
            // Steady-state slide-left + place new at slot W-1.
            // (Cold-start when realHistoryCount<W is the same operation:
            // leading slots still hold firstHidden from earlier calls.)
            for c in 0..<TH {
                let row = c * SP
                for s in 0..<(W - 1) { bufHistory[row + s] = bufHistory[row + s + 1] }
                bufHistory[row + (W - 1)] = currentHidden[c]
            }
            if realHistoryCount < W { realHistoryCount += 1 }
        }

        // === 2. input_proj ANE: [TH, SP] -> [D, SP] ===
        try bufHistory.withUnsafeBytes {
            try kInputProj.writeInput(0, $0.baseAddress!, bytes: $0.count)
        }
        try kInputProj.eval()
        try bufHidden.withUnsafeMutableBytes {
            try kInputProj.readOutput(0, into: $0.baseAddress!, bytes: $0.count)
        }
        // === 2b. LayerNorm over channel axis (CPU, v3 JEPA scale fix) ===
        // bufHidden is [D, SP]; mean/var per spatial position; affine [D]
        bufHidden.withUnsafeMutableBufferPointer { hpBuf in
            inputProjLnGamma.withUnsafeBufferPointer { g in
                inputProjLnBeta.withUnsafeBufferPointer { b in
                    cpu_layernorm(
                        x: hpBuf.baseAddress!,
                        gamma: g.baseAddress!,
                        beta:  b.baseAddress!,
                        out: hpBuf.baseAddress!,
                        ch: D, sp: SP,
                        eps: cfg.architecture.rmsEps,
                        scratchMean: &lnScratchMean,
                        scratchVar:  &lnScratchVar)
                }
            }
        }

        // === 3. Overwrite draft slots W..W+K-1 with learned draftQueries ===
        // draftQueries layout: [D, K] channel-major.
        for c in 0..<D {
            for k in 0..<K {
                bufHidden[c * SP + (W + k)] = draftQueries[c * K + k]
            }
        }
        // Slots W+K..SP-1 stay at the input_proj output (mostly zero from
        // the zero-input region); causal mask makes them irrelevant.

        // === 4. Add positional bias ===
        for i in 0..<(D * SP) { bufHidden[i] += positionalBias[i] }

        // === 5. Transformer layers ===
        for l in 0..<cfg.architecture.nLayers {
            // Save residual (vDSP_mmov: hand-tuned SIMD memcpy).
            bufHidden.withUnsafeBufferPointer { sb in
                bufHidden2.withUnsafeMutableBufferPointer { db in
                    vDSP_mmov(sb.baseAddress!, db.baseAddress!,
                              vDSP_Length(D * SP), 1, vDSP_Length(D * SP), vDSP_Length(D * SP))
                }
            }

            // 5a. RMSNorm(att)
            bufHidden.withUnsafeBufferPointer { xb in
                gammaAtt[l].withUnsafeBufferPointer { gb in
                    bufNorm.withUnsafeMutableBufferPointer { ob in
                        cpu_rmsnorm(x: xb.baseAddress!, gamma: gb.baseAddress!,
                                    out: ob.baseAddress!, ch: D, sp: SP,
                                    eps: cfg.architecture.rmsEps, scratch: &scratchRms)
                    }
                }
            }

            // 5b. Q, K, V projections
            try evalLinear(kernel: kQ[l], src: &bufNorm, dst: &bufQ, inCh: D, outCh: D, sp: SP)
            try evalLinear(kernel: kK[l], src: &bufNorm, dst: &bufK, inCh: D, outCh: D, sp: SP)
            try evalLinear(kernel: kV[l], src: &bufNorm, dst: &bufV, inCh: D, outCh: D, sp: SP)

            // 5c. Concat Q | K | V along channel axis.
            bufQKV.withUnsafeMutableBufferPointer { dst in
                bufQ.withUnsafeBufferPointer { src in dst.baseAddress!.update(from: src.baseAddress!, count: D * SP) }
                bufK.withUnsafeBufferPointer { src in (dst.baseAddress! + D * SP).update(from: src.baseAddress!, count: D * SP) }
                bufV.withUnsafeBufferPointer { src in (dst.baseAddress! + 2 * D * SP).update(from: src.baseAddress!, count: D * SP) }
            }

            // 5d. SDPA
            try bufQKV.withUnsafeBytes { try kSDPA[l].writeInput(0, $0.baseAddress!, bytes: $0.count) }
            try kSDPA[l].eval()
            try bufAttn.withUnsafeMutableBytes { try kSDPA[l].readOutput(0, into: $0.baseAddress!, bytes: $0.count) }

            // 5e. Wo projection
            try evalLinear(kernel: kO[l], src: &bufAttn, dst: &bufOO, inCh: D, outCh: D, sp: SP)

            // 5f. Residual: bufHidden = bufHidden2 + bufOO
            bufHidden2.withUnsafeBufferPointer { ab in
                bufOO.withUnsafeBufferPointer { bb in
                    bufHidden.withUnsafeMutableBufferPointer { ob in
                        cpu_residual_add(a: ab.baseAddress!, b: bb.baseAddress!,
                                         out: ob.baseAddress!, count: D * SP)
                    }
                }
            }

            // Save residual #2.
            bufHidden.withUnsafeBufferPointer { sb in
                bufHidden2.withUnsafeMutableBufferPointer { db in
                    vDSP_mmov(sb.baseAddress!, db.baseAddress!,
                              vDSP_Length(D * SP), 1, vDSP_Length(D * SP), vDSP_Length(D * SP))
                }
            }

            // 5g. RMSNorm(ffn)
            bufHidden.withUnsafeBufferPointer { xb in
                gammaFfn[l].withUnsafeBufferPointer { gb in
                    bufNorm.withUnsafeMutableBufferPointer { ob in
                        cpu_rmsnorm(x: xb.baseAddress!, gamma: gb.baseAddress!,
                                    out: ob.baseAddress!, ch: D, sp: SP,
                                    eps: cfg.architecture.rmsEps, scratch: &scratchRms)
                    }
                }
            }

            // 5h. FFN up
            try evalLinear(kernel: kFFNUp[l], src: &bufNorm, dst: &bufHUp, inCh: D, outCh: F_, sp: SP)

            // 5i. SiLU
            bufHUp.withUnsafeBufferPointer { xb in
                bufHSilu.withUnsafeMutableBufferPointer { ob in
                    cpu_silu(x: xb.baseAddress!, out: ob.baseAddress!,
                             count: F_ * SP, scratch: &scratchSilu)
                }
            }

            // 5j. FFN down
            try evalLinear(kernel: kFFNDown[l], src: &bufHSilu, dst: &bufHDown, inCh: F_, outCh: D, sp: SP)

            // 5k. Residual
            bufHidden2.withUnsafeBufferPointer { ab in
                bufHDown.withUnsafeBufferPointer { bb in
                    bufHidden.withUnsafeMutableBufferPointer { ob in
                        cpu_residual_add(a: ab.baseAddress!, b: bb.baseAddress!,
                                         out: ob.baseAddress!, count: D * SP)
                    }
                }
            }
        }

        // === 6. Final RMSNorm ===
        bufHidden.withUnsafeBufferPointer { xb in
            gammaFinal.withUnsafeBufferPointer { gb in
                bufNorm.withUnsafeMutableBufferPointer { ob in
                    cpu_rmsnorm(x: xb.baseAddress!, gamma: gb.baseAddress!,
                                out: ob.baseAddress!, ch: D, sp: SP,
                                eps: cfg.architecture.rmsEps, scratch: &scratchRms)
                }
            }
        }

        // === 7. Head: [D, SP] -> [OUT=TH+out_conf, SP] ===
        try evalLinear(kernel: kHead, src: &bufNorm, dst: &bufHeadOut, inCh: D, outCh: OUT, sp: SP)

        // === 8. Extract draft-slot outputs: positions W..W+K-1 ===
        // bufHeadOut layout: [OUT, SP] channel-major.
        // Channels 0..TH-1 = predicted next-trunk-hidden;
        // channels TH..OUT-1 = raw confidence logits (OUT_CONF channels).
        var predDraft = [Float](repeating: 0, count: TH * K)
        var confDraft = [Float](repeating: 0, count: K)
        for c in 0..<TH {
            for k in 0..<K {
                predDraft[c * K + k] = bufHeadOut[c * SP + (W + k)]
            }
        }
        for k in 0..<K {
            confDraft[k] = cpu_sigmoid(bufHeadOut[TH * SP + (W + k)])
        }
        return DonkeyDraft(predHidden: predDraft, confidence: confDraft)
    }

    public func reset() {
        firstHidden = nil
        realHistoryCount = 0
        for i in 0..<bufHistory.count { bufHistory[i] = 0 }
    }

    // === Private helper: linear1x1 dispatch with shape assertions. ===
    private func evalLinear(kernel: ANEKernel,
                            src: inout [Float], dst: inout [Float],
                            inCh: Int, outCh: Int, sp: Int) throws {
        precondition(src.count == inCh * sp, "evalLinear: src size \(src.count) != \(inCh * sp)")
        precondition(dst.count == outCh * sp, "evalLinear: dst size \(dst.count) != \(outCh * sp)")
        try src.withUnsafeBytes { try kernel.writeInput(0, $0.baseAddress!, bytes: $0.count) }
        try kernel.eval()
        try dst.withUnsafeMutableBytes { try kernel.readOutput(0, into: $0.baseAddress!, bytes: $0.count) }
    }

    // MARK: - Kernel compilation helpers (static, called from init)

    private static func loadTemplate(_ path: String) throws -> String {
        let url = URL(fileURLWithPath: path)
        do {
            return try String(contentsOf: url, encoding: .utf8)
        } catch {
            throw DonkeyWorldError.compileFailed("load template \(path): \(error)")
        }
    }

    private static func compileLinear1x1(W: [Float], inCh: Int, outCh: Int, sp: Int) throws -> ANEKernel {
        var mil = try loadTemplate("donkey-trainer/kernels/linear1x1.mil.template")
        mil = mil.replacingOccurrences(of: "{IN_CH}",  with: String(inCh))
        mil = mil.replacingOccurrences(of: "{OUT_CH}", with: String(outCh))
        mil = mil.replacingOccurrences(of: "{SP}",     with: String(sp))
        let blob = aneBuildWeightBlobFP16(W, rows: outCh, cols: inCh)
        return try aneCompile(
            milText: mil,
            weights: [("@model_path/weights/weight.bin", blob)],
            inputBytes:  [inCh * sp * 4],
            outputBytes: [outCh * sp * 4]
        )
    }

    private static func compileSDPA(cfg: DonkeyConfig) throws -> ANEKernel {
        let D = cfg.architecture.hidden_dim
        let H = cfg.architecture.heads
        let HD = cfg.architecture.head_dim
        let SP = cfg.sequence.spatial_pad
        var mil = try loadTemplate("donkey-trainer/kernels/sdpa.mil.template")
        let scale = 1.0 / Float(HD).squareRoot()
        mil = mil.replacingOccurrences(of: "{DIM3}",  with: String(3 * D))
        mil = mil.replacingOccurrences(of: "{DIM2}",  with: String(2 * D))
        mil = mil.replacingOccurrences(of: "{DIM}",   with: String(D))
        mil = mil.replacingOccurrences(of: "{HEADS}", with: String(H))
        mil = mil.replacingOccurrences(of: "{HD}",    with: String(HD))
        mil = mil.replacingOccurrences(of: "{SP}",    with: String(SP))
        mil = mil.replacingOccurrences(of: "{SCALE}", with: String(format: "%f", scale))

        // Causal mask blob (same convention as v1 SDPA).
        var mask = [Float](repeating: 0, count: SP * SP)
        for r in 0..<SP {
            for c in 0..<SP {
                mask[r * SP + c] = (c <= r) ? 0.0 : -65504.0
            }
        }
        let maskBlob = aneBuildWeightBlobFP16(mask, rows: SP, cols: SP)
        return try aneCompile(
            milText: mil,
            weights: [("@model_path/weights/weight.bin", maskBlob)],
            inputBytes:  [3 * D * SP * 4],
            outputBytes: [D * SP * 4]
        )
    }
}

// Swift access to DonkeyConfig's CodingKeys-aliased field names. The struct
// stores hidden_dim as `hiddenDim` but the spec JSON uses snake_case. Keep
// our orchestrator readable: use `cfg.architecture.hidden_dim` style by
// exposing snake_case computed properties on a small extension.
extension DonkeyConfig.ArchitectureConfig {
    var hidden_dim: Int { hiddenDim }
    var ffn_dim: Int    { ffnDim }
    var head_dim: Int   { headDim }
    var n_layers: Int   { nLayers }
    var out_conf_dim: Int { outConfDim }
    var rms_eps: Float  { rmsEps }
}
extension DonkeyConfig.SequenceConfig {
    var spatial_pad: Int { spatialPad }
    var window_size: Int { windowSize }
    var draft_size: Int  { draftSize }
}
extension DonkeyConfig.TrunkConfig {
    var hidden_dim: Int { hiddenDim }
}
extension DonkeyConfig {
    public var out_ch: Int { outCh }
}
