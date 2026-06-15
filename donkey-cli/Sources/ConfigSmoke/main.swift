// ConfigSmoke — verify DonkeyConfig parses the canonical spec, enforces
// invariants, and rejects deliberately invalid configs. First test in the
// v2 accumulating regression harness.
import Foundation
import DonkeyConfig

fileprivate func slog(_ s: String) {
    FileHandle.standardError.write(Data("\(s)\n".utf8))
}

fileprivate func fail(_ msg: String, _ code: Int32) -> Never {
    slog("[config] FAIL: \(msg)")
    exit(code)
}

@main
struct ConfigSmoke {
    static func main() {
        let specPath = ProcessInfo.processInfo.environment["DONKEY_SPEC"]
            ?? "spec/donkey_v2_default.json"
        let url = URL(fileURLWithPath: specPath)

        // === 1. Positive case: canonical spec must load + validate.
        let cfg: DonkeyConfig
        do {
            cfg = try DonkeyConfig(jsonURL: url)
        } catch {
            fail("load \(specPath): \(error)", 1)
        }
        slog("[config] loaded \(specPath)")
        slog("[config]   model_name        \(cfg.modelName)")
        slog("[config]   trunk             \(cfg.trunk.name) hidden=\(cfg.trunk.hiddenDim) vocab=\(cfg.trunk.vocabSize)")
        slog("[config]   architecture      hidden=\(cfg.architecture.hiddenDim) ffn=\(cfg.architecture.ffnDim) heads=\(cfg.architecture.heads) hd=\(cfg.architecture.headDim) layers=\(cfg.architecture.nLayers)")
        slog("[config]   sequence          W=\(cfg.sequence.windowSize) K=\(cfg.sequence.draftSize) SP=\(cfg.sequence.spatialPad)")
        slog("[config]   derived           qkvCh=\(cfg.qkvCh) outCh=\(cfg.outCh)")

        // Identity check on a couple of fields (catches accidental rename / mis-map).
        if cfg.architecture.hiddenDim != cfg.architecture.heads * cfg.architecture.headDim {
            fail("self-consistency: hidden_dim != heads * head_dim", 2)
        }
        if cfg.sequence.windowSize + cfg.sequence.draftSize > cfg.sequence.spatialPad {
            fail("self-consistency: W + K > SP", 3)
        }
        // Derived-size sanity. outCh MUST equal trunk.hidden_dim + out_conf_dim,
        // not architecture.hidden_dim — the head projects back UP to trunk size
        // so verification through the trunk's lm_head works.
        let expectedOutCh = cfg.trunk.hiddenDim + cfg.architecture.outConfDim
        if cfg.outCh != expectedOutCh {
            fail("derived: outCh = \(cfg.outCh), expected trunk(\(cfg.trunk.hiddenDim)) + outConf(\(cfg.architecture.outConfDim)) = \(expectedOutCh)", 8)
        }
        if cfg.qkvCh != 3 * cfg.architecture.hiddenDim {
            fail("derived: qkvCh != 3 * hidden_dim", 9)
        }

        // === 2. Negative case A: invariant violation (hidden_dim != heads * head_dim)
        let badInvariant = """
        {
            "schema_version": 1, "model_name": "bad",
            "trunk": {"hidden_dim": 4096, "vocab_size": 152576, "name": "T"},
            "architecture": {
                "hidden_dim": 1024, "ffn_dim": 4096, "heads": 8, "head_dim": 64,
                "n_layers": 2, "out_conf_dim": 1, "ffn_activation": "silu", "rms_eps": 1.0e-6
            },
            "sequence": {"window_size": 13, "draft_size": 3, "spatial_pad": 16},
            "training": {"loss_l2": 1.0, "loss_cos": 0.1, "loss_ce": 0.5, "loss_calib": 0.01,
                         "learning_rate": 3.0e-4, "lr_warmup_steps": 200, "lr_final": 1.0e-4, "ewma_decay": 0.999}
        }
        """
        var threw = false
        do {
            _ = try DonkeyConfig(jsonData: badInvariant.data(using: .utf8)!)
        } catch {
            threw = true
            slog("[config] (expected) rejected bad invariant: \(error)")
        }
        if !threw { fail("invariant: should have thrown for heads*head_dim != hidden_dim", 4) }

        // === 3. Negative case B: unsupported SP (e.g. 24)
        let badSP = """
        {
            "schema_version": 1, "model_name": "bad_sp",
            "trunk": {"hidden_dim": 4096, "vocab_size": 152576, "name": "T"},
            "architecture": {
                "hidden_dim": 1024, "ffn_dim": 4096, "heads": 16, "head_dim": 64,
                "n_layers": 2, "out_conf_dim": 1, "ffn_activation": "silu", "rms_eps": 1.0e-6
            },
            "sequence": {"window_size": 13, "draft_size": 3, "spatial_pad": 24},
            "training": {"loss_l2": 1.0, "loss_cos": 0.1, "loss_ce": 0.5, "loss_calib": 0.01,
                         "learning_rate": 3.0e-4, "lr_warmup_steps": 200, "lr_final": 1.0e-4, "ewma_decay": 0.999}
        }
        """
        threw = false
        do {
            _ = try DonkeyConfig(jsonData: badSP.data(using: .utf8)!)
        } catch {
            threw = true
            slog("[config] (expected) rejected bad SP: \(error)")
        }
        if !threw { fail("unsupported value: should have rejected SP=24", 5) }

        // === 4. Negative case C: schema_version mismatch
        let badSchema = """
        {
            "schema_version": 999, "model_name": "bad_schema",
            "trunk": {"hidden_dim": 4096, "vocab_size": 152576, "name": "T"},
            "architecture": {
                "hidden_dim": 1024, "ffn_dim": 4096, "heads": 16, "head_dim": 64,
                "n_layers": 2, "out_conf_dim": 1, "ffn_activation": "silu", "rms_eps": 1.0e-6
            },
            "sequence": {"window_size": 13, "draft_size": 3, "spatial_pad": 16},
            "training": {"loss_l2": 1.0, "loss_cos": 0.1, "loss_ce": 0.5, "loss_calib": 0.01,
                         "learning_rate": 3.0e-4, "lr_warmup_steps": 200, "lr_final": 1.0e-4, "ewma_decay": 0.999}
        }
        """
        threw = false
        do {
            _ = try DonkeyConfig(jsonData: badSchema.data(using: .utf8)!)
        } catch {
            threw = true
            slog("[config] (expected) rejected bad schema_version: \(error)")
        }
        if !threw { fail("schema_version: should have rejected v999", 6) }

        slog("[config] all checks passed")
    }
}
