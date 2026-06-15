"""DonkeyConfig — Python loader for the canonical donkey v2 spec.

Mirrors Swift's DonkeyConfig.swift. Same JSON file is the single source of
truth; this file just exposes it as a typed Python dataclass and enforces
the same invariants.
"""
from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import List


class DonkeyConfigError(Exception):
    pass


@dataclass
class TrunkConfig:
    hidden_dim: int
    vocab_size: int
    name: str


@dataclass
class ArchitectureConfig:
    hidden_dim: int
    ffn_dim: int
    heads: int
    head_dim: int
    n_layers: int
    out_conf_dim: int
    ffn_activation: str
    rms_eps: float
    # Rollout depth: 1 = single forward (parallel-K). >1 = tree decoding, future option.
    rollout_depth: int = 1


@dataclass
class SequenceConfig:
    window_size: int
    draft_size: int
    spatial_pad: int


@dataclass
class TrainingConfig:
    loss_l2: float
    loss_cos: float
    loss_ce: float
    loss_calib: float
    learning_rate: float
    lr_warmup_steps: int
    lr_final: float
    ewma_decay: float


@dataclass
class DonkeyConfig:
    schema_version: int
    model_name: str
    trunk: TrunkConfig
    architecture: ArchitectureConfig
    sequence: SequenceConfig
    training: TrainingConfig

    @classmethod
    def from_json(cls, path) -> "DonkeyConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "DonkeyConfig":
        try:
            cfg = cls(
                schema_version=data["schema_version"],
                model_name=data["model_name"],
                trunk=TrunkConfig(**{k: v for k, v in data["trunk"].items()
                                     if k in TrunkConfig.__dataclass_fields__}),
                architecture=ArchitectureConfig(**{k: v for k, v in data["architecture"].items()
                                                   if k in ArchitectureConfig.__dataclass_fields__}),
                sequence=SequenceConfig(**{k: v for k, v in data["sequence"].items()
                                           if k in SequenceConfig.__dataclass_fields__}),
                training=TrainingConfig(**{k: v for k, v in data["training"].items()
                                           if k in TrainingConfig.__dataclass_fields__}),
            )
        except (KeyError, TypeError) as e:
            raise DonkeyConfigError(f"invalid JSON structure: {e}")
        cfg.validate()
        return cfg

    def validate(self):
        if self.schema_version != 1:
            raise DonkeyConfigError(f"schema_version: expected 1, got {self.schema_version}")

        a = self.architecture
        if a.hidden_dim != a.heads * a.head_dim:
            raise DonkeyConfigError(
                f"architecture.hidden_dim ({a.hidden_dim}) != heads ({a.heads}) * head_dim ({a.head_dim})")
        if a.heads <= 0 or a.head_dim <= 0 or a.n_layers <= 0:
            raise DonkeyConfigError("architecture: heads/head_dim/n_layers must be > 0")
        if a.rollout_depth < 1:
            raise DonkeyConfigError(f"architecture.rollout_depth: must be >= 1, got {a.rollout_depth}")

        s = self.sequence
        if s.window_size + s.draft_size > s.spatial_pad:
            raise DonkeyConfigError(
                f"sequence: window_size ({s.window_size}) + draft_size ({s.draft_size}) > spatial_pad ({s.spatial_pad})")
        allowed_sp = [16, 32, 64, 128]
        if s.spatial_pad not in allowed_sp:
            raise DonkeyConfigError(
                f"sequence.spatial_pad: must be one of {allowed_sp}, got {s.spatial_pad}")
        if s.window_size <= 0 or s.draft_size <= 0:
            raise DonkeyConfigError("sequence: window_size and draft_size must be > 0")

        if a.ffn_activation != "silu":
            raise DonkeyConfigError(
                f"architecture.ffn_activation: only 'silu' supported in v2.0, got '{a.ffn_activation}'")

        if self.trunk.hidden_dim <= 0 or self.trunk.vocab_size <= 0:
            raise DonkeyConfigError("trunk: dimensions must be > 0")

        t = self.training
        for name, val in [("loss_l2", t.loss_l2), ("loss_cos", t.loss_cos),
                          ("loss_ce", t.loss_ce), ("loss_calib", t.loss_calib)]:
            if val < 0:
                raise DonkeyConfigError(f"training.{name}: must be >= 0, got {val}")
        if t.learning_rate <= 0 or t.lr_final <= 0 or t.lr_warmup_steps < 0:
            raise DonkeyConfigError("training: learning rates must be > 0, warmup >= 0")
        if not (0 <= t.ewma_decay <= 1):
            raise DonkeyConfigError(f"training.ewma_decay: must be in [0,1], got {t.ewma_decay}")

    # === Derived sizes (must match Swift's computed properties) ===
    @property
    def qkv_ch(self) -> int:
        return 3 * self.architecture.hidden_dim

    @property
    def out_ch(self) -> int:
        # Head projects donkey's working dim back UP to trunk.hidden_dim so
        # output can flow through trunk's lm_head for verification, plus
        # confidence channel(s).
        return self.trunk.hidden_dim + self.architecture.out_conf_dim
