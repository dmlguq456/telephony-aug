"""telephony_aug — CPU-side telephony augmentation utility for ML training pipelines.

Quick start:
    >>> import soundfile as sf
    >>> from telephony_aug import TelephonyAugmentation
    >>> audio, sr = sf.read("input.wav", dtype="float32")
    >>> aug = TelephonyAugmentation("config.yaml")   # or pass a dict
    >>> out = aug(audio, sr)                          # apply CPU pipeline
    >>> out = aug(audio, sr, rir=my_rir)              # runtime RIR override

Pipeline stages (in order, each gated by `prob`):
    rir → near_end_dsp → codec → packet_loss_plc → far_end_dsp →
    bandwidth_switch → mic_capture_cpu → (final_normalize, always applied)

Telephony categories covered: PSTN G.711 / Cellular AMR-NB·WB / VoIP Opus·G.722.

See examples/usage.py for the DataLoader integration pattern (worker_init_fn
+ set_random_state for reproducibility).
"""

from .augmentation import (
    CodecSampler,
    DistortionLogger,
    RIRSampler,
    TelephonyAugmentation,
    codec_roundtrip,
    g711_app_i_plc_fill,
    gilbert_elliott_mask,
)

# Module-level flags (test / docs may probe these)
from .augmentation import (  # noqa: F401
    _HAS_AUDIOMENTATIONS,
    _HAS_G711,
    _HAS_G722,
    _HAS_OPUSLIB,
    _HAS_PYROOMACOUSTICS,
    _HAS_PYRNNOISE,
    _HAS_TORCHAUDIO,
    _HAS_WEBRTC,
)

__version__ = "0.1.0"

__all__ = [
    "TelephonyAugmentation",
    "RIRSampler",
    "CodecSampler",
    "codec_roundtrip",
    "gilbert_elliott_mask",
    "g711_app_i_plc_fill",
    "DistortionLogger",
    "__version__",
]
