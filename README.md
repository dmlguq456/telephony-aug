# telephony-aug

CPU-side **telephony augmentation** utility for ML training pipelines. Simulates the distortions a clean speech signal accumulates while traversing a telephony chain — codec compression, packet loss, device DSP cascade, bandwidth limits, and room/handset acoustics — so models trained on clean studio data can generalize to PSTN / cellular / VoIP audio.

Designed as a lightweight `Dataset.__getitem__` utility, no GPU dependency.

## Highlights

- **7-stage CPU chain**: `rir → near_end_dsp → codec → packet_loss_plc → far_end_dsp → bandwidth_switch → mic_capture_cpu` (+ always-on `final_normalize`)
- **4 telephony categories**: PSTN G.711 μ/A-law · Cellular AMR-NB/WB · VoIP Opus·G.722 (hierarchical sampler with per-category bitrate distributions)
- **Runtime RIR override**: pass your own IR per call: `aug(audio, sr, rir=my_rir)`
- **audiomentations-compatible**: `Compose([...])` works via `samples=`/`sample_rate=` kwargs
- **Graceful degradation**: every optional dep is `try/except`-guarded; missing libs disable the corresponding stage instead of raising
- **Reproducibility**: `set_random_state(rng)` for DataLoader `worker_init_fn` pattern

## Install

```bash
pip install git+https://github.com/dmlguq456/telephony-aug.git

# Or for development:
git clone https://github.com/dmlguq456/telephony-aug.git
cd telephony-aug
pip install -e .
```

Optional dependency groups (each stage's backend is opt-in):

```bash
pip install "telephony-aug[codec]"        # opuslib + g711 + libg722 direct bindings
pip install "telephony-aug[dsp]"          # pyrnnoise + webrtc-noise-gain
pip install "telephony-aug[rir]"          # pyroomacoustics
pip install "telephony-aug[audiomentations]"  # for Compose pattern
pip install "telephony-aug[eval]"         # pesq
pip install "telephony-aug[all]"          # everything except dev tooling
```

System dependency: **ffmpeg** (used by `torchaudio.io.AudioEffector` as codec backend). Verify with `ffmpeg -encoders | grep -E "opus|amr|g722"`.

## Quick start

```python
import soundfile as sf
import numpy as np
from telephony_aug import TelephonyAugmentation

audio, sr = sf.read("input.wav", dtype="float32")

# Pass a YAML path or a dict; see config.yaml for the default schema
aug = TelephonyAugmentation("config.yaml")

# CPU pipeline (called inside Dataset.__getitem__)
out = aug(audio, sr)

# audiomentations Compose alias (samples/sample_rate kwargs)
out = aug(samples=audio, sample_rate=sr)

# Runtime RIR override — bypass internal sampler with caller-provided IR
my_rir = np.load("my_rirs/00042.npy")  # 1D float32; sr must match audio sr
out = aug(audio, sr, rir=my_rir)

# Optional defensive sr-mismatch check
out = aug(audio, sr, rir=my_rir, rir_sr=sr)   # raises ValueError if rir_sr != sr
```

## DataLoader integration

```python
import torch
from torch.utils.data import Dataset, DataLoader
from telephony_aug import TelephonyAugmentation

class TelephonyDataset(Dataset):
    def __init__(self, audio_paths, aug):
        self.paths = audio_paths
        self.aug = aug
    def __getitem__(self, idx):
        audio, sr = sf.read(self.paths[idx], dtype="float32")
        audio = self.aug(audio, sr)
        return torch.from_numpy(audio), sr
    def __len__(self):
        return len(self.paths)

def make_worker_init_fn(base_seed: int):
    """Seed all RNG sources per worker for reproducible augmentation."""
    def _init(worker_id: int):
        import random, numpy as np
        seed = base_seed + worker_id
        random.seed(seed)
        np.random.seed(seed)
        info = torch.utils.data.get_worker_info()
        if info is not None and hasattr(info.dataset, "aug"):
            info.dataset.aug.set_random_state(np.random.RandomState(seed))
    return _init

aug = TelephonyAugmentation("config.yaml")
ds = TelephonyDataset(paths, aug)
loader = DataLoader(
    ds, batch_size=4, num_workers=4,
    worker_init_fn=make_worker_init_fn(base_seed=42),
    persistent_workers=True,
)
```

## Pipeline stages

| Stage | What it does | Key config keys |
|---|---|---|
| `rir` | Convolve with room impulse response (manifest / ISM / handset dirac) | `manifest_path`, `rir_dir`, `room_type_split`, `wet_ratio_range` |
| `near_end_dsp` | WebRTC APM (NS/AGC) **before** codec | `noise_suppression_range`, `auto_gain_range` |
| `codec` | Hierarchical telephony codec roundtrip (PSTN/Cellular/VoIP) | `category_probs`, `codec_tree` (per-category bitrate ranges/modes) |
| `packet_loss_plc` | Gilbert-Elliott loss mask + G.711 App I PLC fill | `frame_size_ms`, `p_good_to_bad_range`, `p_bad_to_good_range`, `plc_weights` |
| `far_end_dsp` | WebRTC APM / RNNoise / none (post-codec cascade artifact) | `backend_weights` |
| `bandwidth_switch` | NB(8k)/WB(16k)/FB(48k) downsample→upsample roundtrip | `target_modes`, `target_weights` |
| `mic_capture_cpu` | Hard clipping + linear gain (audiomentations Compose if available) | `clip_prob`, `clip_threshold_range`, `gain_db_range`, `gain_prob` |
| `final_normalize` | Peak-normalize to `target_peak` (always applied unless disabled) | `enabled`, `target_peak` |

See `src/telephony_aug/config.yaml` for the full default config and inline comments.

## Runtime RIR override

When your training pipeline already has a RIR dataset, inject IRs per call instead of pre-loading a manifest:

```python
# Override the internal RIRSampler for this call only
out = aug(audio, sr, rir=my_rir_1d_ndarray)

# rir is validated: must be 1D (or squeezeable to 1D), non-empty.
# dtype is auto-cast to float32; non-contiguous arrays are made contiguous.
# Sample rate matching is caller responsibility (defensive check via rir_sr).

# Force prob=1.0 pattern: when rir is provided, the rir stage fires
# regardless of cfg["rir"]["prob"], preserving the gate-RNG stream
# consumption pattern (other stages' RNG state stays unchanged).
```

## Deferral / out of scope

- **Multi-speaker mixing** — handled by your upstream Dataset layer
- **GPU differentiable augmentation** — not in this package (see torch-audiomentations / audiomentations for that)
- **EVS codec** — license-restricted; reference C source only
- **Handset HATS-style measured IR** — `room_type='handset'` returns identity (dirac); replace with your own IR pool when available
- **Jitter buffer reordering** — current packet-loss path is loss-only, no reorder

## Compose / chaining caveats

If chaining with another general-purpose augmentation library:
1. Sample rates of both modules **must match** (this module's `sample_rate` config + caller's audio sr)
2. Same-name probabilistic stages **multiply** (both at `prob=0.5` → effective `0.25`)
3. If both have a `codec` stage with different semantics, disable one (`codec.prob: 0.0`) to avoid double roundtrip

## Reproducibility note

The `prob=1.0` force pattern (used when `rir` kwarg is provided) preserves **gate-level RNG** consumption (one `random.random()` per stage in the loop). Stage-internal RNG (e.g., `wet_ratio` uniform inside `_apply_rir`) is **not** preserved when override stage activation differs from baseline. For strict bit-exact reproducibility across runs, either avoid runtime overrides or match stage activation patterns.

## License

MIT — see [LICENSE](LICENSE).
