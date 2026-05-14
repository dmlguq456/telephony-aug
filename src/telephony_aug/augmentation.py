"""
통화 환경 오디오 왜곡 시뮬레이션/증강 파이프라인 (telephony_augmentation.py).

## 7단계 통화 신호 체인 (CPU_PIPELINE_ORDER)
  rir → near_end_dsp → codec → packet_loss_plc
  → far_end_dsp → bandwidth_switch → mic_capture_cpu

  Note: 멀티스피커 믹싱은 upstream Dataset 레이어에서 처리한다 (파이프라인에서 제거됨).

## CPU augmentation pipeline
- **CPU (비미분)** : `__call__` 루프에서 순차 적용.
    codec / PLC / DSP 같은 C-라이브러리 기반 처리를 담당.

## 지연 목록 (현재 미구현 — 후속 연구 과제)
- **멀티채널 / 공간음향** : 다채널 RIR 합성, 바이노럴 렌더링.
- **EVS (Enhanced Voice Services)** : 3GPP EVS 코덱 (라이선스 미확보).
- **핸드셋 IR (핸드셋 임펄스 응답)** : 기기별 주파수 특성 시뮬레이션.
- **코덱 내부 PLC** : 인코더 결합 리팩토링 필요 (future work).

## 의존성
  Required (audio_augmentation.py와 동일 기반):
    numpy scipy soundfile pyyaml librosa

  Optional (telephony 전용 — requirements_telephony.txt 참조):
    torchaudio audiomentations
    pyroomacoustics pyrnnoise opuslib g711 g722
    webrtc-noise-gain (기존 audio_augmentation.py 공유)

## 코딩 컨벤션 (audio_augmentation.py와 동일 유지)
  - 정수 랜덤 선택: random.randint(a, b) (양 끝 포함) — np.random.randint 금지
  - 연속 랜덤값: random.uniform(a, b)
  - 배열 레벨 랜덤: np.random.randn 또는 np.random.RandomState
"""

# ---------------------------------------------------------------------------
# Standard library imports
# ---------------------------------------------------------------------------

import csv
import functools  # noqa: F401 (reserved for lru_cache use in sub-modules)
import logging
import math  # noqa: F401 (reserved for DSP sub-module)
import os  # noqa: F401 (reserved for PATH manipulation)
import random
import subprocess  # noqa: F401 (reserved for ffmpeg fallback)
import tempfile  # noqa: F401 (reserved for codec round-trip)
import threading
from datetime import datetime, timezone
from pathlib import Path  # noqa: F401 (reserved for manifest / dir loading)
from typing import Any, Callable, Dict, List, Optional, Set, Union

# ---------------------------------------------------------------------------
# Third-party imports (required)
# ---------------------------------------------------------------------------

import numpy as np
from scipy import signal
import soundfile as sf
import yaml
import librosa

# ---------------------------------------------------------------------------
# Reuse utilities from audio_augmentation.py
# ---------------------------------------------------------------------------
# All seven symbols exist as module-level functions in audio_augmentation.py
# (verified at lines 79, 98, 144, 187, 192, 200, 221).
# No renaming was necessary — names match exactly.

from ._utils import (
    _check_ffmpeg,
    _generate_colored_noise,
    _generate_synthetic_rir,
    _rms,
    _ensure_range,
    _peak_normalize,
    _validate_audio,
    _parse_ffmpeg_encoders,
)

# ---------------------------------------------------------------------------
# Conditional imports with graceful fallback
# Each library gets its own try/except block so one missing dep does not
# shadow another.  Pattern mirrors audio_augmentation.py lines 45-65.
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

try:
    import torchaudio  # noqa: F401
    _HAS_TORCHAUDIO = True
    logger.info("telephony_augmentation: torchaudio available.")
except ImportError:
    _HAS_TORCHAUDIO = False
    logger.debug("telephony_augmentation: torchaudio not installed (optional).")

try:
    import audiomentations  # noqa: F401
    _HAS_AUDIOMENTATIONS = True
    logger.info("telephony_augmentation: audiomentations available.")
except ImportError:
    _HAS_AUDIOMENTATIONS = False
    logger.debug("telephony_augmentation: audiomentations not installed (optional).")

try:
    import pyroomacoustics  # noqa: F401
    _HAS_PYROOMACOUSTICS = True
    logger.info("telephony_augmentation: pyroomacoustics available.")
except ImportError:
    _HAS_PYROOMACOUSTICS = False
    logger.debug("telephony_augmentation: pyroomacoustics not installed (optional).")

try:
    import pyrnnoise  # noqa: F401
    _HAS_PYRNNOISE = True
    logger.info("telephony_augmentation: pyrnnoise available.")
except ImportError:
    _HAS_PYRNNOISE = False
    logger.debug("telephony_augmentation: pyrnnoise not installed (optional).")

try:
    import opuslib  # noqa: F401
    _HAS_OPUSLIB = True
    logger.info("telephony_augmentation: opuslib available.")
except ImportError:
    _HAS_OPUSLIB = False
    logger.debug("telephony_augmentation: opuslib not installed (optional).")

try:
    import g711  # noqa: F401
    _HAS_G711 = True
    logger.info("telephony_augmentation: g711 available.")
except ImportError:
    _HAS_G711 = False
    logger.debug("telephony_augmentation: g711 not installed (optional).")

try:
    import g722  # noqa: F401
    _HAS_G722 = True
    logger.info("telephony_augmentation: g722 available.")
except ImportError:
    _HAS_G722 = False
    logger.debug("telephony_augmentation: g722 not installed (optional).")

try:
    from webrtc_noise_gain import AudioProcessor as _WebRTCAudioProcessor
    _HAS_WEBRTC = True
    logger.info("telephony_augmentation: webrtc_noise_gain available.")
except ImportError:
    _WebRTCAudioProcessor = None
    _HAS_WEBRTC = False
    logger.debug("telephony_augmentation: webrtc_noise_gain not installed (optional).")

# ---------------------------------------------------------------------------
# Module logger is declared above (before conditional imports so import-time
# log messages are captured).  Re-exported here for clarity.
# ---------------------------------------------------------------------------
# logger = logging.getLogger(__name__)  # already set above

# ---------------------------------------------------------------------------
# Module-level aliases for the inline resample idiom (avoids per-call import
# lookup in DataLoader hot loops).
# Guarded by _HAS_TORCHAUDIO at every call site; None branches are defensive.
# ---------------------------------------------------------------------------
if _HAS_TORCHAUDIO:
    import torch as _torch
    import torchaudio.functional as _taF
else:
    _torch = None
    _taF = None

from math import gcd as _gcd  # used by the scipy fallback path



# ===========================================================================
# === Sub-module 1: RIR (Step 2.1) ===
# ===========================================================================


class RIRSampler:
    """Room Impulse Response 샘플러.

    manifest (JSON/CSV) → rir_dir → pyroomacoustics ShoeBox → 합성 RIR
    우선 순위로 RIR을 샘플링한다.

    Args:
        manifest_path: RIR 메타데이터 JSON 또는 CSV 파일 경로.
                       JSON: ``[{"rir_path":..., "rt60":..., "room_size":..., "room_type":...}]``
                       CSV: 헤더 ``rir_path,rt60,room_size,room_type``
        rir_dir: flat 디렉토리 경로 (manifest 없을 때 fallback).
        room_type_split: room_type별 샘플링 비율
                         (예: ``{'meeting': 0.7, 'handset': 0.3}``).
    """

    def __init__(self, manifest_path=None, rir_dir=None, room_type_split=None):
        self.manifest_path = manifest_path
        self.rir_dir = rir_dir
        self.room_type_split = room_type_split
        self._manifest_entries: list = []
        self._handset_warning_emitted: bool = False

        # Manifest 로드
        if manifest_path is not None:
            self._load_manifest(manifest_path)

        # rir_dir glob (manifest 없을 때 fallback용)
        self._rir_dir_files: list = []
        if rir_dir is not None:
            p = Path(rir_dir)
            wav_files = [str(f) for f in p.rglob("*.wav")]
            flac_files = [str(f) for f in p.rglob("*.flac")]
            self._rir_dir_files = sorted(wav_files + flac_files)
            if self._rir_dir_files:
                logger.info(
                    "RIRSampler: rir_dir에서 %d개의 RIR 파일을 로드했습니다 (%s).",
                    len(self._rir_dir_files),
                    rir_dir,
                )
            else:
                logger.warning(
                    "RIRSampler: rir_dir=%s에서 .wav/.flac 파일을 찾지 못했습니다.",
                    rir_dir,
                )

    def _load_manifest(self, manifest_path: str) -> None:
        """JSON 또는 CSV manifest 파일을 파싱하여 `_manifest_entries`를 채운다.

        파싱 오류가 있는 항목은 경고 후 건너뛴다 (예외 발생 없음).
        """
        path = Path(manifest_path)
        if not path.exists():
            logger.warning("RIRSampler: manifest_path=%s 파일이 존재하지 않습니다.", manifest_path)
            return

        suffix = path.suffix.lower()
        try:
            if suffix == ".json":
                import json
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if not isinstance(raw, list):
                    logger.warning(
                        "RIRSampler: JSON manifest가 리스트 형태가 아닙니다 (%s). 건너뜁니다.",
                        manifest_path,
                    )
                    return
                for i, entry in enumerate(raw):
                    if not isinstance(entry, dict) or "rir_path" not in entry:
                        logger.warning(
                            "RIRSampler: JSON manifest 항목 #%d에 'rir_path' 키가 없습니다. 건너뜁니다.",
                            i,
                        )
                        continue
                    self._manifest_entries.append(entry)

            elif suffix == ".csv":
                import csv
                with open(path, "r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    if reader.fieldnames is None or "rir_path" not in reader.fieldnames:
                        logger.warning(
                            "RIRSampler: CSV manifest에 'rir_path' 컬럼이 없습니다 (%s). 건너뜁니다.",
                            manifest_path,
                        )
                        return
                    for i, row in enumerate(reader):
                        if not row.get("rir_path"):
                            logger.warning(
                                "RIRSampler: CSV manifest 행 #%d의 'rir_path' 값이 비어 있습니다. 건너뜁니다.",
                                i,
                            )
                            continue
                        self._manifest_entries.append(dict(row))
            else:
                logger.warning(
                    "RIRSampler: manifest_path=%s의 확장자(%s)를 지원하지 않습니다 (JSON/CSV만 허용).",
                    manifest_path,
                    suffix,
                )
                return

            logger.info(
                "RIRSampler: manifest에서 %d개의 RIR 항목을 로드했습니다 (%s).",
                len(self._manifest_entries),
                manifest_path,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "RIRSampler: manifest_path=%s 로드 중 오류 발생: %s. 건너뜁니다.",
                manifest_path,
                exc,
            )

    def has_manifest(self) -> bool:
        """manifest가 로드되어 적어도 1개 이상의 항목이 있으면 True."""
        return len(self._manifest_entries) > 0

    def sample_room_type(self) -> str:
        """room_type_split의 가중치 분포에 따라 room_type을 샘플링한다.

        room_type_split이 None이거나 비어 있으면 'meeting'을 반환한다.

        Returns:
            샘플링된 room_type 문자열.
        """
        if not self.room_type_split:
            return "meeting"
        types = list(self.room_type_split.keys())
        weights = list(self.room_type_split.values())
        return random.choices(types, weights=weights)[0]

    def manifest_paths(self) -> "list[str]":
        """manifest에서 추출한 rir_path 절대경로 목록을 반환한다."""
        result = []
        for entry in self._manifest_entries:
            rir_path = entry.get("rir_path", "")
            if rir_path:
                result.append(str(Path(rir_path).resolve()))
        return result

    def sample(self, sr: int, room_type=None) -> np.ndarray:
        """RIR을 샘플링하여 반환한다.

        우선 순위: manifest → rir_dir → pyroomacoustics ShoeBox → 합성 RIR.

        ``room_type='handset'``이면 dirac IR ``np.array([1.0], dtype=np.float32)``를
        즉시 반환한다 (컨볼루션이 identity가 되어 오디오 무변경).

        Args:
            sr: 대상 샘플링 레이트.
            room_type: 요청하는 방 유형 (예: 'meeting', 'office', 'handset').

        Returns:
            float32 RIR 배열.
        """
        # --- handset SHORT-CIRCUIT ---
        if room_type == "handset":
            if not self._handset_warning_emitted:
                logger.warning(
                    "RIRSampler: handset pool deferred, returning dirac IR (identity)"
                )
                self._handset_warning_emitted = True
            return np.array([1.0], dtype=np.float32)

        # --- Priority 1: manifest ---
        if self.has_manifest():
            # room_type 필터 적용 (필터링 결과가 비면 전체 항목 사용)
            if room_type is not None:
                filtered = [e for e in self._manifest_entries if e.get("room_type") == room_type]
                pool = filtered if filtered else self._manifest_entries
            else:
                pool = self._manifest_entries

            entry = random.choice(pool)
            rir_path = entry.get("rir_path", "")
            if rir_path and Path(rir_path).exists():
                try:
                    rir_raw, rir_sr = sf.read(rir_path, dtype="float32", always_2d=False)
                    if rir_raw.ndim == 2:
                        rir_raw = rir_raw[:, 0]
                    if rir_sr != sr:
                        if _HAS_TORCHAUDIO:
                            _t = _torch.from_numpy(np.asarray(rir_raw, dtype=np.float32))
                            rir_raw = _taF.resample(_t, int(rir_sr), int(sr)).numpy()
                        else:
                            _g = _gcd(int(rir_sr), int(sr))
                            rir_raw = signal.resample_poly(rir_raw, int(sr) // _g, int(rir_sr) // _g).astype(np.float32)
                    return rir_raw.astype(np.float32)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "RIRSampler: manifest RIR 로드 실패 (%s): %s. 다음 fallback으로 이동합니다.",
                        rir_path,
                        exc,
                    )

        # --- Priority 2: rir_dir ---
        if self._rir_dir_files:
            rir_path = random.choice(self._rir_dir_files)
            try:
                rir_raw, rir_sr = sf.read(rir_path, dtype="float32", always_2d=False)
                if rir_raw.ndim == 2:
                    rir_raw = rir_raw[:, 0]
                if rir_sr != sr:
                    if _HAS_TORCHAUDIO:
                        _t = _torch.from_numpy(np.asarray(rir_raw, dtype=np.float32))
                        rir_raw = _taF.resample(_t, int(rir_sr), int(sr)).numpy()
                    else:
                        _g = _gcd(int(rir_sr), int(sr))
                        rir_raw = signal.resample_poly(rir_raw, int(sr) // _g, int(rir_sr) // _g).astype(np.float32)
                return rir_raw.astype(np.float32)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "RIRSampler: rir_dir RIR 로드 실패 (%s): %s. 다음 fallback으로 이동합니다.",
                    rir_path,
                    exc,
                )

        # --- Priority 3: pyroomacoustics ShoeBox ---
        if _HAS_PYROOMACOUSTICS:
            try:
                import pyroomacoustics as pra

                room_x = random.uniform(3.0, 7.0)
                room_y = random.uniform(3.0, 7.0)
                room_z = random.uniform(2.5, 3.0)
                room_dims = [room_x, room_y, room_z]

                # rt60_range는 RIRSampler 수준에서 config에 접근할 수 없으므로
                # TelephonyAugmentation._apply_rir에서 rt60을 sample하여
                # 여기서는 합리적인 기본 범위 [0.2, 0.8] 사용
                rt60 = random.uniform(0.2, 0.8)

                e_absorption, max_order = pra.inverse_sabine(rt60, room_dims)
                materials = pra.Material(e_absorption)
                room = pra.ShoeBox(
                    room_dims,
                    fs=sr,
                    materials=materials,
                    max_order=max_order,
                )

                # 무작위 source / receiver 위치 (벽에서 0.5m 이상 떨어지도록)
                margin = 0.5
                src_pos = [
                    random.uniform(margin, room_x - margin),
                    random.uniform(margin, room_y - margin),
                    random.uniform(margin, room_z - margin),
                ]
                mic_pos = [
                    random.uniform(margin, room_x - margin),
                    random.uniform(margin, room_y - margin),
                    random.uniform(margin, room_z - margin),
                ]

                room.add_source(src_pos)
                mic_array = np.array(mic_pos).reshape(3, 1)
                room.add_microphone(mic_array)
                room.simulate()

                rir_pra = room.rir[0][0].astype(np.float32)
                if len(rir_pra) == 0:
                    raise ValueError("pyroomacoustics returned empty RIR")
                peak = np.max(np.abs(rir_pra))
                if peak > 1e-8:
                    rir_pra = rir_pra / peak
                return rir_pra

            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "RIRSampler: pyroomacoustics ShoeBox 생성 실패: %s. 합성 RIR로 fallback합니다.",
                    exc,
                )

        # --- Priority 4: synthetic fallback ---
        rt60 = random.uniform(0.2, 0.8)
        rir_synth = _generate_synthetic_rir(sr, rt60)
        return rir_synth.astype(np.float32)


# ===========================================================================
# === Sub-module 2: Codec (Step 3.1 + 3.2) ===
# ===========================================================================


class CodecSampler:
    """계층형 코덱 샘플러.

    codec_tree 카테고리(PSTN / Cellular / VoIP) 중 하나를
    category_probs 가중치로 샘플링하고, 해당 카테고리 내 codec entry를
    균등 샘플링한다.  비트레이트는 entry 딕셔너리의 키 유무에 따라
    3-way branch로 결정된다.

    Args:
        codec_tree: 카테고리 이름 → codec entry list 매핑.
                    각 entry는 최소한 'format', 'encoder', 'sr' 키를 포함해야 함.
        category_probs: 카테고리 이름 → 샘플링 가중치(float) 매핑.
                        keys와 codec_tree keys가 일치해야 한다.
    """

    def __init__(self, codec_tree: dict, category_probs: dict) -> None:
        self.codec_tree = codec_tree
        self.category_probs = category_probs

        # Validate: all categories in category_probs must be present in codec_tree
        missing = [cat for cat in category_probs if cat not in codec_tree]
        if missing:
            raise ValueError(
                f"CodecSampler: category_probs에 codec_tree에 없는 카테고리가 있습니다: {missing}"
            )

        # Pre-build ordered lists for random.choices (order must be stable)
        self._category_list = list(category_probs.keys())
        self._category_weights = [category_probs[c] for c in self._category_list]

    def sample_config(self) -> dict:
        """코덱 설정 dict를 샘플링하여 반환한다.

        Returns:
            {'format': str, 'encoder': str, 'sr': int, 'bitrate_kbps': float | None}
        """
        # 1. Sample category by weight
        category = random.choices(self._category_list, weights=self._category_weights, k=1)[0]

        # 2. Sample one codec entry uniformly
        entry = random.choice(self.codec_tree[category])

        # 3. Resolve bitrate (3-way branch)
        if "bitrate_modes_kbps" in entry:
            # Discrete mode list — sample one
            bitrate_kbps = float(random.choice(entry["bitrate_modes_kbps"]))
        elif "bitrate_range_kbps" in entry:
            # Continuous range — uniform sample
            lo, hi = entry["bitrate_range_kbps"]
            bitrate_kbps = random.uniform(lo, hi)
        elif "bitrate_kbps" in entry:
            # Fixed scalar
            bitrate_kbps = float(entry["bitrate_kbps"])
        else:
            # No bitrate key (e.g. G.711 PCM — constant bit rate by definition)
            bitrate_kbps = None

        return {
            "format": entry["format"],
            "encoder": entry["encoder"],
            "sr": int(entry["sr"]),
            "bitrate_kbps": bitrate_kbps,
        }

    @classmethod
    def default(cls) -> "CodecSampler":
        """플랜 사양에 따른 기본 코덱 트리로 CodecSampler를 생성한다.

        카테고리 / 인코더 / 비트레이트는 telephony_config.yaml 명세 그대로:
        - PSTN   : pcm_mulaw + pcm_alaw @ 8 kHz (비트레이트 없음)
        - Cellular: AMR-NB @ 8 kHz (8 모드), AMR-WB @ 16 kHz (9 모드)
        - VoIP   : libopus @ 48 kHz [12-64] kbps, G.722 @ 16 kHz [48-64] kbps
        """
        codec_tree = {
            "PSTN": [
                {"format": "wav", "encoder": "pcm_mulaw", "sr": 8000},
                {"format": "wav", "encoder": "pcm_alaw",  "sr": 8000},
            ],
            "Cellular": [
                {
                    "format": "amr",
                    "encoder": "libopencore_amrnb",
                    "sr": 8000,
                    "bitrate_modes_kbps": [4.75, 5.15, 5.90, 6.70, 7.40, 7.95, 10.2, 12.2],
                },
                {
                    "format": "amr",
                    "encoder": "libvo_amrwbenc",
                    "sr": 16000,
                    "bitrate_modes_kbps": [6.60, 8.85, 12.65, 14.25, 15.85, 18.25, 19.85, 23.05, 23.85],
                },
            ],
            "VoIP": [
                {
                    "format": "opus",
                    "encoder": "libopus",
                    "sr": 48000,
                    "bitrate_range_kbps": [12, 64],
                    # fec_prob=0.3 is a placeholder field — not consumed by codec_roundtrip
                    "fec_prob": 0.3,
                },
                {
                    "format": "g722",
                    "encoder": "g722",
                    "sr": 16000,
                    "bitrate_range_kbps": [48, 64],
                },
            ],
        }
        category_probs = {
            "PSTN": 0.2,
            "Cellular": 0.3,
            "VoIP": 0.5,
        }
        return cls(codec_tree=codec_tree, category_probs=category_probs)


def codec_roundtrip(
    audio: np.ndarray, sr: int, codec_config: dict
) -> "tuple[np.ndarray, bool]":
    """코덱 인코딩 → 디코딩 라운드트립을 수행하여 왜곡된 오디오를 반환한다.

    백엔드 우선 순위:
      1. torchaudio.io.AudioEffector  (``_HAS_TORCHAUDIO`` 플래그가 True인 경우)
      2. ffmpeg subprocess            (``_check_ffmpeg()``가 True인 경우)
      3. 직접 바인딩 폴백             (g711 / opuslib / g722 패키지)

    모든 백엔드 실패 시 원본 ``audio``를 그대로 반환하되 ``applied=False``를
    반환한다 (예외를 발생시키지 않음 — graceful degradation 계약 유지).

    Args:
        audio: float32 mono 입력 오디오.
        sr: 입력 샘플링 레이트.
        codec_config: CodecSampler.sample_config() 반환값.
                      필수 키: 'format', 'encoder', 'sr'.
                      선택 키: 'bitrate_kbps' (None이면 비트레이트 미지정).

    Returns:
        (audio_out, applied) 튜플.
        audio_out: 왜곡이 적용된 float32 mono 배열 (원본과 길이 동일).
        applied: True이면 적어도 하나의 백엔드가 성공했음을 의미.
                 False이면 모든 백엔드 실패 → 원본 audio 반환.
    """
    target_sr: int = codec_config["sr"]
    bitrate_kbps = codec_config.get("bitrate_kbps")  # may be None
    fmt: str = codec_config["format"]
    encoder: str = codec_config["encoder"]
    original_len: int = len(audio)

    # ------------------------------------------------------------------
    # Helper: resample to target_sr then back, and length-match
    # ------------------------------------------------------------------
    def _resample_to_target(x: np.ndarray) -> np.ndarray:
        if int(sr) == int(target_sr):
            out = x.astype(np.float32, copy=False)
        elif _HAS_TORCHAUDIO:
            _t = _torch.from_numpy(np.asarray(x, dtype=np.float32))
            out = _taF.resample(_t, int(sr), int(target_sr)).numpy()
        else:
            _g = _gcd(int(sr), int(target_sr))
            out = signal.resample_poly(x, int(target_sr) // _g, int(sr) // _g).astype(np.float32)
        return out

    def _resample_from_target(x: np.ndarray) -> np.ndarray:
        if int(target_sr) == int(sr):
            out = x.astype(np.float32, copy=False)
        elif _HAS_TORCHAUDIO:
            _t = _torch.from_numpy(np.asarray(x, dtype=np.float32))
            out = _taF.resample(_t, int(target_sr), int(sr)).numpy()
        else:
            _g = _gcd(int(target_sr), int(sr))
            out = signal.resample_poly(x, int(sr) // _g, int(target_sr) // _g).astype(np.float32)
        return out

    def _length_match(x: np.ndarray) -> np.ndarray:
        if len(x) > original_len:
            return x[:original_len]
        if len(x) < original_len:
            return np.pad(x, (0, original_len - len(x)), mode="constant")
        return x

    # ==================================================================
    # Backend 1: torchaudio AudioEffector
    # ==================================================================
    if _HAS_TORCHAUDIO:
        try:
            import torch
            from torchaudio.io import AudioEffector, CodecConfig

            in_audio = _resample_to_target(audio)
            # AudioEffector expects (T, C) — channel last
            in_tensor = torch.from_numpy(in_audio.astype(np.float32)).unsqueeze(-1)  # (T, 1)

            codec_cfg = (
                CodecConfig(bit_rate=int(bitrate_kbps * 1000))
                if bitrate_kbps is not None
                else None
            )
            effector = AudioEffector(
                effect=None,
                format=fmt,
                encoder=encoder,
                codec_config=codec_cfg,
            )
            out_tensor = effector.apply(in_tensor, sample_rate=target_sr)
            out_audio = out_tensor.squeeze(-1).numpy().astype(np.float32)

            out_audio = _resample_from_target(out_audio)
            out_audio = _length_match(out_audio)
            return out_audio.astype(np.float32), True

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "codec_roundtrip (AudioEffector): %s/%s 실패: %s. ffmpeg로 fallback합니다.",
                fmt, encoder, exc,
            )

    # ==================================================================
    # Backend 2: ffmpeg subprocess
    # ==================================================================
    if _check_ffmpeg():
        try:
            # Verify encoder availability (skip if AMR, G.711, G.722 — built-in)
            _builtin_encoders = {"pcm_mulaw", "pcm_alaw", "g722"}
            if encoder not in _builtin_encoders:
                available = _parse_ffmpeg_encoders()
                if encoder not in available:
                    raise RuntimeError(
                        f"ffmpeg encoder '{encoder}' not available (not in ffmpeg -encoders)"
                    )

            in_audio = _resample_to_target(audio)
            in_bytes = in_audio.astype(np.float32).tobytes()

            # Build encode argv
            bitrate_flag: list = ["-b:a", f"{bitrate_kbps}k"] if bitrate_kbps is not None else []

            if encoder in ("pcm_mulaw", "pcm_alaw"):
                enc_cmd = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-f", "f32le", "-ar", str(target_sr), "-ac", "1", "-i", "pipe:0",
                    "-ar", str(target_sr), "-ac", "1",
                    "-codec:a", encoder,
                    "-f", fmt, "pipe:1",
                ]
            elif encoder == "libopencore_amrnb":
                enc_cmd = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-f", "f32le", "-ar", "8000", "-ac", "1", "-i", "pipe:0",
                    "-ar", "8000", "-ac", "1",
                    "-codec:a", "libopencore_amrnb",
                    *bitrate_flag,
                    "-f", "amr", "pipe:1",
                ]
            elif encoder == "libvo_amrwbenc":
                enc_cmd = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-f", "f32le", "-ar", "16000", "-ac", "1", "-i", "pipe:0",
                    "-ar", "16000", "-ac", "1",
                    "-codec:a", "libvo_amrwbenc",
                    *bitrate_flag,
                    "-f", "amr", "pipe:1",
                ]
            elif encoder == "libopus":
                enc_cmd = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-f", "f32le", "-ar", str(target_sr), "-ac", "1", "-i", "pipe:0",
                    "-ar", "48000", "-ac", "1",
                    "-codec:a", "libopus",
                    *bitrate_flag,
                    "-f", "opus", "pipe:1",
                ]
            elif encoder == "g722":
                enc_cmd = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-f", "f32le", "-ar", "16000", "-ac", "1", "-i", "pipe:0",
                    "-ar", "16000", "-ac", "1",
                    "-codec:a", "g722",
                    "-f", "g722", "pipe:1",
                ]
            else:
                # Generic fallback (may not work for all encoders)
                enc_cmd = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-f", "f32le", "-ar", str(target_sr), "-ac", "1", "-i", "pipe:0",
                    "-ar", str(target_sr), "-ac", "1",
                    "-codec:a", encoder,
                    *bitrate_flag,
                    "-f", fmt, "pipe:1",
                ]

            enc_result = subprocess.run(
                enc_cmd, input=in_bytes, capture_output=True, timeout=30
            )
            if enc_result.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg encode failed (rc={enc_result.returncode}): "
                    f"{enc_result.stderr.decode(errors='replace')[:300]}"
                )
            encoded_bytes = enc_result.stdout

            # Decode back to f32le via second ffmpeg invocation
            dec_cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", "pipe:0",
                "-ar", str(target_sr), "-ac", "1",
                "-f", "f32le", "pipe:1",
            ]
            dec_result = subprocess.run(
                dec_cmd, input=encoded_bytes, capture_output=True, timeout=30
            )
            if dec_result.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg decode failed (rc={dec_result.returncode}): "
                    f"{dec_result.stderr.decode(errors='replace')[:300]}"
                )

            out_audio = np.frombuffer(dec_result.stdout, dtype=np.float32).copy()
            out_audio = _resample_from_target(out_audio)
            out_audio = _length_match(out_audio)
            return out_audio.astype(np.float32), True

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "codec_roundtrip (ffmpeg): %s/%s 실패: %s. 직접 바인딩으로 fallback합니다.",
                fmt, encoder, exc,
            )

    # ==================================================================
    # Backend 3: Direct Python bindings (final fallback)
    # ==================================================================
    try:
        in_audio = _resample_to_target(audio)

        if encoder in ("pcm_mulaw", "pcm_alaw") and _HAS_G711:
            import g711 as _g711
            # g711 library: encode float32 array → bytes, decode bytes → float32 array
            if encoder == "pcm_mulaw":
                encoded = _g711.encode_ulaw(in_audio)
                out_audio = _g711.decode_ulaw(encoded).astype(np.float32)
            else:
                encoded = _g711.encode_alaw(in_audio)
                out_audio = _g711.decode_alaw(encoded).astype(np.float32)
            out_audio = _resample_from_target(out_audio)
            out_audio = _length_match(out_audio)
            return out_audio.astype(np.float32), True

        elif encoder == "libopus" and _HAS_OPUSLIB:
            import opuslib
            FRAME_SIZE = 960  # 20 ms @ 48 kHz
            bitrate_bps = int(bitrate_kbps * 1000) if bitrate_kbps is not None else 32000
            enc = opuslib.Encoder(target_sr, 1, opuslib.APPLICATION_VOIP)
            enc.bitrate = bitrate_bps
            dec = opuslib.Decoder(target_sr, 1)
            pcm_int16 = (in_audio * 32767.0).clip(-32768, 32767).astype(np.int16)
            frames_out = []
            for start in range(0, len(pcm_int16), FRAME_SIZE):
                frame = pcm_int16[start: start + FRAME_SIZE]
                if len(frame) < FRAME_SIZE:
                    frame = np.pad(frame, (0, FRAME_SIZE - len(frame)))
                encoded_frame = enc.encode(frame.tobytes(), FRAME_SIZE)
                decoded_bytes = dec.decode(encoded_frame, FRAME_SIZE)
                decoded_frame = np.frombuffer(decoded_bytes, dtype=np.int16).astype(np.float32) / 32767.0
                frames_out.append(decoded_frame)
            if frames_out:
                out_audio = np.concatenate(frames_out)
            else:
                out_audio = in_audio.copy()
            out_audio = _resample_from_target(out_audio)
            out_audio = _length_match(out_audio)
            return out_audio.astype(np.float32), True

        elif encoder == "g722" and _HAS_G722:
            import g722 as _g722
            codec = _g722.G722()
            encoded = codec.encode(in_audio)
            out_audio = codec.decode(encoded).astype(np.float32)
            out_audio = _resample_from_target(out_audio)
            out_audio = _length_match(out_audio)
            return out_audio.astype(np.float32), True

        elif encoder in ("libopencore_amrnb", "libvo_amrwbenc"):
            # AMR direct binding: no clean Python ctypes wrapper available.
            # Fall through to no-op (warning emitted below).
            raise RuntimeError(
                f"AMR direct binding not supported for encoder='{encoder}'. "
                "Install ffmpeg with AMR support for AMR codec roundtrip."
            )

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "codec_roundtrip (direct binding): %s/%s 실패: %s. 원본 오디오를 반환합니다.",
            fmt, encoder, exc,
        )

    # All backends failed — return original audio unchanged with applied=False
    logger.warning(
        "codec_roundtrip: 모든 백엔드 실패 (format=%s, encoder=%s). 원본 오디오를 반환합니다.",
        fmt, encoder,
    )
    return audio.astype(np.float32), False


# ===========================================================================
# === Sub-module 3: PLC (Step 4.1) ===
# ===========================================================================


def gilbert_elliott_mask(
    num_frames: int,
    p_good_to_bad: float = 0.05,
    p_bad_to_good: float = 0.5,
    rng=None,
) -> np.ndarray:
    """2-state Gilbert-Elliott Markov chain으로 패킷 손실 마스크를 생성한다.

    State 'good' → True (received), 'bad' → False (lost).
    Steady-state loss rate ≈ p_good_to_bad / (p_good_to_bad + p_bad_to_good).

    Args:
        num_frames: 마스크 길이 (프레임 수).
        p_good_to_bad: good → bad 전이 확률 (기본값 0.05).
        p_bad_to_good: bad → good 전이 확률 (기본값 0.5).
        rng: 랜덤 소스.
             - None (기본값, 프로덕션 경로): 전역 ``random`` 모듈 사용.
               ``worker_init_fn``이 worker당 1회 시드를 설정하므로 결정적.
             - ``random.Random`` 인스턴스 (단위 테스트 주입용):
               ``rng.random()``을 사용하여 전역 RNG 상태를 오염시키지 않음.

    Returns:
        shape ``(num_frames,)``, dtype bool의 ndarray.
        True = 수신됨, False = 손실됨.
    """
    if num_frames <= 0:
        return np.empty(0, dtype=bool)

    rand_fn = rng.random if rng is not None else random.random

    mask = np.empty(num_frames, dtype=bool)
    # Start in 'good' state with probability equal to steady-state
    state_good = True
    for i in range(num_frames):
        mask[i] = state_good
        r = rand_fn()
        if state_good:
            if r < p_good_to_bad:
                state_good = False
        else:
            if r < p_bad_to_good:
                state_good = True
    return mask


def g711_app_i_plc_fill(
    audio: np.ndarray,
    frame_size: int,
    loss_mask: np.ndarray,
    sr: int,
) -> np.ndarray:
    """G.711 Appendix I 스타일 PLC(Packet Loss Concealment) fill.

    각 손실 프레임에 대해:
      1. 이전 히스토리가 없으면 (첫 프레임 손실): 무음(0) 채움.
      2. 이전 히스토리가 있으면: 이전 30 ms에서 자기상관(autocorrelation)으로
         피치 주기를 추정(pitch range 50-500 Hz → 2-20 ms 클램프)하고,
         해당 pitch period를 반복 복제하여 프레임을 채운다.
         프레임 경계에서 2 ms overlap-add(OLA)를 적용하여 클릭 아티팩트를 방지.
      3. 연속 버스트 손실이 60 ms 초과 시: simplified G.191 STL plc.c burst
         attenuation — 10 ms 이후 매 프레임마다 지수적 감쇠(~0.8 배)를 적용.
         (**Ref: "simplified G.191 plc.c burst attenuation"**)

    Args:
        audio: float32 mono 입력 오디오. 이 배열은 변경되지 않음(copy 반환).
        frame_size: 프레임 크기 (샘플 수).
        loss_mask: shape ``(num_frames,)``, bool. True = 수신, False = 손실.
        sr: 샘플링 레이트.

    Returns:
        PLC fill이 적용된 float32 배열. 길이는 입력과 동일.
    """
    if len(audio) == 0 or len(loss_mask) == 0:
        return audio.astype(np.float32)

    out = audio.copy().astype(np.float32)

    # Pitch search parameters
    pitch_lo_hz = 50.0
    pitch_hi_hz = 500.0
    period_min = int(sr / pitch_hi_hz)  # ~2 ms at 8 kHz → 16 samples
    period_max = int(sr / pitch_lo_hz)  # ~20 ms at 8 kHz → 160 samples
    period_min = max(period_min, 1)

    history_len = int(0.030 * sr)  # 30 ms history window for pitch detection
    ola_len = int(0.002 * sr)      # 2 ms overlap-add window
    ola_len = max(ola_len, 1)

    # Burst attenuation thresholds (simplified G.191 plc.c)
    burst_onset_ms = 10.0
    burst_onset_frames = max(1, int(burst_onset_ms * sr / 1000 / frame_size))
    burst_attenuation = 0.8  # per-frame multiplier after onset

    num_frames = len(loss_mask)
    burst_count = 0  # consecutive lost frames counter

    for i in range(num_frames):
        start = i * frame_size
        end = min(start + frame_size, len(out))
        seg_len = end - start

        if loss_mask[i]:
            # Frame received — reset burst counter
            burst_count = 0
            continue

        # --- Lost frame ---
        burst_count += 1

        # Determine attenuation for burst losses (simplified G.191 plc.c)
        if burst_count > burst_onset_frames:
            extra_frames = burst_count - burst_onset_frames
            attenuation = burst_attenuation ** extra_frames
        else:
            attenuation = 1.0

        # Case 1: No prior history → zero-fill
        hist_end = start
        if hist_end == 0:
            out[start:end] = 0.0
            continue

        # Case 2: Pitch-period repetition from prior 30 ms
        hist_start = max(0, hist_end - history_len)
        history = out[hist_start:hist_end]

        # Estimate pitch via autocorrelation
        if len(history) >= period_min + period_max:
            corr = signal.correlate(history, history, mode="full")
            corr = corr[len(corr) // 2:]  # keep lags >= 0
            # Search in [period_min, period_max]
            search_end = min(period_max + 1, len(corr))
            if search_end > period_min:
                lag_idx = np.argmax(corr[period_min:search_end]) + period_min
                pitch_period = lag_idx
            else:
                pitch_period = period_min
        else:
            pitch_period = max(period_min, len(history) // 2 if len(history) > 0 else period_min)

        pitch_period = max(period_min, min(pitch_period, period_max))

        # Extract one pitch period from end of history
        pitch_template = history[-pitch_period:] if pitch_period <= len(history) else history

        # Replicate pitch_template to fill seg_len samples with OLA at joins
        filled = np.zeros(seg_len, dtype=np.float32)
        pos = 0
        while pos < seg_len:
            chunk = pitch_template[: seg_len - pos]
            # OLA fade-in at the join boundary
            if pos == 0 and pos + len(chunk) <= seg_len:
                fade_n = min(ola_len, len(chunk))
                fade_in = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
                fade_chunk = chunk.copy()
                fade_chunk[:fade_n] *= fade_in
                filled[pos: pos + len(chunk)] += fade_chunk
            else:
                filled[pos: pos + len(chunk)] += chunk
            pos += len(chunk)

        # Apply burst attenuation
        filled *= attenuation

        out[start:end] = filled[:seg_len]

    return out.astype(np.float32)


# ===========================================================================
# === Sub-module 4: DSP (Step 5.1-5.2) ===
# ===========================================================================
# webrtc_apm_process + rnnoise_process + ns_cascade will be implemented
# here in Phase 5 (Steps 5.1-5.2).
# Placeholder — do not add code below this line until Step 5.x.


# ===========================================================================
# === Sub-module 5: Sampler + Logger (Step 6.x) ===
# ===========================================================================


class DistortionLogger:
    """Append-only logger for augmentation stage parameters.

    Thread-safe (lock-guarded) — important for DataLoader num_workers > 1.

    CSV format: each row = (timestamp, stage, param_key1, param_key2, ...).
    Dynamic columns: if a new stage introduces new param keys, new columns
    are appended (DictWriter pattern with fieldnames discovered on first write).

    Optional WandB integration: if wandb_run is passed, every log() call
    also issues ``wandb_run.log({f"aug/{stage}/{k}": v for k, v in params.items()})``.
    """

    def __init__(
        self,
        csv_path: Optional[Union[str, Path]] = None,
        wandb_run=None,
    ) -> None:
        """Initialize DistortionLogger.

        Args:
            csv_path: Path to CSV file.  None → CSV disabled.
            wandb_run: Active wandb Run object (caller-managed lifecycle).  None → WandB disabled.
        """
        self._csv_path: Optional[Path] = Path(csv_path) if csv_path is not None else None
        self._wandb_run = wandb_run
        self._lock: threading.Lock = threading.Lock()
        self._fieldnames: List[str] = []  # ordered list; first two are always timestamp, stage
        self._first_write: bool = True

        # Detect existing file so we can read existing headers
        if self._csv_path is not None and self._csv_path.exists() and self._csv_path.stat().st_size > 0:
            try:
                with open(self._csv_path, "r", newline="", encoding="utf-8") as _f:
                    _reader = csv.DictReader(_f)
                    if _reader.fieldnames:
                        self._fieldnames = list(_reader.fieldnames)
                        self._first_write = False
            except Exception as exc:
                logger.warning(
                    "DistortionLogger: 기존 CSV 헤더 읽기 실패 (%s) — 새 파일로 처리합니다.", exc
                )
                self._fieldnames = []
                self._first_write = True

    # ------------------------------------------------------------------
    def log(self, stage: str, params: dict) -> None:
        """Append one row to CSV and/or log to WandB.

        Args:
            stage: Augmentation stage name (e.g. "codec", "packet_loss_plc").
            params: Dict of parameter key→value pairs for this stage.
        """
        try:
            with self._lock:
                timestamp = datetime.now(timezone.utc).isoformat()
                # Flatten values to str for CSV safety
                flat_params: Dict[str, str] = {str(k): str(v) for k, v in params.items()}

                if self._csv_path is not None:
                    # Build the row dict (timestamp + stage + params)
                    row: Dict[str, str] = {"timestamp": timestamp, "stage": stage}
                    row.update(flat_params)

                    # Discover any new param keys and extend fieldnames
                    for key in row:
                        if key not in self._fieldnames:
                            self._fieldnames.append(key)

                    if self._first_write:
                        # Write header + first row (overwrite-safe: file was empty/new)
                        with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(
                                f,
                                fieldnames=self._fieldnames,
                                extrasaction="ignore",
                            )
                            writer.writeheader()
                            writer.writerow(row)
                        self._first_write = False
                    else:
                        # Append row; if fieldnames changed (new columns), rewrite header
                        with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(
                                f,
                                fieldnames=self._fieldnames,
                                extrasaction="ignore",
                            )
                            writer.writerow(row)

                if self._wandb_run is not None:
                    wandb_payload = {
                        f"aug/{stage}/{k}": v for k, v in params.items()
                    }
                    self._wandb_run.log(wandb_payload)

        except Exception as exc:
            logger.warning(
                "DistortionLogger.log: 로깅 중 예외 발생 (파이프라인은 계속됩니다): %s", exc
            )

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Flush any pending writes.  Idempotent.

        CSV writes use ``open()`` per call (no persistent file handle),
        so nothing to flush; this method exists for lifecycle symmetry.
        """
        # No persistent handle to close; idempotent by design.


# ===========================================================================
# === Main class: TelephonyAugmentation (Step 1.3+) ===
# ===========================================================================


class TelephonyAugmentation:
    """통화 환경 오디오 왜곡 증강기 (CPU augmentation pipeline).

    config dict 또는 YAML 파일 경로를 받아 TelephonyAugmentation 인스턴스를
    생성한다.  인스턴스를 직접 호출(callable)하면 CPU 파이프라인(7단계 통화
    신호 체인)이 순차적으로 적용된다.

    audiomentations Compose 호환:
        Compose([TelephonyAugmentation(cfg)]) 패턴으로 직접 사용 가능.
        ``__call__``이 ``(audio, sr)`` 및 ``(samples=..., sample_rate=...)``
        두 인터페이스를 모두 허용한다.

    Example::

        aug = TelephonyAugmentation("telephony_config.yaml")
        augmented = aug(audio_array, sr=48000)

        # audiomentations Compose 내부 호출 패턴도 동작:
        augmented = aug(samples=audio_array, sample_rate=48000)
    """

    # -----------------------------------------------------------------------
    # Class-level pipeline order constants
    # -----------------------------------------------------------------------

    ALLOWED_CONFIG_KEYS = frozenset({
        # Pipeline stages
        "noise_inject", "rir", "near_end_dsp", "codec", "packet_loss_plc",
        "far_end_dsp", "bandwidth_switch", "mic_capture_cpu",
        # Global
        "sample_rate", "final_normalize", "distortion_logging",
    })

    CPU_PIPELINE_ORDER = [
        "noise_inject",       # Stage 0 — additive noise (SNR sampling) BEFORE other stages
                              #            so downstream NS sees noisy input → over-suppression
                              #            artifact (the perceptually dominant telephony distortion).
        "rir",                # Stage A — room impulse response convolution
        "near_end_dsp",       # Stage B — optional WebRTC APM pre-codec
        "codec",              # Stage C — hierarchical codec sampler
        "packet_loss_plc",    # Stages D+E — Gilbert-Elliott + App I PLC
        "far_end_dsp",        # Stage F — NS cascade (RNNoise / WebRTC)
        "bandwidth_switch",   # Stage G — NB ↔ WB ↔ FB resample
        "mic_capture_cpu",    # Stage I — clipping at CPU side (optional)
    ]
    # Stage J (quality eval) is NOT in pipeline — performed externally.

    # -----------------------------------------------------------------------
    # __init__
    # -----------------------------------------------------------------------

    def __init__(self, config: Union[str, Path, Dict[str, Any]]) -> None:
        """TelephonyAugmentation 초기화.

        Args:
            config: YAML 파일 경로(str 또는 Path) 또는 설정 dict.
                    최소한 ``sample_rate`` 키를 포함하는 것을 권장한다.
                    각 스테이지 키(e.g. ``rir``, ``codec``)가 없으면 해당
                    스테이지는 ``prob=0.0``으로 취급되어 건너뛰어진다.
        """
        # --- Config loading (mirror AudioAugmentor.__init__) ---------------
        if isinstance(config, (str, Path)):
            with open(config, "r", encoding="utf-8") as f:
                self.cfg: Dict[str, Any] = yaml.safe_load(f)
        else:
            self.cfg = dict(config)

        # Validate config keys — warn on unknown (caller may have stale config from
        # pre-slim cycle with multispeaker_mix / GPU stages like gain/noise/etc.)
        unknown_keys = set(self.cfg.keys()) - self.ALLOWED_CONFIG_KEYS
        if unknown_keys:
            logger.warning(
                "TelephonyAugmentation: unknown config keys ignored (likely from "
                "pre-slim cycle): %s. Allowed keys: %s",
                sorted(unknown_keys),
                sorted(self.ALLOWED_CONFIG_KEYS),
            )

        # --- Dependency flags -----------------------------------------------
        self.has_torchaudio: bool = _HAS_TORCHAUDIO
        self.has_audiomentations: bool = _HAS_AUDIOMENTATIONS
        self.has_pyroomacoustics: bool = _HAS_PYROOMACOUSTICS
        self.has_pyrnnoise: bool = _HAS_PYRNNOISE
        self.has_opuslib: bool = _HAS_OPUSLIB
        self.has_g711: bool = _HAS_G711
        self.has_g722: bool = _HAS_G722
        self.has_webrtc: bool = _HAS_WEBRTC
        self.has_ffmpeg: bool = _check_ffmpeg()

        if not self.has_ffmpeg:
            logger.warning(
                "TelephonyAugmentation: ffmpeg를 찾을 수 없습니다. "
                "codec 시뮬레이션 단계가 건너뛰어집니다."
            )

        # --- Sub-module: RIRSampler (Step 2.1) --------------------------------
        rir_cfg = self.cfg.get("rir", {})
        self.rir_sampler = RIRSampler(
            manifest_path=rir_cfg.get("manifest_path"),
            rir_dir=rir_cfg.get("rir_dir"),
            room_type_split=rir_cfg.get("room_type_split"),
        )

        # --- Sub-module: CodecSampler (Step 3.1) ------------------------------
        codec_cfg = self.cfg.get("codec", {})
        _codec_tree = codec_cfg.get("codec_tree")
        _category_probs = codec_cfg.get("category_probs")
        if _codec_tree is not None and _category_probs is not None:
            self.codec_sampler = CodecSampler(
                codec_tree=_codec_tree,
                category_probs=_category_probs,
            )
        else:
            self.codec_sampler = CodecSampler.default()

        # --- Sub-module: DistortionLogger (Step 6.2) -------------------------
        _log_cfg = self.cfg.get("distortion_logging", {})
        if _log_cfg.get("enabled", False) and _log_cfg.get("csv_path"):
            self.distortion_logger: Optional[DistortionLogger] = DistortionLogger(
                csv_path=_log_cfg["csv_path"]
            )
        else:
            self.distortion_logger = None

        # --- audiomentations CPU compose (optional mic_capture stage) ------
        # Attribute name is pinned here; set_random_state references this
        # exact name.  Populated here if audiomentations is available and
        # mic_capture_cpu config has clip or gain probability.
        self._cpu_audiomentations_compose = None
        if _HAS_AUDIOMENTATIONS:
            mic_cpu_cfg = self.cfg.get("mic_capture_cpu", {})
            _compose_transforms = []
            try:
                from audiomentations import Compose as _ACompose, Clip as _AClip, Gain as _AGain
                clip_prob = float(mic_cpu_cfg.get("clip_prob", 0.0))
                if clip_prob > 0:
                    clip_thr = mic_cpu_cfg.get("clip_threshold_range", [0.7, 0.95])
                    _compose_transforms.append(
                        _AClip(
                            a_min=-float(clip_thr[1]),
                            a_max=float(clip_thr[1]),
                            p=clip_prob,
                        )
                    )
                gain_range = mic_cpu_cfg.get("gain_db_range", None)
                gain_prob = float(mic_cpu_cfg.get("gain_prob", 0.0))
                if gain_range is not None and gain_prob > 0:
                    _compose_transforms.append(
                        _AGain(
                            min_gain_in_db=float(gain_range[0]),
                            max_gain_in_db=float(gain_range[1]),
                            p=gain_prob,
                        )
                    )
                if _compose_transforms:
                    self._cpu_audiomentations_compose = _ACompose(_compose_transforms)
            except Exception as _exc:
                logger.debug(
                    "TelephonyAugmentation: audiomentations Compose 초기화 실패: %s — "
                    "manual fallback 사용.",
                    _exc,
                )

        # --- CPU dispatch table --------------------------------------------
        # Validated at init time: AttributeError if any _apply_* is missing.
        self._cpu_dispatch: Dict[str, Callable] = {
            stage: getattr(self, f"_apply_{stage}")
            for stage in self.CPU_PIPELINE_ORDER
        }

    # -----------------------------------------------------------------------
    # from_audio_augmentor_config — migration helper (Step 6.1)
    # -----------------------------------------------------------------------

    @classmethod
    def from_audio_augmentor_config(cls, cfg: dict) -> "TelephonyAugmentation":
        """Migration helper for AudioAugmentor users.

        Maps legacy AudioAugmentor config schema to TelephonyAugmentation config.

        Legacy → Telephony stage mapping:
        | legacy stage (AudioAugmentor) | telephony stage | notes |
        |---|---|---|
        | reverb | rir | RIRSampler covers same intent; cfg keys (rir_dir, wet_ratio_range) carry across |
        | codec | codec | **semantic change** — legacy uniform sampler → telephony hierarchical sampler; legacy `types: [...]` must remap to category_probs + codec_tree |
        | packet_loss | packet_loss_plc | legacy Bernoulli loss_rate → Gilbert-Elliott; p_good_to_bad ≈ loss_rate, p_bad_to_good = 0.5 (geometric-burst baseline) |
        | webrtc_proc | near_end_dsp | (and optionally far_end_dsp) helper defaults to near_end_dsp, leaves far_end_dsp at prob=0.0 |
        | bandwidth | bandwidth_switch | mode enum telephone_nb/telephone_wb/webapp → target_modes [NB, WB, FB] |
        | resample | bandwidth_switch | folded into same target-sr enum dispatch |
        | clipping | mic_capture_cpu | hard-clipping path shared; threshold_range carries over |
        | noise | (deferred — no equivalent in slimmed Telephony CPU pipeline) | intentionally unmapped; may map to near_end_dsp noise injection in a future cycle |
        | (no legacy equiv.) | (no direct mapping) | all CPU stages have no-op default (prob=0.0) |

        Args:
            cfg: AudioAugmentor-format config (dict or YAML path).

        Returns:
            TelephonyAugmentation instance with mapped config.

        Raises:
            NotImplementedError: full implementation deferred to next cycle.
                Use the mapping table above to manually translate config.
        """
        raise NotImplementedError(
            "from_audio_augmentor_config: deferred to next cycle. "
            "See docstring mapping table to translate AudioAugmentor config manually."
        )

    # -----------------------------------------------------------------------
    # __call__  (CPU non-differentiable pipeline)
    # -----------------------------------------------------------------------

    def __call__(
        self,
        audio: Optional[np.ndarray] = None,
        sr: Optional[int] = None,
        *,
        samples: Optional[np.ndarray] = None,
        sample_rate: Optional[int] = None,
        rir: Optional[np.ndarray] = None,
        rir_sr: Optional[int] = None,
    ) -> np.ndarray:
        """CPU 파이프라인을 오디오에 순차 적용한다.

        audiomentations Compose 호환을 위해 두 가지 호출 방식을 모두 허용한다:
          - 위치 인수:  aug(audio, sr)
          - 키워드 인수: aug(samples=audio, sample_rate=sr)   ← Compose 내부 패턴

        ``rir`` 인자를 제공하면 RIRSampler를 우회하고 직접 컨볼루션한다.
        sr 매칭은 caller 책임 (rir_sr ≠ audio_sr 이면 caller가 미리 resample).

        Note:
            audiomentations Compose는 ``samples``/``sample_rate``만 introspect하므로
            ``rir`` 인자는 Compose 외부 직접 호출 시에만 사용 가능.

        Args:
            audio: 입력 오디오 배열 (임의 dtype, mono 권장).
            sr: 샘플링 레이트.
            samples: ``audio``의 audiomentations 키워드 별칭.
            sample_rate: ``sr``의 audiomentations 키워드 별칭.
            rir: 1D ndarray IR; 주어지면 RIRSampler 우회하고 직접 컨볼루션.
                 sr 매칭은 caller 책임. (2D인 경우 1축이 1이면 squeeze 허용.)
            rir_sr: rir IR의 샘플링 레이트 (선택). 제공 시 audio sr와 불일치하면
                    ValueError를 발생시킨다 (방어적 sr-매칭 검사). None (기본값)이면
                    검사를 생략하며 caller가 sr 매칭 책임을 진다.

        Returns:
            왜곡이 적용된 float32 mono 배열, 길이는 입력과 동일.

        NOTE on RNG-stream preservation: the rir override uses a prob=1.0 force
        pattern so the gate-level `random.random()` consumption matches baseline.
        However, stage-internal RNG (wet_ratio sampling inside _apply_rir, codec
        sampling, etc.) is NOT preserved between override and non-override paths
        when stage activation differs. Strict reproducibility requires either
        matching activation (use rir override when baseline would also fire rir)
        or seeding caller-side.
        """
        # audiomentations kwarg alias — Compose passes samples= / sample_rate=
        audio = audio if audio is not None else samples
        sr = sr if sr is not None else sample_rate

        # Pre-validate runtime rir override if provided
        rir_stage_override: Optional[np.ndarray] = None
        if rir is not None:
            rir_arr = np.asarray(rir)
            # Squeeze (1, T) or (T, 1) → (T,) but preserve 1D inputs (np.squeeze
            # on shape (1,) produces a 0-D scalar — atleast_1d guards against that).
            if rir_arr.ndim > 1:
                rir_arr = np.atleast_1d(np.squeeze(rir_arr))
            if rir_arr.ndim != 1:
                raise ValueError(
                    f"rir kwarg must be 1D after squeeze (or squeezeable to 1D); "
                    f"got shape {np.shape(rir)}"
                )
            if rir_arr.size == 0:
                raise ValueError("rir kwarg must be non-empty")
            # NEW: SR mismatch check
            if rir_sr is not None and int(rir_sr) != int(sr or self.cfg.get("sample_rate", 48000)):
                raise ValueError(
                    f"rir_sr={rir_sr} mismatches audio sr={sr or self.cfg.get('sample_rate', 48000)}. "
                    f"Caller must resample IR to audio sr before passing to TelephonyAugmentation."
                )
            rir_arr = np.ascontiguousarray(rir_arr).astype(np.float32, copy=False)
            rir_stage_override = rir_arr

        sr = sr or self.cfg.get("sample_rate", 48000)
        audio, sr = _validate_audio(audio, sr)
        original_len = len(audio)

        # Probability-gated CPU pipeline loop
        for stage in self.CPU_PIPELINE_ORDER:
            cfg_stage = self.cfg.get(stage, {})
            if stage == "rir" and rir_stage_override is not None:
                # Inject runtime IR; force apply (prob=1.0) so RNG stream remains unchanged
                cfg_stage = {**cfg_stage, "rir_override": rir_stage_override, "prob": 1.0}
            if random.random() < cfg_stage.get("prob", 0.0):
                audio = self._cpu_dispatch[stage](audio, sr, cfg_stage)

        # Length-preserve: truncate or zero-pad to match original_len
        if len(audio) > original_len:
            audio = audio[:original_len]
        elif len(audio) < original_len:
            pad_len = original_len - len(audio)
            audio = np.pad(audio, (0, pad_len), mode="constant", constant_values=0.0)

        # NaN/Inf scrub FIRST — _peak_normalize skips non-finite values, so we
        # must clean them before normalization (Bug 1 fix).
        invalid_mask = ~np.isfinite(audio)
        if np.any(invalid_mask):
            logger.warning(
                "TelephonyAugmentation.__call__: 파이프라인 후 %d개의 NaN/Inf가 "
                "감지되어 0.0으로 대체합니다.",
                int(np.sum(invalid_mask)),
            )
            audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)

        # final_normalize (reuse _peak_normalize pattern from audio_augmentation)
        # Applied AFTER scrub so non-finite values cannot bypass normalization.
        final_norm_cfg = self.cfg.get("final_normalize", {})
        if final_norm_cfg.get("enabled", True):
            target_peak = final_norm_cfg.get("target_peak", 0.95)
            audio = _peak_normalize(audio, target_peak=target_peak)

        return audio.astype(np.float32)

    # -----------------------------------------------------------------------
    # set_random_state  (reproducibility support — Decision 6)
    # -----------------------------------------------------------------------

    def set_random_state(self, rng) -> None:
        """모든 transform 인스턴스의 내부 RNG를 재시드한다.

        DataLoader worker_init_fn (Step 6.4) 내부에서 fork 후 호출된다.
        전역 ``random`` / ``numpy`` / ``torch`` 모듈의 시드는 worker_init_fn이
        직접 설정하므로 여기서는 transform 인스턴스 수준 RNG만 처리한다.

        Args:
            rng: np.random.RandomState 또는 random.Random — 시드 소스.
        """
        # audiomentations CPU compose
        if self._cpu_audiomentations_compose is not None and hasattr(
            self._cpu_audiomentations_compose, "set_random_state"
        ):
            self._cpu_audiomentations_compose.set_random_state(rng)

        # Child samplers (rir_sampler, codec_sampler) currently use global
        # random module — no per-instance RNG state to re-seed here.

    # -----------------------------------------------------------------------
    # describe
    # -----------------------------------------------------------------------

    def describe(self) -> str:
        """파이프라인 구성 요약 문자열을 반환한다.

        Returns:
            포맷된 요약 문자열 (활성 스테이지, 의존성 상태, 주요 파라미터).
        """
        lines: list = []
        lines.append("=" * 60)
        lines.append("TelephonyAugmentation Pipeline Configuration")
        lines.append("=" * 60)

        # Dependencies
        lines.append("[Dependencies]")
        lines.append(f"  ffmpeg                  : {'available' if self.has_ffmpeg else 'NOT FOUND'}")
        lines.append(f"  torchaudio              : {'available' if self.has_torchaudio else 'not installed (optional)'}")
        lines.append(f"  audiomentations         : {'available' if self.has_audiomentations else 'not installed (optional)'}")
        lines.append(f"  pyroomacoustics         : {'available' if self.has_pyroomacoustics else 'not installed (optional)'}")
        lines.append(f"  pyrnnoise               : {'available' if self.has_pyrnnoise else 'not installed (optional)'}")
        lines.append(f"  opuslib                 : {'available' if self.has_opuslib else 'not installed (optional)'}")
        lines.append(f"  g711                    : {'available' if self.has_g711 else 'not installed (optional)'}")
        lines.append(f"  g722                    : {'available' if self.has_g722 else 'not installed (optional)'}")
        lines.append(f"  webrtc_noise_gain       : {'available' if self.has_webrtc else 'not installed (optional)'}")
        lines.append("")

        # CPU pipeline stages
        lines.append("[CPU Pipeline Stages]  (CPU_PIPELINE_ORDER)")
        for stage in self.CPU_PIPELINE_ORDER:
            cfg_stage = self.cfg.get(stage, {})
            prob = cfg_stage.get("prob", 0.0)
            lines.append(f"  {stage:<20} prob={prob:.2f}  (stub — Phase 2+ will implement)")
        lines.append("")

        # Compose state
        lines.append("[Compose State]")
        lines.append(f"  _cpu_audiomentations_compose : {'set' if self._cpu_audiomentations_compose is not None else 'None (Step 5.2 will populate)'}")
        lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # _apply_* methods (dispatched via CPU_PIPELINE_ORDER)
    # Each stage method receives (self, audio: np.ndarray, sr: int, cfg_stage: dict) -> np.ndarray
    # Stage count is authoritative in CPU_PIPELINE_ORDER — do not hard-code here.
    # -----------------------------------------------------------------------

    def _apply_noise_inject(
        self, audio: np.ndarray, sr: int, cfg_stage: Dict[str, Any]
    ) -> np.ndarray:
        """Additive noise (Stage 0) — caller speech 에 잡음 합성하여 noisy input
        분포 생성. 이후 stage 의 NS / codec 이 noisy speech 에서 음성을 over-suppress
        하는 telephony-typical artifact 를 재현하는 데 결정적.

        Config keys (모두 선택):
          noise_dir:    str | None    # noise wav/flac 디렉토리 (재귀 X — 1 level)
          noise_type:   list[str] | str  # synthetic 노이즈 fallback: white/pink/brown
          snr_range:    list[float, float]   # SNR (dB), default [5, 25]
          dual_noise:   bool          # True 면 real (noise_dir) + synthetic 동시
        """
        if len(audio) == 0:
            return audio

        snr_range = _ensure_range(cfg_stage.get("snr_range", [5, 25]), [5, 25])
        noise_types = cfg_stage.get("noise_type", ["white", "pink", "brown"])
        if isinstance(noise_types, str):
            noise_types = [noise_types]
        if not noise_types:
            noise_types = ["white"]

        noise_dir = cfg_stage.get("noise_dir")
        dual_noise = bool(cfg_stage.get("dual_noise", False)) and bool(noise_dir)

        audio_rms = _rms(audio)
        audio_out = audio.astype(np.float64)

        def _scale(noise_arr: np.ndarray, snr_db: float) -> np.ndarray:
            return noise_arr * audio_rms / (10 ** (snr_db / 20.0)) / (_rms(noise_arr) + 1e-8)

        def _load_random_noise_file() -> np.ndarray:
            """Pick a random noise file from noise_dir, load + sr-match + length-match.

            Uses soundfile (no librosa numba dependency). sr mismatch handled
            via the module-level _taF.resample (or scipy fallback) inline.
            """
            from pathlib import Path as _P
            d = _P(noise_dir)
            cands = list(d.glob("*.wav")) + list(d.glob("*.flac"))
            if not cands:
                return None
            path = random.choice(cands)
            try:
                noise_raw, n_sr = sf.read(str(path), dtype="float32")
                if noise_raw.ndim == 2:
                    noise_raw = noise_raw.mean(axis=1)
                # SR match (inline pattern — same as elsewhere in module)
                if int(n_sr) != int(sr):
                    if _taF is not None:
                        t = _torch.from_numpy(np.ascontiguousarray(noise_raw, dtype=np.float32))
                        noise_raw = _taF.resample(t, int(n_sr), int(sr)).numpy()
                    else:
                        g = _gcd(int(n_sr), int(sr))
                        noise_raw = signal.resample_poly(noise_raw, int(sr) // g, int(n_sr) // g).astype(np.float32)
                # Length match — tile or trim
                if len(noise_raw) == 0:
                    return None
                if len(noise_raw) >= len(audio):
                    # Random start offset for variety
                    start = random.randint(0, len(noise_raw) - len(audio))
                    noise_raw = noise_raw[start : start + len(audio)]
                else:
                    repeats = int(np.ceil(len(audio) / len(noise_raw)))
                    noise_raw = np.tile(noise_raw, repeats)[: len(audio)]
                return noise_raw.astype(np.float64)
            except Exception as exc:
                logger.warning("_apply_noise_inject: 노이즈 파일 load 실패 (%s): %s", path, exc)
                return None

        # Path A: real-noise file
        if noise_dir:
            real_noise = _load_random_noise_file()
            if real_noise is not None:
                snr = random.uniform(*snr_range)
                audio_out = audio_out + _scale(real_noise, snr)

        # Path B: synthetic colored noise
        if (not noise_dir) or dual_noise:
            snr = random.uniform(*snr_range)
            ntype = random.choice(noise_types) if isinstance(noise_types, list) else noise_types
            synth = _generate_colored_noise(len(audio), ntype).astype(np.float64)
            audio_out = audio_out + _scale(synth, snr)

        audio_out = np.clip(audio_out, -1.0, 1.0)
        return audio_out.astype(np.float32)

    def _apply_rir(
        self, audio: np.ndarray, sr: int, cfg_stage: Dict[str, Any]
    ) -> np.ndarray:
        """RIR 컨볼루션 (Stage A) — RIRSampler 기반 구현.

        manifest → rir_dir → pyroomacoustics → 합성 RIR 우선 순위로 샘플링한다.
        ``room_type='handset'``이면 dirac IR을 사용하여 컨볼루션이 identity가 된다.

        Args:
            audio: 입력 float32 mono 배열.
            sr: 샘플링 레이트.
            cfg_stage: 'rir' 스테이지 config dict. 주요 키:
                - ``room_type`` (str): 방 유형 (handset/meeting/office 등).
                - ``wet_ratio_range`` (list[float, float]): wet 비율 범위.

        Returns:
            RIR 컨볼루션 + wet/dry 혼합이 적용된 float32 배열 (원본과 길이 동일).
        """
        original_len = len(audio)

        # Runtime override: use caller-provided IR instead of RIRSampler
        override = cfg_stage.get("rir_override")
        if override is not None:
            rir = override
        else:
            room_type = cfg_stage.get("room_type")

            # Bug 2 fix: when room_type is not explicitly set in config, sample it
            # per the weighted room_type_split distribution instead of passing None
            # (which caused room_type_split to be silently ignored).
            if room_type is None:
                room_type = self.rir_sampler.sample_room_type()

            rir = self.rir_sampler.sample(sr, room_type=room_type)

        # Dirac IR 경로 (handset deferred + dirac runtime override) — convolution
        # identity + wet/dry mix + clip 모두 우회. 임의 진폭 input (예: random.randn —
        # [-1,1] 범위 벗어남) 에서도 `output == input` 보장 (test 기대치).
        # NOTE: dirac 경로는 wet_ratio uniform 을 소비하지 않으므로 downstream
        # stage RNG 가 baseline 대비 *덜* 소비됨 — 이는 의도된 short-circuit. caller
        # 가 RNG 결정성을 강하게 요구하면 dirac 대신 일반 IR 사용 권장.
        if rir.size == 1 and np.isclose(rir[0], 1.0):
            return audio.astype(np.float32)

        # 컨볼루션 (length-preserving truncate)
        reverbed = signal.fftconvolve(audio, rir, mode="full")[:original_len]

        wet_ratio = random.uniform(
            *_ensure_range(cfg_stage.get("wet_ratio_range", [0.3, 0.8]), [0.3, 0.8])
        )
        orig_rms = _rms(audio)
        mixed = (1.0 - wet_ratio) * audio + wet_ratio * reverbed

        # 에너지 정규화 (원본 RMS 유지) — audio_augmentation._apply_reverb 패턴 그대로
        mixed_rms = _rms(mixed)
        out = mixed * (orig_rms / (mixed_rms + 1e-8))
        out = np.clip(out, -1.0, 1.0)

        return out.astype(np.float32)

    def _apply_near_end_dsp(
        self, audio: np.ndarray, sr: int, cfg_stage: Dict[str, Any]
    ) -> np.ndarray:
        """Near-end DSP (Stage B) — 코덱 전 WebRTC APM 노이즈 억제 + AGC.

        webrtc-noise-gain 라이브러리를 사용하여 실제 WebRTC APM 처리를
        시뮬레이션한다 (near-end 측, 즉 마이크 입력 직후).

        처리 절차:
          1. 16kHz 리샘플링 (torchaudio.functional.resample, scipy fallback)
          2. float32 → int16 변환
          3. 10ms 청크 단위(160 샘플) 처리
          4. int16 → float32 역변환
          5. 원본 SR로 리샘플링
          6. 길이 맞추기

        Config keys:
          noise_suppression_range: [int, int]  # 1=Low … 4=VeryHigh
          auto_gain_range:         [int, int]  # 0=off, 1-31 dB target
        """
        if not _HAS_WEBRTC:
            logger.debug(
                "_apply_near_end_dsp: webrtc_noise_gain 미설치 — 원본 반환."
            )
            return audio

        ns_level = random.randint(
            *_ensure_range(cfg_stage.get("noise_suppression_range", [1, 4]), [1, 4])
        )
        ag_level = random.randint(
            *_ensure_range(cfg_stage.get("auto_gain_range", [0, 31]), [0, 31])
        )

        # 둘 다 0이면 처리 불필요
        if ns_level == 0 and ag_level == 0:
            return audio

        original_len = len(audio)
        WEBRTC_SR = 16000
        CHUNK_SIZE = 160  # 10ms @ 16kHz

        # 1. 16kHz 리샘플링
        if int(sr) == int(WEBRTC_SR):
            audio_16k = audio.astype(np.float32, copy=False)
        elif _HAS_TORCHAUDIO:
            _t = _torch.from_numpy(np.asarray(audio, dtype=np.float32))
            audio_16k = _taF.resample(_t, int(sr), int(WEBRTC_SR)).numpy()
        else:
            _g = _gcd(int(sr), int(WEBRTC_SR))
            audio_16k = signal.resample_poly(audio, int(WEBRTC_SR) // _g, int(sr) // _g).astype(np.float32)

        # 2. float32 → int16
        audio_int16 = np.clip(audio_16k * 32767.0, -32768, 32767).astype(np.int16)

        # 3. 패딩: CHUNK_SIZE 배수로
        n_samples_16k = len(audio_int16)
        remainder = n_samples_16k % CHUNK_SIZE
        if remainder != 0:
            pad_len = CHUNK_SIZE - remainder
            audio_int16 = np.pad(audio_int16, (0, pad_len), mode="constant", constant_values=0)

        # AudioProcessor 호출마다 새로 생성 (상태 초기화 보장)
        ap = _WebRTCAudioProcessor(ag_level, ns_level)

        # 4. 청크 단위 처리
        processed_chunks = []
        for offset in range(0, len(audio_int16), CHUNK_SIZE):
            chunk = audio_int16[offset: offset + CHUNK_SIZE]
            result = ap.Process10ms(chunk.tobytes())
            chunk_out = np.frombuffer(result.audio, dtype=np.int16)
            processed_chunks.append(chunk_out)

        processed_int16 = np.concatenate(processed_chunks)

        # 패딩 제거 (원본 16kHz 샘플 수 기준)
        processed_int16 = processed_int16[:n_samples_16k]

        # 5. int16 → float32
        audio_processed = processed_int16.astype(np.float32) / 32767.0

        # 6. 원본 SR로 리샘플링
        if int(WEBRTC_SR) == int(sr):
            audio_processed = audio_processed.astype(np.float32, copy=False)
        elif _HAS_TORCHAUDIO:
            _t = _torch.from_numpy(np.asarray(audio_processed, dtype=np.float32))
            audio_processed = _taF.resample(_t, int(WEBRTC_SR), int(sr)).numpy()
        else:
            _g = _gcd(int(WEBRTC_SR), int(sr))
            audio_processed = signal.resample_poly(audio_processed, int(sr) // _g, int(WEBRTC_SR) // _g).astype(np.float32)

        # 7. 길이 맞추기
        if len(audio_processed) >= original_len:
            audio_processed = audio_processed[:original_len]
        else:
            pad_len = original_len - len(audio_processed)
            audio_processed = np.pad(
                audio_processed, (0, pad_len), mode="constant", constant_values=0.0
            )

        return audio_processed.astype(np.float32)

    def _apply_codec(
        self, audio: np.ndarray, sr: int, cfg_stage: Dict[str, Any]
    ) -> np.ndarray:
        """코덱 인코딩/디코딩 라운드트립 (Stage C).

        CodecSampler로 코덱 설정을 샘플링하고 codec_roundtrip을 통해
        인코딩-디코딩 사이클을 적용한다.

        Args:
            audio: float32 mono 입력 오디오.
            sr: 샘플링 레이트.
            cfg_stage: 'codec' 스테이지 config dict (현재 미사용 — 샘플링은
                       self.codec_sampler 내부에서 처리).

        Returns:
            코덱 라운드트립이 적용된 float32 배열 (원본과 길이 동일).
            codec_sampler가 None이거나 모든 백엔드 실패 시 원본 반환.
        """
        if self.codec_sampler is None:
            return audio

        codec_cfg = self.codec_sampler.sample_config()
        try:
            out, applied = codec_roundtrip(audio, sr, codec_cfg)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "_apply_codec: codec_roundtrip 실패 (format=%s, encoder=%s): %s",
                codec_cfg.get("format"),
                codec_cfg.get("encoder"),
                exc,
            )
            return audio

        # distortion logger (Step 6.2; may be None).
        # Bug 3 fix: only log when the codec was actually applied (applied=True).
        # When applied=False all backends fell through → distortion_logger would
        # record a false positive codec entry for what is actually a no-op.
        if self.distortion_logger is not None and applied:
            self.distortion_logger.log("codec", codec_cfg)

        return out.astype(np.float32)

    def _apply_packet_loss_plc(
        self, audio: np.ndarray, sr: int, cfg_stage: Dict[str, Any]
    ) -> np.ndarray:
        """패킷 손실 시뮬레이션 + PLC 복원 (Stages D+E).

        Gilbert-Elliott 2-state Markov chain으로 손실 마스크를 생성하고,
        샘플링된 PLC 전략(App I pitch-period repetition 또는 zero-fill)을
        적용하여 손실 프레임을 복원한다.

        Frame size 기본값(20 ms)은 AMR-NB/WB 및 Opus(VoIP) 표준에 해당:
          - G.711  = 10 ms (80 samples @ 8 kHz)
          - AMR-NB = 20 ms (160 samples @ 8 kHz)
          - AMR-WB = 20 ms (320 samples @ 16 kHz)
          - Opus   = 2.5/5/10/20/40/60 ms (20 ms 일반적, VoIP 기본값)
        단일 config 키 ``frame_size_ms``로 제어.
        **Future work**: ``self.codec_sampler.last_sampled_config``에서
        활성 코덱의 공칭 프레임 크기를 읽어 자동 파생.

        PLC 전략:
          - ``"app_i"`` (기본 가중치 0.7): G.711 App I 피치 반복 + 지수 감쇠
          - ``"zero"``  (기본 가중치 0.3): 손실 프레임 무음 + 경계 fade
            (audio_augmentation._apply_packet_loss:858-892 패턴 재사용)

        Args:
            audio: float32 mono 입력 오디오.
            sr: 샘플링 레이트.
            cfg_stage: 'packet_loss_plc' 스테이지 config dict. 주요 키:
                - ``frame_size_ms`` (int, 기본 20)
                - ``p_good_to_bad_range`` (list[float,float], 기본 [0.02, 0.10])
                - ``p_bad_to_good_range`` (list[float,float], 기본 [0.3, 0.8])
                - ``plc_weights`` (list[float], 기본 [0.7, 0.3])

        Returns:
            PLC 복원이 적용된 float32 배열 (원본과 길이 동일).
        """
        # Graceful: empty / very short audio → no-op
        if len(audio) == 0:
            return audio.astype(np.float32)

        original_len = len(audio)

        # 1. Frame size resolution
        frame_size_ms = cfg_stage.get("frame_size_ms", 20)
        frame_size = int(sr * frame_size_ms / 1000)
        frame_size = max(frame_size, 1)
        num_frames = math.ceil(len(audio) / frame_size)

        # 2. Sample Gilbert-Elliott transition probabilities
        p_g2b = random.uniform(
            *_ensure_range(cfg_stage.get("p_good_to_bad_range", [0.02, 0.10]), [0.02, 0.10])
        )
        p_b2g = random.uniform(
            *_ensure_range(cfg_stage.get("p_bad_to_good_range", [0.3, 0.8]), [0.3, 0.8])
        )

        # 3. Generate loss mask (global random — worker_init_fn seeds per worker)
        loss_mask = gilbert_elliott_mask(num_frames, p_g2b, p_b2g, rng=None)

        # 4. Sample PLC strategy
        plc_weights = cfg_stage.get("plc_weights", [0.7, 0.3])
        strategy = random.choices(["app_i", "zero"], weights=plc_weights)[0]

        # 5. Apply fill
        if strategy == "app_i":
            out = g711_app_i_plc_fill(audio, frame_size, loss_mask, sr)
        else:
            # zero-fill + boundary fade (reuse audio_augmentation._apply_packet_loss pattern)
            out = audio.copy().astype(np.float32)
            fade_size = int(0.002 * sr)  # 2 ms fade
            fade_size = max(1, min(fade_size, frame_size // 2))
            for i, is_received in enumerate(loss_mask):
                start = i * frame_size
                end = min(start + frame_size, len(out))
                if not is_received:
                    out[start:end] = 0.0
                else:
                    # Fade-in if previous frame was lost
                    if i > 0 and not loss_mask[i - 1]:
                        fade_end = min(start + fade_size, end)
                        actual_fade = fade_end - start
                        if actual_fade > 0:
                            fade_in = np.linspace(0.0, 1.0, actual_fade, dtype=np.float32)
                            out[start:fade_end] *= fade_in
                    # Fade-out if next frame will be lost
                    if i < num_frames - 1 and not loss_mask[i + 1]:
                        fade_start = max(end - fade_size, start)
                        actual_fade = end - fade_start
                        if actual_fade > 0:
                            fade_out = np.linspace(1.0, 0.0, actual_fade, dtype=np.float32)
                            out[fade_start:end] *= fade_out

        # 6. Log
        if self.distortion_logger is not None:
            empirical_loss_rate = float((~loss_mask).mean())
            self.distortion_logger.log(
                "packet_loss_plc",
                {
                    "frame_size_ms": frame_size_ms,
                    "p_good_to_bad": p_g2b,
                    "p_bad_to_good": p_b2g,
                    "strategy": strategy,
                    "empirical_loss_rate": empirical_loss_rate,
                },
            )

        # 7. Length-preserve + return float32
        if len(out) > original_len:
            out = out[:original_len]
        elif len(out) < original_len:
            out = np.pad(out, (0, original_len - len(out)), mode="constant")

        return out.astype(np.float32)

    def _apply_far_end_dsp(
        self, audio: np.ndarray, sr: int, cfg_stage: Dict[str, Any]
    ) -> np.ndarray:
        """Far-end DSP (Stage F) — NS 캐스케이드 (RNNoise / WebRTC / none).

        backend_weights로 백엔드를 확률적으로 선택한다:
          - "webrtc": near_end_dsp WebRTC APM 로직을 재사용.
          - "rnnoise": pyrnnoise를 통한 RNN 기반 노이즈 억제.
          - "none"   : 오디오를 변경하지 않고 반환.

        Config keys:
          backend_weights: [float, float, float]  # [webrtc, rnnoise, none] 가중치
          noise_suppression_range: [int, int]     # webrtc 백엔드 전달
          auto_gain_range:         [int, int]     # webrtc 백엔드 전달
        """
        backend_weights = cfg_stage.get("backend_weights", [0.5, 0.3, 0.2])
        backend = random.choices(["webrtc", "rnnoise", "none"], weights=backend_weights)[0]

        if backend == "webrtc":
            return self._apply_near_end_dsp(audio, sr, cfg_stage)

        elif backend == "rnnoise":
            if not _HAS_PYRNNOISE:
                logger.debug(
                    "_apply_far_end_dsp: pyrnnoise 미설치 — 원본 반환."
                )
                return audio

            original_len = len(audio)
            RNNOISE_FRAME = 480  # 10ms @ 48kHz

            from pyrnnoise import RNNoise
            ns = RNNoise(sample_rate=sr)

            # 480 샘플 배수로 패딩
            remainder = len(audio) % RNNOISE_FRAME
            if remainder != 0:
                pad_len = RNNOISE_FRAME - remainder
                padded = np.pad(audio, (0, pad_len), mode="constant", constant_values=0.0)
            else:
                padded = audio

            # pyrnnoise API: process_frame(frame) → ndarray or scalar
            # 프레임 단위 처리 후 누적
            processed_chunks = []
            for i in range(0, len(padded), RNNOISE_FRAME):
                frame = padded[i: i + RNNOISE_FRAME].astype(np.float32)
                result = ns.process_frame(frame)
                if isinstance(result, np.ndarray):
                    processed_chunks.append(result)
                else:
                    # fallback: result가 scalar이거나 None이면 원본 프레임 사용
                    processed_chunks.append(frame)

            out = np.concatenate(processed_chunks).astype(np.float32)

            # 원본 길이로 트리밍
            out = out[:original_len]

            return out

        else:  # "none"
            return audio

    def _apply_bandwidth_switch(
        self, audio: np.ndarray, sr: int, cfg_stage: Dict[str, Any]
    ) -> np.ndarray:
        """대역폭 전환 (Stage G) — NB ↔ WB ↔ FB 리샘플 아티팩트 시뮬레이션.

        target_mode로 NB(8kHz) / WB(16kHz) / FB(48kHz) 중 하나를 샘플링하여
        다운샘플 → 업샘플로 대역 제한 아티팩트를 재현한다.

        처리 절차:
          1. target_sr 결정 (NB=8000, WB=16000, FB=48000)
          2. sr == target_sr이면 no-op 반환
          3. audio → target_sr 다운/업샘플 (torchaudio.functional.resample, scipy fallback)
          4. intermediate → sr로 역샘플 (torchaudio.functional.resample, scipy fallback)
          5. 원본 길이 맞추기

        Config keys:
          target_modes:   list[str]   # 기본 ["NB", "WB", "FB"]
          target_weights: list[float] # 기본 균등 분포
        """
        original_len = len(audio)

        target_modes = cfg_stage.get("target_modes", ["NB", "WB", "FB"])
        target_weights = cfg_stage.get("target_weights", None)  # None → uniform

        target_mode = random.choices(target_modes, weights=target_weights)[0]

        mode_sr_map = {"NB": 8000, "WB": 16000, "FB": 48000}
        target_sr = mode_sr_map.get(target_mode, sr)

        if target_sr == sr:
            return audio

        # 다운샘플 → 업샘플로 대역 제한 아티팩트 재현
        if int(sr) == int(target_sr):
            intermediate = audio.astype(np.float32, copy=False)
        elif _HAS_TORCHAUDIO:
            _t = _torch.from_numpy(np.asarray(audio, dtype=np.float32))
            intermediate = _taF.resample(_t, int(sr), int(target_sr)).numpy()
        else:
            _g = _gcd(int(sr), int(target_sr))
            intermediate = signal.resample_poly(audio, int(target_sr) // _g, int(sr) // _g).astype(np.float32)
        if int(target_sr) == int(sr):
            back = intermediate.astype(np.float32, copy=False)
        elif _HAS_TORCHAUDIO:
            _t = _torch.from_numpy(np.asarray(intermediate, dtype=np.float32))
            back = _taF.resample(_t, int(target_sr), int(sr)).numpy()
        else:
            _g = _gcd(int(target_sr), int(sr))
            back = signal.resample_poly(intermediate, int(sr) // _g, int(target_sr) // _g).astype(np.float32)

        # 길이 맞추기
        if len(back) >= original_len:
            back = back[:original_len]
        else:
            pad_len = original_len - len(back)
            back = np.pad(back, (0, pad_len), mode="constant", constant_values=0.0)

        return back.astype(np.float32)

    def _apply_mic_capture_cpu(
        self, audio: np.ndarray, sr: int, cfg_stage: Dict[str, Any]
    ) -> np.ndarray:
        """마이크 캡처 CPU 단계 (Stage I) — 클리핑 + 게인 시뮬레이션.

        audiomentations Compose (pre-built in __init__)가 있으면 위임하고,
        없으면 manual fallback으로 하드 클리핑 + 선형 게인을 적용한다.

        Config keys:
          clip_prob:             float        # 클리핑 적용 확률
          clip_threshold_range:  [float, float]  # 클리핑 임계값 범위 (absolute)
          gain_db_range:         [float, float]  # 게인 범위 (dB)
          gain_prob:             float        # 게인 적용 확률
        """
        original_len = len(audio)

        # Path A: pre-built audiomentations Compose
        if _HAS_AUDIOMENTATIONS and self._cpu_audiomentations_compose is not None:
            try:
                out = self._cpu_audiomentations_compose(samples=audio, sample_rate=sr)
                if len(out) >= original_len:
                    out = out[:original_len]
                else:
                    out = np.pad(out, (0, original_len - len(out)), mode="constant")
                return out.astype(np.float32)
            except Exception as exc:
                logger.debug(
                    "_apply_mic_capture_cpu: audiomentations Compose 실패: %s — manual fallback.",
                    exc,
                )

        # Path B: manual fallback — hard clip + gain
        out = audio.copy().astype(np.float32)

        clip_prob = float(cfg_stage.get("clip_prob", 0.0))
        if random.random() < clip_prob:
            thr_range = cfg_stage.get("clip_threshold_range", [0.7, 0.95])
            threshold = random.uniform(float(thr_range[0]), float(thr_range[1]))
            # Hard clip at absolute threshold (reuse audio_augmentation._apply_clipping pattern)
            out = np.clip(out, -threshold, threshold)

        gain_range = cfg_stage.get("gain_db_range", None)
        gain_prob = float(cfg_stage.get("gain_prob", 0.0))
        if gain_range is not None and random.random() < gain_prob:
            gain_db = random.uniform(float(gain_range[0]), float(gain_range[1]))
            out *= 10 ** (gain_db / 20.0)

        # Length-preserve
        if len(out) >= original_len:
            out = out[:original_len]
        else:
            out = np.pad(out, (0, original_len - len(out)), mode="constant")

        return out.astype(np.float32)
