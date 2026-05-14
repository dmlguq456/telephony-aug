"""Internal utility functions for telephony_aug.

Copied verbatim from audio_augmentation.py (original source repo:
dmlguq456/audio-augmentation-pipeline) so this package can be used
standalone without depending on that module.

Functions:
  - _check_ffmpeg: PATH/static-ffmpeg detection (cached)
  - _generate_colored_noise: white/pink/brown noise (colorednoise lib or FFT fallback)
  - _generate_synthetic_rir: exponential-decay synthetic RIR
  - _rms: root-mean-square energy
  - _ensure_range: validate 2-element list/tuple
  - _peak_normalize: peak-target normalization with NaN/Inf guard
  - _validate_audio: dtype/shape/NaN normalization at pipeline entry
"""

from __future__ import annotations

import functools
import logging
import os
import shutil
import subprocess

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional dependency probes
# ---------------------------------------------------------------------------

try:
    import colorednoise as _colorednoise
    _HAS_COLOREDNOISE = True
except ImportError:
    _colorednoise = None
    _HAS_COLOREDNOISE = False


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _parse_ffmpeg_encoders() -> list:
    """``ffmpeg -encoders`` 출력을 파싱해 인코더 이름 목록을 반환한다.

    ffmpeg 미설치 또는 timeout 시 빈 list 반환 (예외 raise 안 함).
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-encoders", "-v", "quiet"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        encoders = []
        for line in result.stdout.splitlines():
            line = line.strip()
            # 인코더 목록 행 형식: " V..... libmp3lame ..."
            if len(line) > 7 and line[0] in "VASD" and line[1] == ".":
                parts = line.split()
                if len(parts) >= 2 and parts[1] != "=":
                    encoders.append(parts[1])
        return encoders
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        logger.warning("telephony_aug: ffmpeg 인코더 목록 파싱 실패: %s", exc)
        return []


@functools.lru_cache(maxsize=1)
def _check_ffmpeg() -> bool:
    """ffmpeg 바이너리 존재 여부를 확인한다 (결과 캐시됨).

    시스템 PATH에 ffmpeg가 없으면 static-ffmpeg 패키지에서 가져와
    PATH에 등록한다.
    """
    if shutil.which("ffmpeg") is not None:
        return True
    # static-ffmpeg 패키지가 설치되어 있으면 PATH에 등록
    try:
        from static_ffmpeg import run as _sf_run
        ffmpeg_path, _ = _sf_run.get_or_fetch_platform_executables_else_raise()
        ffmpeg_dir = os.path.dirname(ffmpeg_path)
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        return shutil.which("ffmpeg") is not None
    except Exception:
        return False


def _generate_colored_noise(
    n_samples: int,
    noise_type: str = "white",
    rng: np.random.RandomState = None,
) -> np.ndarray:
    """컬러 노이즈(white/pink/brown)를 생성한다.

    colorednoise 라이브러리가 설치된 경우 이를 우선 사용하고,
    없으면 FFT 기반 자체 구현으로 대체한다.

    Args:
        n_samples: 생성할 샘플 수.
        noise_type: 'white', 'pink', 'brown' 중 하나.
        rng: numpy RandomState 인스턴스. None이면 전역 np.random 사용.

    Returns:
        shape (n_samples,) float64 노이즈 배열.
    """
    if rng is None:
        rng = np.random.RandomState()

    if _HAS_COLOREDNOISE:
        beta_map = {"white": 0, "pink": 1, "brown": 2}
        beta = beta_map.get(noise_type, 0)
        return _colorednoise.powerlaw_psd_gaussian(beta, n_samples, random_state=rng)

    # Fallback: FFT-based colored noise generation
    white = rng.randn(n_samples)

    if noise_type == "white":
        return white

    X = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, d=1.0)
    freqs[0] = 1.0  # DC 성분 0 나눗셈 방지

    if noise_type == "pink":
        X *= 1.0 / np.sqrt(freqs)   # 1/f 스펙트럼
    elif noise_type == "brown":
        X *= 1.0 / freqs             # 1/f² 스펙트럼

    X[0] = 0.0  # DC 성분 제거
    result = np.fft.irfft(X, n=n_samples)
    return result


def _generate_synthetic_rir(
    sr: int,
    rt60: float,
    rng: np.random.RandomState = None,
) -> np.ndarray:
    """RT60 값을 기반으로 합성 RIR(Room Impulse Response)을 생성한다.

    지수 감쇠 모델: h(t) = noise * exp(-6.9 * t / RT60)
    -60dB 아래 구간은 제거하고 peak를 1.0으로 정규화한다.
    """
    if rng is None:
        rng = np.random.RandomState()

    n_samples = int(sr * rt60 * 1.2)
    t = np.arange(n_samples) / sr

    envelope = np.exp(-6.9 * t / rt60)
    noise = rng.randn(n_samples)
    h = noise * envelope

    threshold = 10 ** (-60.0 / 20.0)
    cutoff = np.searchsorted(-envelope, -threshold)
    if cutoff > 0:
        h = h[:cutoff]

    peak = np.max(np.abs(h))
    if peak > 1e-8:
        h = h / peak

    return h


def _rms(x: np.ndarray, eps: float = 1e-8) -> float:
    """배열의 RMS(Root Mean Square) 에너지를 반환한다."""
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + eps))


def _ensure_range(val, default: list) -> list:
    """Config 값이 2-element list/tuple인지 확인. 아니면 default 반환."""
    if isinstance(val, (list, tuple)) and len(val) == 2:
        return val
    logger.warning("Config 값 %r이 2-element range가 아닙니다. 기본값 %s 사용.", val, default)
    return default


def _peak_normalize(x: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """배열의 peak를 target_peak로 정규화한다. NaN/Inf 포함 시 그대로 반환."""
    if not np.all(np.isfinite(x)):
        logger.warning("_peak_normalize: NaN/Inf 값이 포함되어 정규화를 건너뜁니다.")
        return x

    peak = np.max(np.abs(x))
    if peak < 1e-8:
        return x

    return (x * (target_peak / peak)).astype(x.dtype)


def _validate_audio(audio: np.ndarray, sr: int):
    """오디오 배열의 유효성을 검증하고 정규화된 형태로 반환한다.

    수행 작업:
    - float32로 변환
    - NaN/Inf 값을 0.0으로 대체 (logger.warning으로 카운트 보고)
    - 2D 입력(멀티채널) → 첫 번째 채널 추출로 mono 보장
    - 빈 배열(len == 0) → ValueError 발생
    - sr이 None이거나 0이면 기본값 48000 사용
    """
    if sr is None or sr <= 0:
        logger.warning("_validate_audio: sr=%s는 유효하지 않습니다. 48000으로 대체합니다.", sr)
        sr = 48000

    audio = np.array(audio, dtype=np.float32)

    if audio.ndim == 2:
        if audio.shape[0] <= audio.shape[1]:
            audio = audio[0]
        else:
            audio = audio[:, 0]
    elif audio.ndim > 2:
        audio = audio.flatten()

    if len(audio) == 0:
        raise ValueError(
            "_validate_audio: 빈 오디오 배열이 입력되었습니다. "
            "upstream 데이터 파이프라인을 확인하세요."
        )

    invalid_mask = ~np.isfinite(audio)
    invalid_count = int(np.sum(invalid_mask))
    if invalid_count > 0:
        logger.warning(
            "_validate_audio: %d개의 NaN/Inf 값을 0.0으로 대체했습니다 — "
            "upstream 데이터 파이프라인을 확인하세요.",
            invalid_count,
        )
        audio[invalid_mask] = 0.0

    return audio, int(sr)
