"""Telephony augmentation 사용 예제.

CPU augmentation pipeline 개요
-------------------------------
TelephonyAugmentation 은 CPU 기반 단일 파이프라인으로 구성된다.

  CPU (__call__):
    비미분 가능한 telephony 효과(코덱 시뮬레이션, RIR 컨볼루션, 노이즈 믹싱 등)를
    numpy 배열에 순서대로 적용한다. 데이터 로더 worker에서 호출된다.

예제 순서
---------
1. CPU 경로: TelephonyAugmentation(yaml)(audio, sr) → sample_telephony_cpu.wav
2. DataLoader 통합: Dataset + worker_init_fn 패턴 — 재현성 시연 (패턴 출력만)
3. aug.describe() 출력

실행:
    cd NN_Zoo/audio-augmentation-pipeline
    python example_telephony_usage.py

산출물:
    sample_telephony_cpu.wav     — CPU 파이프라인 적용 결과
"""

import random
import sys
from pathlib import Path
from textwrap import dedent

import numpy as np
import soundfile as sf

from telephony_aug import TelephonyAugmentation


def main() -> None:
    pipeline_dir = Path(__file__).parent
    sample_path = pipeline_dir / "sample.wav"
    # Default config ships inside the installed package
    config_path = Path(__file__).parent.parent / "src" / "telephony_aug" / "config.yaml"
    if not config_path.exists():
        # Installed mode — locate via importlib.resources
        try:
            from importlib import resources as _res
            with _res.as_file(_res.files("telephony_aug").joinpath("config.yaml")) as p:
                config_path = Path(str(p))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 0) Guard: 필수 파일 존재 확인
    # ------------------------------------------------------------------
    if not sample_path.exists():
        print(f"[error] sample.wav를 찾을 수 없습니다: {sample_path}", file=sys.stderr)
        sys.exit(1)
    if not config_path.exists():
        print(f"[error] config.yaml을 찾을 수 없습니다: {config_path}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1) Load sample
    # ------------------------------------------------------------------
    audio, sr = sf.read(str(sample_path), dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)  # downmix to mono
    print(f"[load] sample.wav: shape={audio.shape}, sr={sr}, duration={len(audio)/sr:.2f}s")

    # ------------------------------------------------------------------
    # 2) CPU path
    # ------------------------------------------------------------------
    aug = TelephonyAugmentation(str(config_path))

    # describe() returns a full multi-line string; show first 10 lines here,
    # full output is printed at the end of this script.
    desc_lines = aug.describe().splitlines()
    preview = "\n".join(desc_lines[:10])
    print(f"\n[init] Pipeline preview:\n{preview}\n  ... (see full describe() below)")

    random.seed(42)
    np.random.seed(42)

    cpu_out = aug(audio.copy(), sr)
    out_cpu = pipeline_dir / "sample_telephony_cpu.wav"
    sf.write(str(out_cpu), cpu_out, sr, subtype="FLOAT")
    print(f"\n[cpu] {out_cpu.name} saved  shape={cpu_out.shape}  peak={float(np.max(np.abs(cpu_out))):.4f}")

    print("\n=== RIR runtime override demo ===")
    # Simulate user-provided IR (e.g., from a separate RIR dataset)
    my_rir = np.random.randn(800).astype(np.float32) * 0.05  # 800-tap noise IR
    my_rir[0] = 1.0  # direct path
    out_with_user_rir = aug(audio.copy(), sr, rir=my_rir)
    sf.write(pipeline_dir / "sample_telephony_user_rir.wav", out_with_user_rir, sr)
    print(f"[rir override] sample_telephony_user_rir.wav saved, shape={out_with_user_rir.shape}")

    # ------------------------------------------------------------------
    # 3) DataLoader integration pattern (재현성 — Decision 6)
    #    실제 DataLoader를 구동하지는 않으며, 사용 패턴만 출력한다.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[dataset] DataLoader integration pattern (worker_init_fn 재현성)")
    print("=" * 60)

    pattern_code = dedent("""\
        import random, numpy as np, torch, soundfile as sf
        from telephony_aug import TelephonyAugmentation

        class TelephonyDataset(torch.utils.data.Dataset):
            def __init__(self, audio_paths, aug: TelephonyAugmentation):
                self.paths = audio_paths
                self.aug = aug

            def __getitem__(self, idx):
                audio, sr = sf.read(self.paths[idx], dtype="float32")
                if audio.ndim == 2:
                    audio = audio.mean(axis=1)
                audio = self.aug(audio, sr)           # CPU pipeline
                return torch.from_numpy(audio), sr

            def __len__(self):
                return len(self.paths)


        def make_worker_init_fn(base_seed: int):
            \"\"\"각 worker마다 독립적·결정론적 RNG 시드를 설정한다.\"\"\"
            def _init(worker_id: int):
                seed = base_seed + worker_id
                random.seed(seed)
                np.random.seed(seed)
                info = torch.utils.data.get_worker_info()
                if info is not None and hasattr(info.dataset, "aug"):
                    info.dataset.aug.set_random_state(
                        np.random.RandomState(seed)
                    )
            return _init


        aug = TelephonyAugmentation("config.yaml")
        ds  = TelephonyDataset(audio_paths=[...], aug=aug)

        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=4,
            num_workers=2,
            worker_init_fn=make_worker_init_fn(base_seed=42),
            persistent_workers=True,
        )
    """)
    print(pattern_code)

    # ------------------------------------------------------------------
    # 4) Full describe() output
    # ------------------------------------------------------------------
    print("=" * 60)
    print("describe() — full pipeline configuration")
    print("=" * 60)
    print(aug.describe())

    print("=" * 60)
    print("example_telephony_usage.py 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
