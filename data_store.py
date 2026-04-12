"""
Self-play verilerini diske kaydetme ve yükleme.
Format: data/selfplay_NNNN.npz (numpy compressed)
"""

import os
import glob
import numpy as np
from pathlib import Path
from typing import List, Tuple
from logger import get_logger

log = get_logger("data_store")

DATA_DIR = "data"

Sample = Tuple[np.ndarray, np.ndarray, float]


def save_samples(samples: List[Sample], data_dir: str = DATA_DIR) -> str:
    """
    Örnekleri diske kaydet.
    Returns: kaydedilen dosya yolu
    """
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    # Mevcut dosya sayısına göre numara ver
    existing = sorted(glob.glob(os.path.join(data_dir, "selfplay_*.npz")))
    idx = len(existing)
    path = os.path.join(data_dir, f"selfplay_{idx:04d}.npz")

    states   = np.array([s[0] for s in samples], dtype=np.float32)
    policies = np.array([s[1] for s in samples], dtype=np.float32)
    values   = np.array([s[2] for s in samples], dtype=np.float32)

    np.savez_compressed(path, states=states, policies=policies, values=values)
    size_kb = os.path.getsize(path) / 1024
    log.info(f"Veri kaydedildi: {path} ({len(samples)} örnek, {size_kb:.1f} KB)")
    return path


def load_samples(data_dir: str = DATA_DIR, max_files: int = None) -> List[Sample]:
    """
    data/ dizinindeki tüm .npz dosyalarını yükle.
    max_files: sadece en son N dosyayı yükle (None = hepsi)
    """
    files = sorted(glob.glob(os.path.join(data_dir, "selfplay_*.npz")))
    if not files:
        log.warning(f"{data_dir}/ dizininde veri bulunamadı.")
        return []

    if max_files is not None:
        files = files[-max_files:]

    all_samples = []
    total = 0
    for f in files:
        data = np.load(f)
        n = len(data["values"])
        for i in range(n):
            all_samples.append((data["states"][i], data["policies"][i], float(data["values"][i])))
        total += n

    log.info(f"{len(files)} dosyadan {total} örnek yüklendi.")
    return all_samples


def list_data(data_dir: str = DATA_DIR):
    """Kayıtlı veri dosyalarını listele."""
    files = sorted(glob.glob(os.path.join(data_dir, "selfplay_*.npz")))
    if not files:
        print(f"  {data_dir}/ dizininde veri yok.")
        return

    total_samples = 0
    total_size = 0
    print(f"\n  {'Dosya':<30} {'Örnek':>8} {'Boyut':>10}")
    print(f"  {'─'*30} {'─'*8} {'─'*10}")
    for f in files:
        data = np.load(f)
        n = len(data["values"])
        size_kb = os.path.getsize(f) / 1024
        total_samples += n
        total_size += size_kb
        print(f"  {os.path.basename(f):<30} {n:>8} {size_kb:>8.1f} KB")
    print(f"  {'─'*30} {'─'*8} {'─'*10}")
    print(f"  {'TOPLAM':<30} {total_samples:>8} {total_size:>8.1f} KB")
    print()
