# prepare_nsynth_subset.py (moved to data/prepare_nsynth_subset.py)
import json
import shutil
from pathlib import Path
from collections import defaultdict

# ====== ĐƯỜNG DẪN ======
PROJECT_ROOT = Path(__file__).resolve().parents[2]
NSYNTH_ROOT  = PROJECT_ROOT / "nsynth-valid"

JSON_PATH = NSYNTH_ROOT / "examples.json"   # metadata
AUDIO_DIR = NSYNTH_ROOT / "audio"           # folder audio file .wav

RAW_DIR   = PROJECT_ROOT / "data" / "raw"   # output
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ====== CONFIG ======
INSTRUMENTS = ["Drum", "Guitar", "Piano", "Violin", "Tambourine"]  # loại nhạc cụ muốn lấy
MAX_PER_CLASS = 200   # số file mỗi loại

# ====== LOAD METADATA ======
print("[INFO] Loading metadata...")
with open(JSON_PATH, "r") as f:
    meta = json.load(f)

# meta[type][instrument][file_key]
files_by_class = defaultdict(list)

print("[INFO] Filtering metadata...")
for key, info in meta.items():
    inst = info["instrument_family_str"]   
    if inst in INSTRUMENTS:
        files_by_class[inst].append(key)

# ====== COPY FILES ======
for inst in INSTRUMENTS:
    inst_dir = RAW_DIR / inst
    inst_dir.mkdir(exist_ok=True)

    selected = files_by_class[inst][:MAX_PER_CLASS]

    print(f"[INFO] Copying {len(selected)} files for {inst}...")

    for key in selected:
        fname = f"{key}.wav"
        src = AUDIO_DIR / fname
        dst = inst_dir / fname
        shutil.copy(src, dst)

print("[DONE] Subset created successfully!")
