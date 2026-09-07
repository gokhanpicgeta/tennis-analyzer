# tennis-analyzer

Cuts dead time out of tennis session recordings. Runs YOLOv11 on frames to
detect players + ball, groups active moments into rallies, and stitches those
segments into a new MP4 with FFmpeg.

## Setup (one time)

```bash
brew install ffmpeg
cd ~/Coding/tennis-analyzer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

First run downloads `yolo11n.pt` (~6MB) automatically.

## Usage

```bash
source .venv/bin/activate
python analyze.py ~/Movies/tennis.mp4 --mode rally
python analyze.py ~/Movies/tennis.mp4 --mode match -o highlights.mp4
python analyze.py ~/Movies/tennis.mp4 --dry-run       # just print segments
```

Modes:
- `rally` — only the ball-in-play moments, ~1.5s padding
- `match` — includes serve setup, wider merges (5s padding, 4s gap tolerance)

## Getting the video off the phone

Recordings save to `Movies/TennisTracker/` on the phone. Easiest transfer paths:
- USB + [OpenMTP](https://openmtp.ganeshrvel.com/) (macOS)
- AirDrop from the phone's Files app after copying to a share
- Google Drive / Dropbox upload

## Performance

On an M-series Mac, sampling at 2Hz analyzes ~2h of FHD video in ~5–10 min.
Encoding uses `h264_videotoolbox` (hardware) — a ~30 min rally output takes
another ~1–2 min. Bump `--sample-hz` to 4 for tighter detection at the cost
of runtime.

## Tuning

Edit `detector.py` if detection is off:
- `motion_threshold` (4.0) — higher = requires more player movement
- `gap_seconds` (2.5 rally / 4.0 match) — max quiet time inside a rally
- `min_rally_seconds` (1.5) — discard short bursts (false positives)
- `padding` (1.5 rally / 5.0 match) — seconds added before/after each rally

The classifier currently marks a frame active when:
- a `sports ball` is detected, **or**
- ≥2 people are detected **and** frame-to-frame motion exceeds `motion_threshold`
