from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

PERSON_CLASS = 0
BALL_CLASS = 32
MOTION_INPUT_SIZE = (320, 180)


@dataclass
class ActivitySample:
    t: float
    active: bool
    people: int
    ball: bool
    motion: float


def analyze_video(
    video_path: Path,
    sample_hz: float = 2.0,
    motion_threshold: float = 4.0,
    conf: float = 0.35,
) -> tuple[list[ActivitySample], float]:
    """Sample the video and return per-sample activity + video duration (s)."""
    model = YOLO("yolo11n.pt")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    frame_stride = max(1, int(round(fps / sample_hz)))

    samples: list[ActivitySample] = []
    prev_small = None
    frame_idx = 0

    with tqdm(total=frame_count, unit="frame", desc="analyzing") as bar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            bar.update(1)

            if frame_idx % frame_stride == 0:
                t = frame_idx / fps

                results = model(
                    frame,
                    classes=[PERSON_CLASS, BALL_CLASS],
                    verbose=False,
                    conf=conf,
                    imgsz=640,
                )
                people = 0
                ball = False
                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls)
                        if cls == PERSON_CLASS:
                            people += 1
                        elif cls == BALL_CLASS:
                            ball = True

                small = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                small = cv2.resize(small, MOTION_INPUT_SIZE)
                motion = 0.0
                if prev_small is not None:
                    motion = float(np.mean(cv2.absdiff(small, prev_small)))
                prev_small = small

                active = ball or (people >= 2 and motion > motion_threshold)
                samples.append(
                    ActivitySample(t=t, active=active, people=people, ball=ball, motion=motion)
                )

            frame_idx += 1

    cap.release()
    return samples, duration


def build_rallies(
    samples: list[ActivitySample],
    duration: float,
    gap_seconds: float = 2.5,
    min_rally_seconds: float = 1.5,
    padding: float = 1.5,
) -> list[tuple[float, float]]:
    """Group active samples into rally segments, then pad and clamp to duration."""
    rallies: list[tuple[float, float]] = []
    start: float | None = None
    last_active: float | None = None

    for s in samples:
        if s.active:
            if start is None:
                start = s.t
            last_active = s.t
        else:
            if start is not None and last_active is not None:
                if s.t - last_active > gap_seconds:
                    if last_active - start >= min_rally_seconds:
                        rallies.append((start, last_active))
                    start = None
                    last_active = None

    if start is not None and last_active is not None:
        if last_active - start >= min_rally_seconds:
            rallies.append((start, last_active))

    padded: list[tuple[float, float]] = []
    for s, e in rallies:
        padded.append((max(0.0, s - padding), min(duration, e + padding)))

    return merge_overlapping(padded)


def merge_overlapping(segments: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not segments:
        return []
    segments = sorted(segments)
    merged = [segments[0]]
    for s, e in segments[1:]:
        ls, le = merged[-1]
        if s <= le:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


def detect_rallies(
    video_path: Path,
    mode: str = "rally",
    sample_hz: float = 2.0,
) -> tuple[list[tuple[float, float]], float]:
    """
    Detect rally segments in the video.

    mode="rally": tight cuts, ~1.5s padding
    mode="match": wider cuts, ~5s padding (captures serve setup)
    """
    samples, duration = analyze_video(video_path, sample_hz=sample_hz)
    padding = 5.0 if mode == "match" else 1.5
    gap = 4.0 if mode == "match" else 2.5
    rallies = build_rallies(
        samples,
        duration,
        gap_seconds=gap,
        padding=padding,
    )
    return rallies, duration
