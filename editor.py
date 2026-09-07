from __future__ import annotations

import platform
import subprocess
from pathlib import Path


def _video_codec() -> tuple[str, list[str]]:
    """Prefer hardware encoding on macOS, fall back to libx264 elsewhere."""
    if platform.system() == "Darwin":
        return "h264_videotoolbox", ["-b:v", "8M"]
    return "libx264", ["-crf", "22", "-preset", "medium"]


def cut_video(
    input_path: Path,
    output_path: Path,
    segments: list[tuple[float, float]],
) -> None:
    """Re-encode input into output, keeping only the given time ranges."""
    if not segments:
        raise ValueError("No segments to keep")

    v_expr = "+".join(f"between(t,{s:.3f},{e:.3f})" for s, e in segments)
    a_expr = v_expr

    codec, codec_args = _video_codec()

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", f"select='{v_expr}',setpts=N/FRAME_RATE/TB",
        "-af", f"aselect='{a_expr}',asetpts=N/SR/TB",
        "-c:v", codec, *codec_args,
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(output_path),
    ]

    subprocess.run(cmd, check=True)
