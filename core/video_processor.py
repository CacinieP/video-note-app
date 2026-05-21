"""Video processing: audio extraction and keyframe extraction."""

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Keyframe:
    timestamp: float  # seconds
    frame: np.ndarray  # BGR image
    index: int


class VideoProcessor:
    def __init__(self, config: dict):
        self.keyframe_cfg = config["video"]["keyframe"]
        self.audio_cfg = config["video"]["audio"]

    # ── Audio ──────────────────────────────────────────────────────

    def extract_audio(self, video_path: str) -> str:
        """Extract audio from video, return path to wav file."""
        out_path = os.path.splitext(video_path)[0] + "_audio.wav"
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn",
            "-ar", str(self.audio_cfg["sample_rate"]),
            "-ac", "1",
            "-f", "wav",
            out_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return out_path

    # ── Keyframes ──────────────────────────────────────────────────

    def extract_keyframes(self, video_path: str) -> list[Keyframe]:
        method = self.keyframe_cfg["method"]
        if method == "scene":
            return self._extract_scene_keyframes(video_path)
        return self._extract_interval_keyframes(video_path)

    def _extract_interval_keyframes(self, video_path: str) -> list[Keyframe]:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        interval = self.keyframe_cfg["interval_sec"]
        frame_interval = int(fps * interval)
        max_kf = self.keyframe_cfg["max_keyframes"]

        keyframes = []
        idx = 0
        while len(keyframes) < max_kf:
            pos = idx * frame_interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if not ret:
                break
            keyframes.append(Keyframe(
                timestamp=pos / fps,
                frame=frame,
                index=idx,
            ))
            idx += 1
        cap.release()
        return keyframes

    def _extract_scene_keyframes(self, video_path: str) -> list[Keyframe]:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        threshold = self.keyframe_cfg["scene_threshold"]
        max_kf = self.keyframe_cfg["max_keyframes"]

        keyframes = []
        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            return keyframes

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        keyframes.append(Keyframe(timestamp=0.0, frame=prev_frame, index=0))

        frame_idx = 1
        while len(keyframes) < max_kf:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_gray, gray)
            score = np.mean(diff) / 255.0
            if score > threshold:
                keyframes.append(Keyframe(
                    timestamp=frame_idx / fps,
                    frame=frame,
                    index=len(keyframes),
                ))
                prev_gray = gray
            frame_idx += 1

        cap.release()
        return keyframes

    # ── Utility ────────────────────────────────────────────────────

    @staticmethod
    def save_keyframe_image(keyframe: Keyframe, output_dir: str) -> str:
        """Save a keyframe as JPEG, return path."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"kf_{keyframe.index:04d}_{keyframe.timestamp:.1f}s.jpg")
        cv2.imwrite(path, keyframe.frame)
        return path

    @staticmethod
    def save_keyframes_temp(keyframes: list[Keyframe]) -> list[str]:
        """Save keyframes to temp dir, return list of paths."""
        tmp_dir = tempfile.mkdtemp(prefix="vn_kf_")
        return [
            VideoProcessor.save_keyframe_image(kf, tmp_dir)
            for kf in keyframes
        ]

    @staticmethod
    def get_video_duration(video_path: str) -> float:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return frames / fps if fps > 0 else 0.0
