"""Vision engine using Ollama for local multimodal inference."""

import base64
import io
import logging
from dataclasses import dataclass

import cv2
import numpy as np
import requests

log = logging.getLogger(__name__)


@dataclass
class VisionResult:
    timestamp: float
    description: str
    confidence: float = 1.0


class VisionEngine:
    def __init__(self, config: dict):
        self.cfg = config["vision"]["ollama"]
        self.base_url = self.cfg["base_url"].rstrip("/")
        self.model = self.cfg["model"]
        self.timeout = self.cfg["timeout"]

    def _encode_image(self, frame: np.ndarray) -> str:
        """Encode BGR frame to base64 JPEG."""
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode("utf-8")

    def _encode_image_path(self, image_path: str) -> str:
        """Encode image file to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def analyze_frame(self, frame: np.ndarray, prompt: str) -> str:
        """Analyze a single frame with the vision model."""
        img_b64 = self._encode_image(frame)
        return self._call_ollama(prompt, img_b64)

    def analyze_image_path(self, image_path: str, prompt: str) -> str:
        """Analyze an image file with the vision model."""
        img_b64 = self._encode_image_path(image_path)
        return self._call_ollama(prompt, img_b64)

    def analyze_keyframes(
        self,
        keyframes: list,
        prompt: str = None,
    ) -> list[VisionResult]:
        """Analyze multiple keyframes, return descriptions."""
        if prompt is None:
            prompt = (
                "请详细描述这个视频画面中的内容。包括：\n"
                "1. 画面中出现的关键元素（人物、物体、文字、图表等）\n"
                "2. 画面传达的主要信息\n"
                "3. 如果有文字，请完整抄录\n"
                "请用简洁的中文回答。"
            )

        results = []
        for kf in keyframes:
            try:
                desc = self.analyze_frame(kf.frame, prompt)
                results.append(VisionResult(
                    timestamp=kf.timestamp,
                    description=desc.strip(),
                ))
            except Exception as e:
                log.warning(f"Failed to analyze keyframe at {kf.timestamp:.1f}s: {e}")
                results.append(VisionResult(
                    timestamp=kf.timestamp,
                    description=f"[分析失败: {e}]",
                    confidence=0.0,
                ))
        return results

    def generate_chapter_title(self, frame: np.ndarray, context_text: str) -> str:
        """Generate a chapter title based on a keyframe and surrounding text."""
        prompt = (
            f"根据以下视频画面和对应的语音文字，生成一个简短的章节标题（10个字以内）：\n"
            f"语音文字：{context_text[:200]}\n"
            f"请只输出标题，不要其他内容。"
        )
        return self.analyze_frame(frame, prompt).strip()

    def _call_ollama(self, prompt: str, image_b64: str) -> str:
        """Call Ollama API for multimodal inference."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 512,
            },
        }
        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json().get("response", "")

    def check_available(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                return any(self.model in m.get("name", "") for m in models)
        except Exception:
            pass
        return False

    def list_models(self) -> list[str]:
        """List available multimodal models in Ollama."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                return [m["name"] for m in models]
        except Exception:
            pass
        return []
