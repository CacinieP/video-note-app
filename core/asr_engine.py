"""ASR engine with FunASR and faster-whisper backends."""

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class ASRSegment:
    start: float   # seconds
    end: float     # seconds
    text: str


class ASREngine:
    def __init__(self, config: dict):
        self.cfg = config["asr"]
        self.backend = self.cfg["backend"]

    def transcribe(self, audio_path: str) -> list[ASRSegment]:
        if self.backend == "funasr":
            return self._transcribe_funasr(audio_path)
        elif self.backend == "faster_whisper":
            return self._transcribe_faster_whisper(audio_path)
        else:
            raise ValueError(f"Unknown ASR backend: {self.backend}")

    # ── FunASR (SenseVoice / Paraformer) - optional ────────────────

    def _transcribe_funasr(self, audio_path: str) -> list[ASRSegment]:
        try:
            from funasr import AutoModel
        except ImportError:
            raise ImportError(
                "FunASR 未安装。请运行: pip install funasr torch torchaudio modelscope\n"
                "或切换到 faster_whisper 后端: config.yaml -> asr.backend: 'faster_whisper'"
            )

        model_name = self.cfg["funasr"]["model"]
        device = self.cfg["funasr"]["device"]

        model = AutoModel(model=model_name, device=device)

        result = model.generate(
            input=audio_path,
            batch_size_s=300,
            hotword="",
        )

        segments = []
        for res in result:
            text = res.get("text", "")
            timestamp = res.get("timestamp", [])

            if timestamp and len(timestamp) >= 2:
                if isinstance(timestamp[0], list):
                    for ts in timestamp:
                        start_s = ts[0] / 1000.0
                        end_s = ts[1] / 1000.0
                        segments.append(ASRSegment(start=start_s, end=end_s, text=""))
                    if segments and text:
                        full_text = self._clean_funasr_text(text)
                        self._distribute_text(segments, full_text)
                else:
                    start_s = timestamp[0] / 1000.0
                    end_s = timestamp[1] / 1000.0
                    segments.append(ASRSegment(
                        start=start_s, end=end_s,
                        text=self._clean_funasr_text(text),
                    ))
            else:
                segments.append(ASRSegment(
                    start=0.0, end=0.0,
                    text=self._clean_funasr_text(text),
                ))

        return segments

    @staticmethod
    def _clean_funasr_text(text: str) -> str:
        import re
        text = re.sub(r'<\|[^|]*\|>', '', text)
        return text.strip()

    @staticmethod
    def _distribute_text(segments: list[ASRSegment], full_text: str):
        if not segments or not full_text:
            return
        total_dur = sum(s.end - s.start for s in segments) or 1.0
        pos = 0
        for seg in segments:
            ratio = (seg.end - seg.start) / total_dur
            char_count = max(1, int(len(full_text) * ratio))
            seg.text = full_text[pos:pos + char_count]
            pos += char_count
        if pos < len(full_text):
            segments[-1].text += full_text[pos:]

    # ── faster-whisper ─────────────────────────────────────────────

    def _transcribe_faster_whisper(self, audio_path: str) -> list[ASRSegment]:
        from faster_whisper import WhisperModel

        model_size = self.cfg["faster_whisper"]["model_size"]
        device = self.cfg["faster_whisper"]["device"]
        compute_type = self.cfg["faster_whisper"]["compute_type"]

        log.info(f"Loading faster-whisper model: {model_size} ({device}/{compute_type})")
        model = WhisperModel(model_size, device=device, compute_type=compute_type)

        log.info(f"Transcribing: {audio_path}")
        segments_iter, info = model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=300,
                speech_pad_ms=200,
            ),
        )

        log.info(f"Detected language: {info.language} (prob: {info.language_probability:.2f})")

        segments = []
        for seg in segments_iter:
            segments.append(ASRSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text.strip(),
            ))

        log.info(f"Transcription complete: {len(segments)} segments")
        return segments
