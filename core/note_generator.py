"""Note generator: merge ASR segments + vision descriptions into structured notes."""

import html
import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class NoteSection:
    title: str
    start: float
    end: float
    text: str
    image_descriptions: list[str] = field(default_factory=list)
    keyframe_indices: list[int] = field(default_factory=list)


@dataclass
class VideoNote:
    title: str
    duration: float
    sections: list[NoteSection]
    full_text: str


class NoteGenerator:
    def __init__(self, config: dict):
        self.cfg = config["output"]
        self.group_window = self.cfg["group_window"]

    def generate(
        self,
        asr_segments: list,
        vision_results: list,
        video_duration: float = 0.0,
        chapter_titles: list[str] | None = None,
    ) -> VideoNote:
        # 1. Group ASR segments by time window
        grouped = self._group_asr_by_window(asr_segments)

        # 2. Assign vision results to groups
        sections = self._merge(grouped, vision_results, chapter_titles)

        # 3. Build full text
        full_text = "\n".join(seg.text for seg in asr_segments if seg.text)

        return VideoNote(
            title="",
            duration=video_duration,
            sections=sections,
            full_text=full_text,
        )

    def _group_asr_by_window(self, segments: list) -> list[dict]:
        if not segments:
            return []

        window = self.group_window
        groups = []
        current_group = {
            "start": segments[0].start,
            "end": segments[0].end,
            "texts": [segments[0].text],
        }

        for seg in segments[1:]:
            if seg.start - current_group["start"] < window:
                current_group["end"] = seg.end
                current_group["texts"].append(seg.text)
            else:
                groups.append(current_group)
                current_group = {
                    "start": seg.start,
                    "end": seg.end,
                    "texts": [seg.text],
                }
        groups.append(current_group)
        return groups

    def _merge(
        self,
        groups: list[dict],
        vision_results: list,
        chapter_titles: list[str] | None,
    ) -> list[NoteSection]:
        sections = []
        for i, group in enumerate(groups):
            # Find matching vision results
            matching_vision = []
            matching_indices = []
            for vr in vision_results:
                if group["start"] <= vr.timestamp <= group["end"]:
                    matching_vision.append(vr.description)
                    matching_indices.append(int(vr.timestamp))

            title = ""
            if chapter_titles and i < len(chapter_titles):
                title = chapter_titles[i]
            else:
                title = self._auto_title(group["texts"])

            sections.append(NoteSection(
                title=title,
                start=group["start"],
                end=group["end"],
                text=" ".join(t for t in group["texts"] if t),
                image_descriptions=matching_vision,
                keyframe_indices=matching_indices,
            ))
        return sections

    @staticmethod
    def _auto_title(texts: list[str]) -> str:
        """Generate a simple title from the first few words of text."""
        combined = " ".join(t for t in texts if t)
        if not combined:
            return "无标题"
        # Take first 15 chars as title
        title = combined[:15]
        if len(combined) > 15:
            title += "..."
        return title

    # ── Export ──────────────────────────────────────────────────────

    def to_markdown(self, note: VideoNote, include_timestamps: bool = None) -> str:
        if include_timestamps is None:
            include_timestamps = self.cfg["include_timestamps"]

        lines = []
        if note.title:
            lines.append(f"# {note.title}\n")

        lines.append(f"> 视频时长: {self._fmt_time(note.duration)}\n")
        lines.append("---\n")

        for section in note.sections:
            header = section.title or "章节"
            if include_timestamps:
                header += f" `[{self._fmt_time(section.start)} - {self._fmt_time(section.end)}]`"
            lines.append(f"## {header}\n")

            # Image descriptions
            for desc in section.image_descriptions:
                lines.append(f"> 🖼️ {desc}\n")

            # Text content
            if section.text:
                lines.append(f"{section.text}\n")

            lines.append("")

        # Full transcript
        if note.full_text:
            lines.append("---\n")
            lines.append("## 完整转写文本\n")
            lines.append(note.full_text)

        return "\n".join(lines)

    def to_json_dict(self, note: VideoNote) -> dict:
        return {
            "title": note.title,
            "duration": note.duration,
            "sections": [
                {
                    "title": s.title,
                    "start": round(s.start, 2),
                    "end": round(s.end, 2),
                    "text": s.text,
                    "image_descriptions": s.image_descriptions,
                }
                for s in note.sections
            ],
            "full_text": note.full_text,
        }

    def to_html(self, note: VideoNote, video_filename: str = "") -> str:
        """Export as styled HTML with interactive timeline (Memo AI style)."""
        sections_html = ""
        for i, section in enumerate(note.sections):
            ts_start = self._fmt_time(section.start)
            ts_end = self._fmt_time(section.end)
            vis_blocks = "\n".join(
                f'<div class="vision-card"><span class="icon">🖼</span> {html.escape(desc)}</div>'
                for desc in section.image_descriptions
            )
            safe_title = html.escape(section.title)
            safe_text = html.escape(section.text)
            sections_html += f"""
            <div class="section" data-start="{section.start}" data-end="{section.end}">
                <div class="section-header">
                    <span class="section-num">{i + 1}</span>
                    <h2>{safe_title}</h2>
                    <span class="timestamp">⏱ {ts_start} - {ts_end}</span>
                </div>
                {vis_blocks}
                <p class="section-text">{safe_text}</p>
            </div>"""

        safe_note_title = html.escape(note.title)
        safe_full_text = html.escape(note.full_text)
        return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{safe_note_title} - 视频笔记</title>
<style>
:root{{--primary:#4a9eff;--bg:#fafbfc;--card:#fff;--border:#e8ecf0;--text:#1a1a2e;--muted:#6b7280}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:var(--bg);color:var(--text);line-height:1.7}}
.container{{max-width:900px;margin:0 auto;padding:24px}}
.header{{text-align:center;margin-bottom:32px}}
.header h1{{font-size:1.8em;color:var(--text)}}
.header .meta{{color:var(--muted);font-size:0.9em;margin-top:8px}}
.section{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px;margin:16px 0;
  cursor:pointer;transition:all .2s;border-left:4px solid var(--primary)}}
.section:hover{{box-shadow:0 4px 12px rgba(74,158,255,.15);transform:translateX(2px)}}
.section-header{{display:flex;align-items:center;gap:12px;margin-bottom:10px}}
.section-num{{background:var(--primary);color:#fff;border-radius:50%;width:28px;height:28px;display:flex;
  align-items:center;justify-content:center;font-size:.85em;font-weight:600;flex-shrink:0}}
.section-header h2{{font-size:1.1em;flex:1}}
.timestamp{{color:var(--primary);font-size:.85em;white-space:nowrap}}
.section-text{{color:#374151;line-height:1.8}}
.vision-card{{background:#f0f7ff;border-radius:8px;padding:10px 14px;margin:8px 0;font-size:.9em;color:#1e40af}}
.vision-card .icon{{margin-right:4px}}
.transcript{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px;margin-top:32px}}
.transcript h2{{margin-bottom:12px;font-size:1.2em;color:var(--text)}}
.transcript p{{white-space:pre-wrap;color:#4b5563}}
.divider{{border:none;border-top:1px solid var(--border);margin:24px 0}}
.footer{{text-align:center;color:var(--muted);font-size:.8em;margin-top:32px;padding:16px}}
</style></head><body>
<div class="container">
<div class="header">
<h1>🎬 {safe_note_title}</h1>
<div class="meta">时长 {self._fmt_time(note.duration)} · {len(note.sections)} 个章节 · 本地AI生成</div>
</div>
{sections_html}
<hr class="divider">
<div class="transcript"><h2>📝 完整转写文本</h2><p>{safe_full_text}</p></div>
<div class="footer">由 视频笔记助手 生成 · 端侧部署 · ASR + 多模态视觉</div>
</div></body></html>"""

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        if seconds <= 0:
            return "00:00"
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"
