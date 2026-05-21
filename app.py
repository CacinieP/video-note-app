"""Video Note App - Main entry point with Gradio UI."""

import json
import logging
import os
import threading
import time
from pathlib import Path

import gradio as gr
import yaml

from core.asr_engine import ASREngine
from core.note_generator import NoteGenerator
from core.video_processor import VideoProcessor
from core.vision_engine import VisionEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = str(SCRIPT_DIR / "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Pipeline ───────────────────────────────────────────────────────

class Pipeline:
    """Orchestrates the full video-to-note pipeline."""

    def __init__(self, config: dict):
        self.config = config
        self.video_proc = VideoProcessor(config)
        self.asr = ASREngine(config)
        self.vision = VisionEngine(config)
        self.note_gen = NoteGenerator(config)
        self._progress = 0.0
        self._status = "就绪"
        self._stop = False

    @property
    def progress(self):
        return self._progress

    @property
    def status(self):
        return self._status

    def reset(self):
        self._progress = 0.0
        self._status = "就绪"
        self._stop = False

    def stop(self):
        self._stop = True
        self._status = "已停止"

    def run(self, video_path: str, title: str = "", enable_vision: bool = True,
            enable_chapters: bool = True, progress_cb=None) -> dict:
        """Run full pipeline, return result dict. progress_cb(fraction, status) called at each step."""
        self.reset()
        result = {
            "video_path": video_path,
            "title": title or Path(video_path).stem,
            "markdown": "",
            "html": "",
            "json": {},
            "keyframe_images": [],
            "error": None,
        }

        def _progress(frac, status):
            self._progress = frac
            self._status = status
            if progress_cb:
                progress_cb(frac, status)

        try:
            # Step 1: Extract audio
            _progress(0.05, "提取音频中...")
            audio_path = self.video_proc.extract_audio(video_path)
            log.info(f"Audio extracted: {audio_path}")
            if self._stop:
                raise InterruptedError("用户停止")

            # Step 2: ASR
            _progress(0.15, "语音识别中...")
            asr_segments = self.asr.transcribe(audio_path)
            log.info(f"ASR segments: {len(asr_segments)}")
            if self._stop:
                raise InterruptedError("用户停止")

            # Clean up temp audio
            try:
                os.remove(audio_path)
            except OSError:
                pass

            # Step 3: Extract keyframes
            _progress(0.35, "提取关键帧中...")
            keyframes = self.video_proc.extract_keyframes(video_path)
            log.info(f"Keyframes: {len(keyframes)}")
            if self._stop:
                raise InterruptedError("用户停止")

            # Save keyframe images
            kf_dir = str(OUTPUT_DIR / ".keyframes")
            kf_images = []
            for kf in keyframes:
                path = self.video_proc.save_keyframe_image(kf, kf_dir)
                kf_images.append(path)

            result["keyframe_images"] = kf_images

            # Step 4: Vision analysis (optional)
            vision_results = []
            chapter_titles = []
            if enable_vision and keyframes:
                _progress(0.5, "视觉分析中...")
                vision_results = self.vision.analyze_keyframes(keyframes)
                log.info(f"Vision results: {len(vision_results)}")
                if self._stop:
                    raise InterruptedError("用户停止")

            # Step 4b: Chapter titles (optional, requires vision for context)
            if enable_chapters and keyframes:
                _progress(0.8, "生成章节标题...")
                for kf in keyframes:
                    if self._stop:
                        break
                    nearby_text = self._get_nearby_text(kf.timestamp, asr_segments)
                    if nearby_text:
                        try:
                            t = self.vision.generate_chapter_title(kf.frame, nearby_text)
                            chapter_titles.append(t)
                        except Exception:
                            chapter_titles.append("")
                    else:
                        chapter_titles.append("")

            # Step 5: Generate notes
            _progress(0.9, "生成笔记中...")
            duration = self.video_proc.get_video_duration(video_path)
            note = self.note_gen.generate(
                asr_segments=asr_segments,
                vision_results=vision_results,
                video_duration=duration,
                chapter_titles=chapter_titles if any(chapter_titles) else None,
            )
            note.title = result["title"]

            result["markdown"] = self.note_gen.to_markdown(note)
            result["html"] = self.note_gen.to_html(note)
            result["json"] = self.note_gen.to_json_dict(note)

            _progress(1.0, "完成！")

        except InterruptedError:
            result["error"] = "已停止"
            self._status = "已停止"
        except Exception as e:
            log.exception("Pipeline failed")
            result["error"] = str(e)
            self._status = f"错误: {e}"

        return result

    @staticmethod
    def _get_nearby_text(timestamp: float, segments: list, window: float = 15.0) -> str:
        texts = [
            s.text for s in segments
            if abs(s.start - timestamp) < window or abs(s.end - timestamp) < window
        ]
        return " ".join(texts)


# ── Gradio UI ──────────────────────────────────────────────────────

def build_ui(config: dict):
    pipeline = Pipeline(config)

    # Check system status
    ollama_ok = pipeline.vision.check_available()
    ollama_models = pipeline.vision.list_models()

    with gr.Blocks(title="视频笔记助手") as app:

        gr.Markdown("# 🎬 视频笔记助手")
        gr.Markdown("本地部署的视频转笔记工具 | ASR + 多模态视觉分析")

        # ── Status bar ─────────────────────────────────────────────
        with gr.Row():
            status_text = gr.Textbox(
                label="系统状态",
                value=f"ASR: {config['asr']['backend']} | Vision: {'✅ Ollama' if ollama_ok else '❌ Ollama 未连接'} ({config['vision']['ollama']['model']})",
                interactive=False,
                scale=3,
            )
            models_text = gr.Textbox(
                label="可用模型",
                value=", ".join(ollama_models) if ollama_models else "未检测到模型",
                interactive=False,
                scale=2,
            )

        # ── Main layout ────────────────────────────────────────────
        with gr.Row():
            # Left: Video input
            with gr.Column(scale=1):
                gr.Markdown("### 视频输入")
                video_input = gr.Video(
                    label="上传视频",
                    elem_id="video_player",
                )
                with gr.Row():
                    title_input = gr.Textbox(
                        label="笔记标题",
                        placeholder="留空则使用文件名",
                        scale=2,
                    )
                with gr.Row():
                    enable_vision = gr.Checkbox(
                        value=True,
                        label="启用视觉分析（需要 Ollama）",
                    )
                    enable_chapters = gr.Checkbox(
                        value=True,
                        label="自动章节标题",
                    )

                # Config
                with gr.Accordion("高级设置", open=False):
                    asr_backend = gr.Dropdown(
                        choices=["funasr", "faster_whisper"],
                        value=config["asr"]["backend"],
                        label="ASR 引擎",
                    )
                    vision_model = gr.Dropdown(
                        choices=ollama_models or ["minicpm-v", "qwen2-vl", "llava"],
                        value=config["vision"]["ollama"]["model"],
                        allow_custom_value=True,
                        label="视觉模型",
                    )
                    kf_method = gr.Dropdown(
                        choices=["scene", "interval"],
                        value=config["video"]["keyframe"]["method"],
                        label="关键帧提取方式",
                    )
                    kf_threshold = gr.Slider(
                        minimum=0.1, maximum=0.8, value=config["video"]["keyframe"]["scene_threshold"],
                        step=0.05, label="场景变化阈值（越低越敏感）",
                    )
                    group_window = gr.Slider(
                        minimum=10, maximum=120, value=config["output"]["group_window"],
                        step=5, label="笔记分段窗口（秒）",
                    )

                btn_run = gr.Button("🚀 开始生成笔记", variant="primary", size="lg")
                btn_stop = gr.Button("⏹ 停止", variant="stop")

            # Right: Notes output
            with gr.Column(scale=1):
                gr.Markdown("### 笔记输出")
                with gr.Tabs():
                    with gr.Tab("笔记"):
                        note_output = gr.Markdown(
                            value="*上传视频后点击「开始生成笔记」*",
                            elem_classes=["note-area"],
                        )
                    with gr.Tab("完整转写"):
                        transcript_output = gr.Textbox(
                            label="转写文本",
                            lines=20,
                            interactive=False,
                        )
                    with gr.Tab("关键帧"):
                        gallery_output = gr.Gallery(
                            label="关键帧截图",
                            columns=3,
                            height=400,
                        )
                    with gr.Tab("JSON"):
                        json_output = gr.Code(
                            label="结构化数据",
                            language="json",
                            lines=20,
                        )

                # Export buttons
                with gr.Row():
                    btn_export_md = gr.DownloadButton("导出 Markdown", variant="secondary")
                    btn_export_html = gr.DownloadButton("导出 HTML", variant="secondary")
                    btn_export_json = gr.DownloadButton("导出 JSON", variant="secondary")

                # Search
                with gr.Row():
                    search_input = gr.Textbox(
                        label="搜索笔记",
                        placeholder="输入关键词...",
                        scale=3,
                    )
                    btn_search = gr.Button("搜索", scale=1)

                search_output = gr.Markdown(visible=False)

        # ── Progress ───────────────────────────────────────────────
        progress_bar = gr.Slider(
            minimum=0, maximum=1, value=0,
            label="进度", interactive=False,
        )
        progress_label = gr.Markdown("")

        # ── State ──────────────────────────────────────────────────
        last_result = gr.State({})
        active_pipeline = gr.State(None)

        # ── Event handlers ─────────────────────────────────────────

        def run_pipeline(video, title, vision_on, chapters_on,
                         asr_be, v_model, kf_met, kf_thresh, gw,
                         _active_pipe, progress=gr.Progress()):
            if not video:
                gr.Warning("请先上传视频")
                yield (
                    "*请先上传视频*", "", [], "{}", {},
                    None, 0, "请上传视频",
                )
                return

            # Update config dynamically (copy to avoid cross-request contamination)
            run_config = json.loads(json.dumps(config))
            run_config["asr"]["backend"] = asr_be
            run_config["vision"]["ollama"]["model"] = v_model
            run_config["video"]["keyframe"]["method"] = kf_met
            run_config["video"]["keyframe"]["scene_threshold"] = kf_thresh
            run_config["output"]["group_window"] = gw

            def on_progress(frac, status):
                progress(frac, desc=status)

            p = Pipeline(run_config)
            yield (
                "*处理中...*", "", [], "{}", {},
                p, 0, "处理中...",
            )

            result = p.run(video, title, enable_vision=vision_on, enable_chapters=chapters_on, progress_cb=on_progress)

            md = result["markdown"] if not result["error"] else f"**错误**: {result['error']}"
            transcript = result["json"].get("full_text", "") if result["json"] else ""
            kf_imgs = result.get("keyframe_images", [])
            js = json.dumps(result["json"], ensure_ascii=False, indent=2) if result["json"] else "{}"

            yield (
                md, transcript, kf_imgs, js, result,
                None, p.progress, p.status,
            )

        def export_markdown(result):
            if not result or not result.get("markdown"):
                return None
            safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in result.get("title", "notes"))
            path = str(OUTPUT_DIR / f"{safe_title}.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(result["markdown"])
            return path

        def export_html(result):
            if not result or not result.get("html"):
                return None
            safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in result.get("title", "notes"))
            path = str(OUTPUT_DIR / f"{safe_title}.html")
            with open(path, "w", encoding="utf-8") as f:
                f.write(result["html"])
            return path

        def export_json(result):
            if not result or not result.get("json"):
                return None
            safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in result.get("title", "notes"))
            path = str(OUTPUT_DIR / f"{safe_title}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result["json"], f, ensure_ascii=False, indent=2)
            return path

        def search_notes(keyword, result):
            if not keyword or not result or not result.get("json"):
                return "*请输入关键词*"
            sections = result["json"].get("sections", [])
            matches = []
            for s in sections:
                if keyword in s.get("text", "") or keyword in s.get("title", ""):
                    matches.append(
                        f"- **{s['title']}** [{NoteGenerator._fmt_time(s['start'])} - {NoteGenerator._fmt_time(s['end'])}]\n"
                        f"  {s['text'][:200]}"
                    )
            if matches:
                return "\n".join(matches)
            return f"*未找到「{keyword}」相关内容*"

        def stop_pipeline(_active_pipe):
            if _active_pipe is not None:
                _active_pipe.stop()
                return _active_pipe.progress, "已停止"
            return 0, "就绪"

        # Wire up
        btn_run.click(
            fn=run_pipeline,
            inputs=[
                video_input, title_input, enable_vision, enable_chapters,
                asr_backend, vision_model, kf_method, kf_threshold, group_window,
                active_pipeline,
            ],
            outputs=[
                note_output, transcript_output, gallery_output, json_output,
                last_result, active_pipeline, progress_bar, progress_label,
            ],
        )

        btn_stop.click(
            fn=stop_pipeline,
            inputs=[active_pipeline],
            outputs=[progress_bar, progress_label],
        )

        btn_export_md.click(fn=export_markdown, inputs=[last_result], outputs=[btn_export_md])
        btn_export_html.click(fn=export_html, inputs=[last_result], outputs=[btn_export_html])
        btn_export_json.click(fn=export_json, inputs=[last_result], outputs=[btn_export_json])

        btn_search.click(
            fn=search_notes,
            inputs=[search_input, last_result],
            outputs=[search_output],
        )

    return app


def main():
    config = load_config()
    app = build_ui(config)
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(),
        css="""
        .note-area {max-height: 600px; overflow-y: auto;}
        .status-bar {padding: 10px; background: #f0f7ff; border-radius: 8px; margin: 5px 0;}
        #video_player {max-height: 450px;}
        """,
    )


if __name__ == "__main__":
    main()
