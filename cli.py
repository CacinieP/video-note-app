"""CLI mode for batch processing videos without the UI."""

import argparse
import json
import sys
from pathlib import Path

from app import load_config, Pipeline


def main():
    parser = argparse.ArgumentParser(description="视频笔记助手 CLI")
    parser.add_argument("video", nargs="+", help="视频文件路径")
    parser.add_argument("--title", "-t", default="", help="笔记标题")
    parser.add_argument("--output", "-o", default="", help="输出目录 (默认视频同目录)")
    parser.add_argument("--format", "-f", choices=["markdown", "html", "json", "all"], default="all")
    parser.add_argument("--no-vision", action="store_true", help="禁用视觉分析")
    parser.add_argument("--asr", choices=["funasr", "faster_whisper"], help="ASR 后端")
    parser.add_argument("--config", "-c", default="", help="配置文件路径")
    args = parser.parse_args()

    config = load_config(args.config or None)
    if args.asr:
        config["asr"]["backend"] = args.asr

    pipeline = Pipeline(config)

    for video_path in args.video:
        video_path = str(Path(video_path).resolve())
        if not Path(video_path).exists():
            print(f"❌ 文件不存在: {video_path}")
            continue

        print(f"\n{'=' * 50}")
        print(f"处理: {video_path}")
        print(f"{'=' * 50}")

        result = pipeline.run(video_path, title=args.title, enable_vision=not args.no_vision)

        if result["error"]:
            print(f"❌ 处理失败: {result['error']}")
            continue

        # Determine output directory
        out_dir = args.output or str(Path(video_path).parent)
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        base_name = Path(video_path).stem

        if args.format in ("markdown", "all"):
            path = str(Path(out_dir) / f"{base_name}_notes.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(result["markdown"])
            print(f"  ✅ Markdown: {path}")

        if args.format in ("html", "all"):
            path = str(Path(out_dir) / f"{base_name}_notes.html")
            with open(path, "w", encoding="utf-8") as f:
                f.write(result["html"])
            print(f"  ✅ HTML: {path}")

        if args.format in ("json", "all"):
            path = str(Path(out_dir) / f"{base_name}_notes.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result["json"], f, ensure_ascii=False, indent=2)
            print(f"  ✅ JSON: {path}")

        print(f"  📝 {len(result['json'].get('sections', []))} 个章节, "
              f"转写文本 {len(result['json'].get('full_text', ''))} 字")

    print("\n✅ 处理完成")


if __name__ == "__main__":
    main()
