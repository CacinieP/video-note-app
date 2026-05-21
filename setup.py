"""Setup helper: check dependencies and download models."""

import subprocess
import sys


def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        print("✅ ffmpeg 已安装")
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("❌ ffmpeg 未安装。请安装: https://ffmpeg.org/download.html")
        return False


def check_ollama():
    import requests
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            print(f"✅ Ollama 运行中, 已有模型: {models}")
            return True
        print("❌ Ollama 返回异常")
        return False
    except Exception:
        print("❌ Ollama 未运行。请启动 Ollama: https://ollama.ai")
        return False


def check_python_deps():
    missing = []
    for pkg in ["gradio", "opencv-python-headless", "numpy", "Pillow", "pyyaml", "requests"]:
        try:
            __import__(pkg.replace("-", "_").split("[")[0])
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"⚠️ 缺少依赖: {missing}")
        print(f"  运行: pip install {' '.join(missing)}")
        return False
    print("✅ Python 依赖已安装")
    return True


def check_asr():
    backend = input("选择 ASR 后端 (1=FunASR/SenseVoice, 2=faster-whisper): ").strip()
    if backend == "1":
        try:
            from funasr import AutoModel
            print("✅ FunASR 已安装")
            return "funasr"
        except ImportError:
            print("❌ FunASR 未安装。运行: pip install funasr torch torchaudio modelscope")
            return None
    elif backend == "2":
        try:
            from faster_whisper import WhisperModel
            print("✅ faster-whisper 已安装")
            return "faster_whisper"
        except ImportError:
            print("❌ faster-whisper 未安装。运行: pip install faster-whisper")
            return None
    return None


def pull_ollama_model():
    model = input("输入要拉取的 Ollama 模型名称 (如 minicpm-v, qwen2-vl): ").strip()
    if model:
        print(f"正在拉取 {model}...")
        subprocess.run(["ollama", "pull", model], check=False)


def main():
    print("=" * 50)
    print("  视频笔记助手 - 环境检查")
    print("=" * 50)
    print()

    checks = [
        ("ffmpeg", check_ffmpeg),
        ("Python 依赖", check_python_deps),
        ("Ollama", check_ollama),
    ]

    results = {}
    for name, fn in checks:
        print(f"\n检查 {name}...")
        results[name] = fn()

    print("\n" + "=" * 50)
    if all(results.values()):
        print("🎉 所有环境检查通过！运行 python app.py 启动应用")
    else:
        print("⚠️ 部分检查未通过，请按提示安装缺失依赖")

    print()


if __name__ == "__main__":
    main()
