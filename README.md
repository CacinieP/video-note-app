# 🎬 视频笔记助手 (video-note-app)

本地部署的视频转笔记工具 —— ASR 语音识别 + 多模态视觉分析，自动生成结构化笔记。

## 功能特点

- **双 ASR 后端**：faster-whisper（默认）或 FunASR/SenseVoice
- **多模态视觉分析**：通过 Ollama（minicpm-v / qwen2-vl / llava）分析关键帧
- **智能关键帧提取**：场景变化检测或固定间隔采样
- **结构化笔记输出**：Markdown / HTML / JSON 三种格式
- **双运行模式**：Gradio Web UI + CLI 批处理
- **完全本地运行**：不依赖云端 API，数据不出本机

## 架构

```
视频 → 音频提取 → ASR 转写 → 关键帧提取 → 视觉分析 → 章节生成 → 结构化笔记
```

```
video-note-app/
├── app.py                 # Gradio UI + Pipeline 主入口
├── cli.py                 # CLI 批处理模式
├── config.yaml            # 配置文件
├── check_env.py           # 环境依赖检查工具
├── core/
│   ├── asr_engine.py      # ASR 引擎 (faster-whisper / FunASR)
│   ├── note_generator.py  # 笔记生成 + Markdown/HTML/JSON 导出
│   ├── video_processor.py # 音频提取 + 关键帧提取
│   └── vision_engine.py   # Ollama 多模态视觉分析
├── requirements.txt
└── .github/workflows/ci.yml
```

## 安装

### 1. 系统依赖

- **Python 3.10+**
- **ffmpeg** —— 音频提取必需 ([下载](https://ffmpeg.org/download.html))
- **Ollama** —— 视觉分析必需 ([安装](https://ollama.ai))，如不需要视觉分析可跳过

### 2. Python 依赖

```bash
pip install -r requirements.txt
```

如需使用 FunASR 后端：

```bash
pip install funasr torch torchaudio modelscope
```

### 3. 拉取 Ollama 视觉模型

```bash
ollama pull minicpm-v    # 推荐，轻量高效
# 或
ollama pull qwen2-vl
ollama pull llava
```

### 4. 环境检查

```bash
python check_env.py
```

## 使用

### Web UI 模式

```bash
python app.py
```

浏览器自动打开 `http://localhost:7860`，上传视频即可生成笔记。

### CLI 批处理模式

```bash
# 处理单个视频
python cli.py video.mp4

# 批量处理，指定标题和输出格式
python cli.py video1.mp4 video2.mp4 -t "我的笔记" -f markdown -o ./notes/

# 禁用视觉分析（加速处理）
python cli.py video.mp4 --no-vision

# 使用 FunASR 后端
python cli.py video.mp4 --asr funasr
```

### CLI 参数

| 参数 | 说明 |
|---|---|
| `--title, -t` | 笔记标题（留空则使用文件名） |
| `--output, -o` | 输出目录 |
| `--format, -f` | 输出格式: markdown / html / json / all |
| `--no-vision` | 禁用视觉分析 |
| `--asr` | ASR 后端: funasr / faster_whisper |
| `--config, -c` | 配置文件路径 |

## 配置

编辑 `config.yaml` 自定义行为：

```yaml
asr:
  backend: "faster_whisper"    # 或 "funasr"
  faster_whisper:
    model_size: "medium"       # tiny/base/small/medium/large-v3
    device: "cpu"              # cpu 或 cuda
    compute_type: "int8"       # float16 或 int8

vision:
  ollama:
    model: "minicpm-v"         # 视觉模型名称
    timeout: 120

video:
  keyframe:
    method: "scene"            # scene（场景检测）或 interval（固定间隔）
    scene_threshold: 0.3       # 场景变化阈值（越低越敏感）
    max_keyframes: 50

output:
  group_window: 30             # 笔记分段窗口（秒）
  include_timestamps: true
```

## 输出格式

### Markdown 示例

```markdown
# 我的视频笔记

> 视频时长: 05:30

---

## 开场介绍 `[00:00 - 00:30]`

> 🖼️ 画面中展示了一个标题幻灯片...

大家好，今天我们来讨论...
```

### HTML

生成带时间轴交互的 HTML 页面，Memo AI 风格，可直接浏览器查看。

### JSON

结构化数据，便于二次开发集成。

## 许可证

MIT License — 详见 [LICENSE](LICENSE)