# 广州鲁迅纪念馆 · 鲁迅数字人

面向广州鲁迅纪念馆的**具身智能**项目——机器人的「AI 大脑」。游客可跟「鲁迅」语音对话，也可切换讲解员模式了解场馆。

本仓库负责软件核心链路：**意图理解 → 查询改写 → 知识检索 → Prompt 拼接 → LLM 生成**，语音 I/O（ASR/TTS）由机器人硬件层处理。

## 项目阶段

1. **知识库构建** — 鲁迅作品/生平/场馆/语录/人设 5 域语料向量化入库
2. **对话管线** — RAG 全链路 + 讲解员 / 鲁迅数字人双模式自动切换
3. **鲁棒性** — 4 层越界防护 + 幻觉检测 + 一致性校验 + 质量守卫
4. **API 化部署** — FastAPI 接口服务，机器人「ASR → `/chat` → TTS」直接对接

## 技术栈

| 层 | 选型 |
|----|------|
| LLM | DeepSeek 云端 API（`deepseek-v4-flash`，thinking=disabled）|
| Embedding | `BAAI/bge-small-zh-v1.5`（~91MB，本地）|
| 向量库 | ChromaDB（本地持久化）|
| 检索 | sentence-transformers + 多子查询合并去重 |
| 服务 | FastAPI + uvicorn|
| 语言 | Python 3.12 |

## 目录结构

```
scenic-ai-guide/
├── scripts/
│   ├── api_server.py          # FastAPI 服务（机器人对接入口）
│   ├── rag_pipeline.py        # RAG 全链路 + 质量守卫
│   ├── ingest.py              # 数据入库
│   ├── pack_offline.py        # 三件套离线资源打包
│   ├── start.bat              # Windows 一键启动
│   └── bge-small-zh-v1.5/     # BGE 模型（gitignored，来自资源包）
├── data/processed/            # 5 域知识库 JSON（gitignored）
├── chroma_db/                 # 向量库（gitignored）
├── docs/                      # 部署指南 / 接口协议 / 各阶段方案
├── prompts/                   # 双模式 System Prompt
├── requirements.txt           # 锁定版本依赖
└── .env.example               # 环境变量模板
```

## 快速开始

> 🚀 **两条启动入口（最快上手）**
> - **看聊天界面**（评审 / 演示）→ `python scripts/gradio_app.py` → 浏览器打开 `http://139.9.103.86:7860`
> - **对接机器人**（FastAPI）→ Windows 一键 `scripts\start.bat`，或 `python scripts/api_server.py` → `http://localhost:8000`
>
> 下面 0~3 步是两者共同的前置准备。

### 0. 前置

- Python 3.12（实测 3.12.1）
- git

### 1. 获取代码 + 三件套资源

```bash
git clone https://github.com/zhangjiaxin13553-source/scenic-ai-guide.git
cd scenic-ai-guide
```

> ⚠️ **代码仓库不包含三件套离线资源**（BGE 模型 / `chroma_db/` / `data/processed/`，均已 gitignore）。
> 换机部署前，先从团队分发处取得 `releases/offline_resources.zip`，解压后按包内 `README.txt` 放置：
>
> | 包内路径 | 放置到 |
> |----------|--------|
> | `bge-small-zh-v1.5/` | `scripts/bge-small-zh-v1.5/` |
> | `chroma_db/` | 项目根目录 |
> | `data/processed/` | 项目根目录 |
>
> 若某台机器已具备三件套，可自行生成资源包：`python scripts/pack_offline.py`

### 2. 安装依赖

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 配置 API Key

```bash
cp .env.example .env            # Windows: copy .env.example .env
# 编辑 .env，填入 LLM_API_KEY
```

### 4. 启动（两条入口，按需选择）

**入口 A · 看效果（Gradio 聊天界面，推荐评审 / 演示）**

```bash
python scripts/gradio_app.py    # 浏览器打开 http://127.0.0.1:7860
```

**入口 B · 机器人对接（FastAPI 接口）**

```bash
# Windows 一键启动（校验环境 + 装依赖 + 校验资源 + 启动 FastAPI）
scripts\start.bat

# 或手动启动
python scripts/api_server.py    
```

### 5. 验证

**入口 A（Gradio）**：浏览器打开 `http://127.0.0.1:7860`，直接提问即可（右侧调试面板会显示意图 / 检索 / 守卫状态）。

**入口 B（FastAPI）**：

```bash
curl http://localhost:8000/health
```

返回 `"status":"ok"` 即就绪。**首次加载约 1 分钟**（实测 77s，BGE 模型 + ChromaDB），建议机器人上电后先请求一次 `/health` 预热。

## 接口

机器人对接契约详见 [docs/deployment-guide.md](docs/deployment-guide.md)。服务启动后访问 `http://localhost:8000/docs` 有交互式接口文档。

## 在线体验 Demo

`space/` / `scripts/gradio_app.py` 是可独立部署的在线体验 Demo（Gradio 界面，讲解员 / 鲁迅双模式 + 调试面板）。
三件套资源（BGE 模型 / 向量库 / 知识库）与 DeepSeek API Key 均**不进 git**：
- 云服务器 / Render 冷启动自动下载 BGE 模型，向量库随 `space/chroma_db/` 提交（约 3MB）；
- API Key 通过环境变量（`LLM_API_KEY`）注入，**不写入代码**。


