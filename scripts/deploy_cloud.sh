#!/usr/bin/env bash
# 云服务器一键部署脚本 —— 鲁迅数字人 在线体验 Demo（Ubuntu 22.04 / 24.04）
# ============================================================================
# 作用：在一台「有公网 IP」的云服务器上，把 Gradio 聊天界面跑起来，
#       公网直接 http://<公网IP>:7860 访问（不需要 gradio --share 隧道，不依赖墙外中转）。
#
# 用法（详见 docs/cloud-server-deploy.md）：
#   1) 本机先打包离线资源： python scripts/pack_offline.py   → releases/offline_resources.zip
#   2) 把资源包 + 本脚本传到服务器（或用 git clone 拉代码）
#   3) 服务器上执行：
#        LLM_API_KEY=sk-xxx bash deploy_cloud.sh /path/to/offline_resources.zip
#      （不传 zip 路径也可以，脚本会 clone 仓库，但需自行补 BGE 模型 / 向量库 / 知识库）
#
# 可覆盖的变量：
#   APP_PORT  服务端口（默认 7860）
#   APP_DIR   安装目录（默认 $HOME/scenic-ai-guide）
#   REPO_URL  仓库地址
# ============================================================================
set -euo pipefail

APP_PORT="${APP_PORT:-7860}"
APP_DIR="${APP_DIR:-$HOME/scenic-ai-guide}"
REPO_URL="${REPO_URL:-https://github.com/zhangjiaxin13553-source/scenic-ai-guide.git}"
ZIP_PATH="${1:-}"                 # 可选：offline_resources.zip 的路径
VENV_DIR="$APP_DIR/venv"

# ---------- 1. 基础环境 ----------
echo "[1/6] 检查基础环境（python3 / git / unzip）..."
command -v python3 >/dev/null 2>&1 || sudo apt-get install -y python3
command -v git     >/dev/null 2>&1 || sudo apt-get install -y git
command -v unzip   >/dev/null 2>&1 || sudo apt-get install -y unzip
python3 -m venv --help >/dev/null 2>&1 || sudo apt-get install -y python3-venv

# ---------- 2. 拉取/更新代码 ----------
if [ ! -d "$APP_DIR" ]; then
  echo "[2/6] 克隆仓库到 $APP_DIR ..."
  git clone "$REPO_URL" "$APP_DIR"
else
  echo "[2/6] 目录已存在，git pull 更新到最新 ..."
  git -C "$APP_DIR" pull --ff-only 2>/dev/null \
    || echo "    ⚠ git pull 未成功（本地有改动或网络问题），继续使用现有代码"
fi
cd "$APP_DIR"

# 前置校验：space/requirements.txt 缺失说明代码是旧版，直接给出可操作提示
if [ ! -f space/requirements.txt ]; then
  echo "❌ 缺少 space/requirements.txt，当前代码不是最新版。"
  echo "   请先执行： cd $APP_DIR && git pull   （确认能联网拉到 GitHub）"
  exit 1
fi

# ---------- 3. 虚拟环境 + 依赖（CPU 版 torch，避免拉 ~2GB CUDA 轮子） ----------
echo "[3/6] 创建 venv 并安装依赖 ..."
python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q
# 先装 CPU 版 torch，再装其余依赖（space/requirements.txt 已刻意不含 torch）
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu -q
pip install -r space/requirements.txt -q

# ---------- 4. 放置离线资源（BGE 模型 / 向量库 / 知识库） ----------
echo "[4/6] 放置离线资源 ..."
if [ -n "$ZIP_PATH" ] && [ -f "$ZIP_PATH" ]; then
  TMPD="$(mktemp -d)"
  unzip -oq "$ZIP_PATH" -d "$TMPD"
  BASE="$TMPD/offline_resources"
  cp -r "$BASE/bge-small-zh-v1.5" scripts/            # → scripts/bge-small-zh-v1.5/
  cp -r "$BASE/chroma_db"           ./chroma_db       # → 项目根 chroma_db/
  cp -r "$BASE/data"                ./data            # → 项目根 data/processed/
  rm -rf "$TMPD"
  echo "    已放置 BGE 模型 + 向量库 + 知识库"
else
  echo "    ⚠ 未提供离线资源包（跳过）。首次运行会从 HF 下载 BGE 模型，且检索为空。"
  echo "      建议本机先跑： python scripts/pack_offline.py 再传上来。"
fi

# ---------- 5. 配置 .env ----------
echo "[5/6] 配置 .env ..."
if [ ! -f .env ]; then
  cp .env.example .env
  if [ -n "${LLM_API_KEY:-}" ]; then
    sed -i "s|^LLM_API_KEY=.*|LLM_API_KEY=${LLM_API_KEY}|" .env
    echo "    已写入 LLM_API_KEY"
  else
    echo "    ⚠ 未检测到 LLM_API_KEY，请编辑 $APP_DIR/.env 手动填入 DeepSeek Key"
  fi
else
  echo "    .env 已存在，跳过"
fi

# ---------- 6. 启动 ----------
echo "[6/6] 启动 Gradio 服务（端口 $APP_PORT）..."
pkill -f "gradio_app.py" 2>/dev/null || true
nohup python scripts/gradio_app.py --host 0.0.0.0 --port "$APP_PORT" > app.log 2>&1 &
sleep 8

# 探活
if curl -sS --max-time 5 "http://127.0.0.1:$APP_PORT" >/dev/null 2>&1; then
  echo ""
  echo "✅ 部署完成，服务已就绪。"
  echo "   本机探活:   curl http://127.0.0.1:$APP_PORT"
  echo "   公网访问:   http://<你的公网IP>:$APP_PORT   （记得在云控制台安全组放行 $APP_PORT 端口）"
  echo "   查看日志:   tail -f $APP_DIR/app.log"
else
  echo ""
  echo "⚠ 服务未在 8 秒内就绪，请查看日志： tail -50 $APP_DIR/app.log"
fi
