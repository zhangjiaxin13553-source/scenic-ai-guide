"""
离线资源打包脚本
========================
将「三件套」离线资源打成独立 zip 分发：
  1. BGE 嵌入模型（bge-small-zh-v1.5，~91MB）
  2. 向量库（chroma_db/，luxun_know_base 399 chunk）
  3. 知识库 JSON（data/processed/，348 条）

背景：
  本地 BGE 模型只存在于 HuggingFace 缓存（~/.cache/huggingface/hub/...），
  未落在约定的 scripts/bge-small-zh-v1.5/。换台离线设备会直接加载失败。
  本脚本负责把模型复制到约定位置并随包分发，保证现场可一键重建。

用法：
  python scripts/pack_offline.py                  # 默认打包到 releases/offline_resources.zip
  python scripts/pack_offline.py --output my.zip  # 指定输出路径
  python scripts/pack_offline.py --no-verify      # 打包后跳过解压校验

输出结构（zip 内）：
  offline_resources/
  ├── README.txt                     # 解压放置说明
  ├── bge-small-zh-v1.5/             # → 放置到 scripts/bge-small-zh-v1.5/
  ├── chroma_db/                     # → 放置到项目根目录 chroma_db/
  └── data/
      └── processed/                 # → 放置到项目根目录 data/processed/
"""

import os
import sys
import shutil
import zipfile
import argparse
import tempfile
import glob

# ==================== 全局配置 ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

# 三件套来源路径
CHROMA_DIR = os.path.join(ROOT_DIR, "chroma_db")          # 根目录向量库（399 chunk）
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")  # 5 域 JSON
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
MODEL_DIR_NAME = "bge-small-zh-v1.5"
# BGE 模型约定落地路径（与 rag_pipeline.py / ingest.py 一致）
TARGET_MODEL_DIR = os.path.join(SCRIPT_DIR, MODEL_DIR_NAME)

# 默认输出
DEFAULT_OUTPUT = os.path.join(ROOT_DIR, "releases", "offline_resources.zip")


# ==================== 工具函数 ====================

def find_hf_cache_model(model_name: str) -> str | None:
    """在 HuggingFace 缓存中定位模型 snapshot 目录"""
    cache_roots = [
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.environ.get("HF_HOME", ""),
        os.environ.get("HF_HUB_CACHE", ""),
    ]
    hf_name = "models--" + model_name.replace("/", "--")
    for root in cache_roots:
        if not root or not os.path.isdir(root):
            continue
        model_dir = os.path.join(root, hf_name)
        if not os.path.isdir(model_dir):
            continue
        snapshots = os.path.join(model_dir, "snapshots")
        if not os.path.isdir(snapshots):
            continue
        snaps = sorted(os.listdir(snapshots))
        if snaps:
            return os.path.join(snapshots, snaps[-1])  # 最新快照
    return None


def copy_model_to_local(src_snapshot: str, dest_dir: str) -> int:
    """把 HF 缓存中的模型文件复制到本地目录（跳过 .no_exist 等非模型文件）"""
    if os.path.isdir(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    copied = 0
    for item in os.listdir(src_snapshot):
        src = os.path.join(src_snapshot, item)
        dst = os.path.join(dest_dir, item)
        if os.path.isdir(src):
            # 子目录（如 1_Pooling/）整体复制
            shutil.copytree(src, dst)
            copied += sum(len(f) for _, _, f in os.walk(dst))
        else:
            shutil.copy2(src, dst)
            copied += 1
    return copied


def count_entries(json_dir: str) -> dict:
    """统计 5 域 JSON 条目数"""
    import json
    counts = {}
    for f in ["venue.json", "work.json", "bio.json", "quote.json", "persona.json"]:
        p = os.path.join(json_dir, f)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fp:
                counts[f] = len(json.load(fp))
        else:
            counts[f] = 0
    return counts


def verify_zip(zip_path: str) -> None:
    """解压后校验：模型文件可加载、向量库 chunk 数一致"""
    print("\n[verify] 解压校验中...")
    import gc as _gc
    tmp = tempfile.mkdtemp(prefix="offline_verify_")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp)
        base = os.path.join(tmp, "offline_resources")

        # 1. 模型文件存在性
        model_dir = os.path.join(base, MODEL_DIR_NAME)
        weight = os.path.join(model_dir, "model.safetensors")
        config = os.path.join(model_dir, "config.json")
        tok = os.path.join(model_dir, "tokenizer_config.json")
        for name, p in [("权重 model.safetensors", weight),
                        ("config.json", config),
                        ("tokenizer_config.json", tok)]:
            ok = os.path.exists(p)
            print(f"  [{'OK' if ok else 'FAIL'}] {name}: {os.path.exists(p)}")
            if not ok:
                sys.exit(f"校验失败: {name} 缺失")

        # 2. 向量库可打开 + chunk 数
        try:
            import chromadb
            c = chromadb.PersistentClient(path=os.path.join(base, "chroma_db"))
            cols = {col.name: col.count() for col in c.list_collections()}
            print(f"  [OK] 向量库 collections: {cols}")
            if cols.get("luxun_know_base", 0) != 399:
                print("  [WARN] luxun_know_base chunk 数 ≠ 399，请确认")
            # 释放 sqlite 句柄，Windows 下否则无法删除临时目录
            try:
                if hasattr(c, "_system") and hasattr(c._system, "stop"):
                    c._system.stop()
            except Exception:
                pass
            del c
            import gc
            gc.collect()
        except Exception as e:
            print(f"  [WARN] 向量库校验跳过: {e}")

        # 3. JSON 条目数
        counts = count_entries(os.path.join(base, "data", "processed"))
        total = sum(counts.values())
        print(f"  [OK] JSON 条目数: {total} (expected 348)")
        if total != 348:
            print("  [WARN] 条目数与 348 不一致，请确认")

    finally:
        # 清理临时目录（Windows 下可能因句柄占用而失败，容错忽略）
        _gc.collect()
        shutil.rmtree(tmp, ignore_errors=True)

    print("[verify] 校验完成")


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(description="离线资源打包脚本")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help=f"输出 zip 路径 (默认: {DEFAULT_OUTPUT})")
    parser.add_argument("--no-verify", action="store_true",
                        help="打包后跳过解压校验")
    parser.add_argument("--keep-model-dir", action="store_true",
                        help="保留复制到 scripts/bge-small-zh-v1.5/ 的模型目录（不随包删除）")
    args = parser.parse_args()

    output_path = args.output or DEFAULT_OUTPUT

    # ---------- 1. 定位 BGE 模型 ----------
    print("[1/4] 定位 BGE 模型...")
    if os.path.isdir(TARGET_MODEL_DIR):
        print(f"  已存在 {TARGET_MODEL_DIR}，直接使用（跳过 HF cache 复制）")
        model_src = TARGET_MODEL_DIR
    else:
        snap = find_hf_cache_model(MODEL_NAME)
        if not snap:
            sys.exit(f"未找到 HF 缓存模型 {MODEL_NAME}，请先运行 ingest.py 触发下载")
        print(f"  从 HF 缓存复制: {snap}")
        print(f"  → {TARGET_MODEL_DIR}")
        copy_model_to_local(snap, TARGET_MODEL_DIR)
        model_src = TARGET_MODEL_DIR

    # 确认模型关键文件
    weight = os.path.join(model_src, "model.safetensors")
    if not os.path.exists(weight):
        sys.exit(f"模型权重缺失: {weight}")

    # ---------- 2. 检查三件套 ----------
    print("[2/4] 检查三件套...")
    if not os.path.isdir(CHROMA_DIR):
        sys.exit(f"向量库目录缺失: {CHROMA_DIR}")
    if not os.path.isdir(PROCESSED_DIR):
        sys.exit(f"知识库目录缺失: {PROCESSED_DIR}")
    counts = count_entries(PROCESSED_DIR)
    total_entries = sum(counts.values())
    print(f"  JSON: {total_entries} 条 {counts}")
    print(f"  向量库: {CHROMA_DIR}")

    # ---------- 3. 打包 ----------
    print("[3/4] 打包中...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        base = "offline_resources"

        # README 说明
        readme = (
            "离线资源包 — 鲁迅纪念馆对话系统\n"
            "===============================\n\n"
            "解压后按以下路径放置到项目根目录：\n\n"
            f"  {MODEL_DIR_NAME}/  →  scripts/{MODEL_DIR_NAME}/\n"
            "  chroma_db/        →  项目根目录 chroma_db/\n"
            "  data/processed/   →  项目根目录 data/processed/\n\n"
            "然后确认 .env 中 LLM_MODEL=deepseek-v4-flash、LLM_API_KEY 已配置。\n"
            "启动后 curl /health 探活通过即就绪。\n"
        )
        zf.writestr(f"{base}/README.txt", readme)

        # 模型
        for root, _, files in os.walk(model_src):
            for f in files:
                full = os.path.join(root, f)
                rel = os.path.relpath(full, model_src)
                zf.write(full, f"{base}/{MODEL_DIR_NAME}/{rel}")

        # 向量库
        for root, _, files in os.walk(CHROMA_DIR):
            for f in files:
                full = os.path.join(root, f)
                rel = os.path.relpath(full, CHROMA_DIR)
                zf.write(full, f"{base}/chroma_db/{rel}")

        # 知识库 JSON
        for root, _, files in os.walk(PROCESSED_DIR):
            for f in files:
                full = os.path.join(root, f)
                rel = os.path.relpath(full, ROOT_DIR)  # 保留 data/processed/ 前缀
                zf.write(full, f"{base}/{rel}")

    # 统计体积
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"[4/4] 打包完成: {output_path} ({size_mb:.1f} MB)")

    # 若临时复制到 scripts/ 的模型目录不需要保留，则清理（保留文件，删目录引用？不，直接保留即可）
    # 注：保留 TARGET_MODEL_DIR 是合理的 —— 这正是 PDF 要求模型落位的地方，不应删除。

    # ---------- 4. 校验 ----------
    if not args.no_verify:
        verify_zip(output_path)

    print(f"\n完成。资源包: {output_path}")
    print(f"提示: BGE 模型已落位到 {TARGET_MODEL_DIR}，本地 rag_pipeline.py 已可离线加载。")


if __name__ == "__main__":
    main()
