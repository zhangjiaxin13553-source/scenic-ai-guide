"""
Gradio 聊天界面 + 向量库构建一体化脚本
==============
功能更新：
1. Web上传文档扩充知识库(txt/json)
2. 模式控制：自动识别 / 手动讲解员 / 手动鲁迅数字人
3. 勾选开关：是否在输出附带检索知识库片段
4. 保留原有自动意图识别，手动模式优先级更高
用法：
  cd scenic-ai-guide

  # 1.构建基础向量知识库（首次运行必须执行）
  python scripts/gradio_app.py --build-db

  # 2.启动web聊天服务
  python scripts/gradio_app.py
"""

import sys
import os
# ==================== 全局环境变量【适配transformers5.x，消除所有警告】 ====================
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import json
import time
import logging
import argparse
import torch
import chromadb
from transformers import AutoTokenizer, AutoModel
import gradio as gr


logging.getLogger("transformers").setLevel(logging.ERROR)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

# ==================== 向量库构建模块（与 ingest.py 完全对齐） ====================
LOCAL_MODEL_PATH = os.path.join(SCRIPT_DIR, "bge-small-zh-v1.5")
HF_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# 优先本地路径（Transformers老版本环境），不存在则用 HF 名称（新版环境）
if os.path.isdir(LOCAL_MODEL_PATH):
    MODEL_PATH = LOCAL_MODEL_PATH
else:
    MODEL_PATH = HF_MODEL_NAME
CACHE_FOLDER = os.path.join(SCRIPT_DIR, "bge-small-zh-v1.5")

JSON_FOLDER = os.path.join(ROOT_DIR, "data", "processed")
CHROMA_STORE = os.path.join(ROOT_DIR, "chroma_db")
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH).to(device)
model.eval()


def get_embedding(text: str):
    """生成文本向量 — 与 ingest.py 保持 Mean Pooling 一致"""
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        out = model(**inputs)
    vec = out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().tolist()
    return vec


def slide_split(text: str):
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + CHUNK_SIZE, length)
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def load_all_formatted_json():
    json_files = ["venue.json", "work.json", "bio.json", "quote.json", "persona.json"]
    total_knowledge = []
    for file in json_files:
        file_path = os.path.join(JSON_FOLDER, file)
        if not os.path.exists(file_path):
            print(f"提示：{file} 不存在，跳过该类数据")
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)
            total_knowledge.extend(data_list)
    return total_knowledge


def build_vector_db_main():
    chroma_client = chromadb.PersistentClient(path=CHROMA_STORE)
    coll = chroma_client.get_or_create_collection(name="luxun_know_base")
    knowledge_data = load_all_formatted_json()
    print(f"成功加载标准化JSON条目总数：{len(knowledge_data)}")

    batch_ids = []
    batch_embeds = []
    batch_docs = []
    batch_meta = []
    for item in knowledge_data:
        base_id = item["id"]
        content_text = item["content"]
        text_blocks = slide_split(content_text)
        for idx, block in enumerate(text_blocks):
            chunk_id = f"{base_id}_blk_{idx}"
            emb = get_embedding(block)
            meta_info = {
                "type": item["type"],
                "title": item["title"],
                "source": item["source"],
                "year": item.get("year", 0),
                "character_voice": item.get("character_voice", False),
                "venue_relevant": item.get("venue_relevant", False),
                "tags": ",".join(item.get("tags", []))
            }
            batch_ids.append(chunk_id)
            batch_embeds.append(emb)
            batch_docs.append(block)
            batch_meta.append(meta_info)
    coll.add(
        ids=batch_ids,
        embeddings=batch_embeds,
        documents=batch_docs,
        metadatas=batch_meta
    )
    print(f"向量入库完成，总分块数量：{len(batch_ids)}")
    print(f"向量库存放路径：{CHROMA_STORE}")


# ========= Web上传文档，动态追加知识库 =========
def upload_knowledge_file(file):
    """web上传txt/json，解析、分块、向量化追加进chroma"""
    if file is None:
        return "⚠️ 未选择文件"
    chroma_client = chromadb.PersistentClient(path=CHROMA_STORE)
    coll = chroma_client.get_or_create_collection(name="luxun_know_base")
    filename = os.path.basename(file.name)
    ext = os.path.splitext(filename)[1].lower()

    all_text = ""
    if ext == ".txt":
        with open(file.name, "r", encoding="utf-8", errors="ignore") as f:
            all_text = f.read()
    elif ext == ".json":
        with open(file.name, "r", encoding="utf-8", errors="ignore") as f:
            json_data = json.load(f)
            if isinstance(json_data, list):
                all_text = "\n".join([i.get("content", "") for i in json_data])
            elif isinstance(json_data, dict):
                all_text = json_data.get("content", json.dumps(json_data, ensure_ascii=False))
    else:
        return f"❌ 不支持文件类型 {ext}，仅支持 .txt / .json"

    if not all_text.strip():
        return "❌ 文件读取后无有效文本"

    chunks = slide_split(all_text)
    batch_ids = []
    batch_embeds = []
    batch_docs = []
    batch_meta = []
    base_id = f"upload_{filename}_{int(time.time())}"
    for idx, block in enumerate(chunks):
        cid = f"{base_id}_blk_{idx}"
        emb = get_embedding(block)
        meta_info = {
            "type": "uploaded",
            "title": filename,
            "source": "web_upload",
            "year": 0,
            "character_voice": False,
            "venue_relevant": False,
            "tags": "upload"
        }
        batch_ids.append(cid)
        batch_embeds.append(emb)
        batch_docs.append(block)
        batch_meta.append(meta_info)
    coll.add(ids=batch_ids, embeddings=batch_embeds, documents=batch_docs, metadatas=batch_meta)
    return f"✅ 文件【{filename}】上传成功，新增 {len(chunks)} 个知识库分块"


# ==================== Gradio RAG聊天管线封装 ====================
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        print("正在加载 RAG 全链路（BGE 模型 + ChromaDB + LLM 客户端）...")
        from rag_pipeline import RAGPipeline, ConversationState
        _pipeline = RAGPipeline()
        print("加载完成。")
    return _pipeline


def reset_pipeline():
    global _pipeline
    if _pipeline is not None:
        from rag_pipeline import ConversationState
        _pipeline.state = ConversationState()


INTENT_LABELS = {
    "narrator":  "🏛️ 讲解员模式",
    "luxun":     "🎭 鲁迅数字人",
    "ambiguous": "🎭 鲁迅数字人（自动）",
    "reject_time":        "⚠️ 时间越界·已拦截",
    "reject_irrelevant":  "🚫 无关内容·已拒绝",
}


def current_mode_label():
    pipeline = get_pipeline()
    intent = pipeline.state.current_intent
    return INTENT_LABELS.get(intent, f"❓ {intent}")


# --------------------------
# 【重要】rag_pipeline.py 需要做微小兼容改动（你原有逻辑不动，新增两个可选参数）
# def ask(self, query:str, force_mode:str=None, return_retrieval:bool=False):
#    force_mode: None=自动识别; "narrator"/"luxun"手动强制模式
#    return_retrieval=True 返回 (answer_text, retrieval_docs)
# --------------------------
def respond(message, history, manual_mode, show_retrieval):
    if not message or not message.strip():
        return history, current_mode_label(), ""

    pipeline = get_pipeline()
    force_mode = None
    if manual_mode == "讲解员模式":
        force_mode = "narrator"
    elif manual_mode == "鲁迅数字人模式":
        force_mode = "luxun"

    try:
        # 透传强制模式、是否返回检索片段
        result = pipeline.ask(message.strip(), force_mode=force_mode, return_retrieval=show_retrieval)
        if show_retrieval and isinstance(result, tuple):
            reply, retrieve_docs = result
        else:
            reply = result
            retrieve_docs = []
    except Exception as e:
        reply = f"（生成回答时出错：{e}）"
        retrieve_docs = []

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})

    # ==========适配你的数据格式的检索输出==========
    retrieve_text = ""
    if show_retrieval and retrieve_docs:
        retrieve_text += "=====🔍检索知识库片段（Top4）=====\n"
        for idx, item in enumerate(retrieve_docs[:4]):
            doc_id = item.get("id", "")
            doc_type = item.get("type", "")
            doc_title = item.get("title", "")
            doc_content = item.get("content", "")
            doc_source = item.get("source", "未知资料来源")
            doc_year = item.get("year", "")
            tags_list = item.get("tags", [])
            tags_str = "、".join(tags_list) if tags_list else ""

            retrieve_text += f"\n———— 片段 {idx+1} ————\n"
            retrieve_text += f"ID: {doc_id} | 类型:{doc_type} | 年份:{doc_year}\n"
            retrieve_text += f"标题：{doc_title}\n"
            retrieve_text += f"资料出处：{doc_source}\n"
            if tags_str:
                retrieve_text += f"标签：{tags_str}\n"
            retrieve_text += f"\n{doc_content.strip()}\n"

    return history, current_mode_label(), retrieve_text

def reset_chat():
    reset_pipeline()
    return [], "🔄 对话已重置", "", "自动识别模式"


# ---------- Gradio UI ----------
def create_ui():
    css = """
    .mode-display textarea {
        font-size: 16px !important;
        font-weight: 600 !important;
        text-align: center !important;
        background: #f0f4f8 !important;
        border: 1px solid #d0d7de !important;
    }
    .guide-box {
        font-size: 13px;
        color: #57606a;
        line-height: 1.7;
    }
    footer { display: none !important; }
    """

    with gr.Blocks(title="鲁迅数字人 · 双模式对话系统") as demo:
        gr.Markdown(
            """
            # 🏛️ 鲁迅数字人 · 双模式对话系统
            **讲解员模式** · 回答场馆、展品、参观信息 ｜ **数字人模式** · 以鲁迅口吻与你对话
            """
        )
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    value=[],
                    height=480,
                    label="对话",
                    placeholder="输入你的问题开始对话...",
                )
                retrieve_output = gr.Textbox(
                    label="🔍知识库检索结果",
                    interactive=False,
                    lines=8,
                    placeholder="勾选「展示检索结果」后这里显示检索片段"
                )

                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="输入问题，如「鲁迅先生，您为什么要弃医从文？」",
                        label="",
                        scale=5,
                        container=False,
                    )
                    submit_btn = gr.Button("发送", variant="primary", scale=1)

                with gr.Row():
                    reset_btn = gr.Button("🔄 重置对话", variant="secondary", size="sm")
                    gr.Markdown(
                        "直接输入即可 · 支持手动强制模式或自动识别",
                        elem_classes="guide-box",
                    )

            with gr.Column(scale=1):
                gr.Markdown("### 当前运行模式")
                mode_display = gr.Textbox(
                    value="⏳ 正在加载模型...",
                    label="",
                    interactive=False,
                    elem_classes="mode-display",
                    container=True,
                )
                manual_mode_radio = gr.Radio(
                    ["自动识别模式", "讲解员模式", "鲁迅数字人模式"],
                    value="自动识别模式",
                    label="手动切换模式（手动优先级高于自动识别）"
                )
                show_retrieval_check = gr.Checkbox(
                    value=False,
                    label="输出附带知识库检索结果"
                )
                gr.Markdown("---")
                gr.Markdown("### 📁上传文档扩充知识库")
                upload_file = gr.File(label="上传txt/json文档", file_types=[".txt", ".json"])
                upload_info = gr.Textbox(label="上传状态", interactive=False)
                gr.Markdown(
                    """
                    **💡 使用提示**
                    - 自动识别：系统根据问题自动切换讲解员/数字人
                    - 手动模式：强制固定回答风格
                    - 勾选【输出附带知识库检索结果】，查看RAG召回的原始片段
                    - 上传txt/json可动态增加知识库，无需重启服务

                    **🗣️ 试试这样问**
                    - 这个展厅主要展什么？
                    - 鲁迅先生，您怎么看现在的年轻人？
                    """,
                    elem_classes="guide-box",
                )

        # 事件绑定
        submit_inputs = [msg, chatbot, manual_mode_radio, show_retrieval_check]
        submit_outputs = [chatbot, mode_display, retrieve_output]

        msg.submit(respond, submit_inputs, submit_outputs).then(lambda: "", None, [msg])
        submit_btn.click(respond, submit_inputs, submit_outputs).then(lambda: "", None, [msg])
        reset_btn.click(reset_chat, None, [chatbot, mode_display, retrieve_output, manual_mode_radio])

        upload_file.upload(upload_knowledge_file, inputs=[upload_file], outputs=[upload_info])

        demo.load(lambda: "✅ 就绪 · 等待输入", None, [mode_display])
    return demo


# ---------- 入口 ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="鲁迅数字人 Gradio聊天 + 向量库一体化脚本")
    parser.add_argument("--build-db", action="store_true", help="执行构建Chroma向量库，构建完成程序直接退出，不启动web界面")
    parser.add_argument("--port", type=int, default=7860, help="服务端口（默认 7860）")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="绑定地址")
    parser.add_argument("--share", action="store_true", help="生成公网链接")
    args = parser.parse_args()

    if args.build_db:
        print("===== 开始构建向量知识库 =====")
        build_vector_db_main()
        print("===== 向量库构建完毕，程序退出 =====")
        sys.exit(0)

    print("正在初始化 RAG 管线...")
    get_pipeline()
    print(f"启动 Gradio 服务 → http://{args.host}:{args.port}")

    demo = create_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True
    )