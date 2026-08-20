# ⚠️ 已归档（2026-08-19）：本文件为 gradio_app.py 的增强实验版，已停止维护。
#   现役 Gradio 入口统一为 scripts/gradio_app.py（本机/云服务器）与 space/app.py（HF/Render）。
#   归档原因：路径依赖 cwd 脆弱、硬编码 HF 国内镜像、内嵌 BGE 加载与 rag_pipeline 重复。
#   留底参考，勿再作为启动入口。

# Gradio 聊天界面 + 向量库构建 + RAG全链路质量守卫一体化脚本
# 【重要：本版本只修改gradio_app，rag_pipeline.py完全不改动】
"""
功能清单：
1. 向量库构建：python scripts/gradio_app.py --build-db 首次初始化知识库
2. Web聊天服务：python scripts/gradio_app.py
3. Web能力：
   - txt/json文件上传动态扩充向量库
   - 模式控制：自动识别 / 手动讲解员 / 手动鲁迅数字人
   - 开关1：展示知识库检索召回片段
   - 开关2：Verbose调试模式，输出完整链路每一步决策日志（对应scripts/demo.py演示脚本需求）
4. 底层完整复用rag_pipeline全链路：意图分类→越界改写→多检索合并→LLM生成→质量校验重试兜底

【rag_pipeline.py 接口契约，必须遵守】
def ask(self,
        user_input: str,
        force_mode: str | None = None,      # None / "narrator" / "luxun"
        return_retrieval: bool = False,
        verbose: bool = False) -> str | tuple[str, list[dict]] | dict:

    返回规则：
    1. return_retrieval=False：返回回答字符串 str，或者 {"success":True,"content":"xxx"} 结构体
    2. return_retrieval=True：返回 (answer, retrieve_docs)
       retrieve_docs中每条文档字段：chunk_id,type,title,content,source,year,tags(字符串)
"""
import sys
import os
import io

# ==================== 全局环境变量【transformers5.x兼容，消除所有警告】====================
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import json
import logging
import time
import torch
import chromadb
from transformers import AutoTokenizer, AutoModel
import gradio as gr
from contextlib import redirect_stdout

# 屏蔽transformers冗余日志
logging.getLogger("transformers").setLevel(logging.ERROR)
logger = logging.getLogger("gradio_rag_demo")

# ---------------- 新增：捕获logging日志的Handler ----------------
class LogCaptureHandler(logging.Handler):
    def __init__(self, buffer: io.StringIO):
        super().__init__()
        self.buffer = buffer

    def emit(self, record):
        self.buffer.write(f"{self.format(record)}\n")

# 路径统一管理
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
current_dir = os.getcwd()

# ==================== 全局BGE向量化模型（统一复用，避免重复加载显存爆炸） ====================
REPO_ID = "BAAI/bge-small-zh-v1.5"
CACHE_FOLDER = os.path.join(current_dir, "bge-small-zh-v1.5")
ROOT = os.path.dirname(current_dir)
JSON_FOLDER = os.path.join(ROOT, "data", "processed")
CHROMA_STORE = os.path.join(ROOT, "chroma_db")
COLLECTION_NAME = "luxun_know_base"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"加载全局BGE模型，设备：{device}")
tokenizer = AutoTokenizer.from_pretrained(REPO_ID, cache_dir=CACHE_FOLDER)
model = AutoModel.from_pretrained(REPO_ID, cache_dir=CACHE_FOLDER).to(device)
model.eval()


def get_embedding(text: str) -> list:
    """全局统一向量化函数，与rag_pipeline、向量入库逻辑100%对齐"""
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        out = model(**inputs)
    vec = out.last_hidden_state[:, 0]
    vec = torch.nn.functional.normalize(vec, p=2, dim=1).squeeze()
    return vec.cpu().numpy().tolist()


# ==================== 向量库构建模块（完整保留原入库逻辑） ====================
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
    coll = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
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
    coll = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
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


# ==================== RAG管线全局单例（复用rag_pipeline完整链路逻辑） ====================
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        print("正在加载 RAG 全链路（意图分类/越界改写/质量守卫/LLM客户端）...")
        # 导入外部完整RAG管线（rag_pipeline.py完全不动）
        from rag_pipeline import RAGPipeline, ConversationState
        _pipeline = RAGPipeline()
        print("RAG全链路加载完成，包含4层越界防护+后置质量校验重试")
    return _pipeline


def reset_pipeline():
    """重置对话历史，清空上下文"""
    global _pipeline
    if _pipeline is not None:
        from rag_pipeline import ConversationState
        _pipeline.state = ConversationState()


INTENT_LABELS = {
    "narrator": "🏛️ 讲解员模式",
    "luxun": "🎭 鲁迅数字人",
    "ambiguous": "🎭 鲁迅数字人（自动）",
    "reject_time": "⚠️ 时间越界·已拦截",
    "reject_irrelevant": "🚫 无关内容·已拒绝",
}


def current_mode_label():
    """获取当前运行模式展示文本"""
    pipeline = get_pipeline()
    intent = pipeline.state.current_intent
    return INTENT_LABELS.get(intent, f"❓ 未知意图:{intent}")


# ==================== 核心对话响应函数【只修改gradio_app，底层rag_pipeline不动】 ====================
def respond(message, history, manual_mode, show_retrieval, verbose_switch):
    if not message or not message.strip():
        return history, current_mode_label(), "", ""

    pipeline = get_pipeline()
    force_mode = None
    # 手动模式优先级高于自动意图识别
    if manual_mode == "讲解员模式":
        force_mode = "narrator"
    elif manual_mode == "鲁迅数字人模式":
        force_mode = "luxun"

    retrieve_docs = []
    raw_reply = ""
    full_verbose_log = ""
    user_input = message.strip()

    stdout_buf = None
    log_buf = None
    log_handler = None

    # 只有勾选Verbose调试，才开启日志捕获
    if verbose_switch:
        stdout_buf = io.StringIO()
        log_buf = io.StringIO()
        log_handler = LogCaptureHandler(log_buf)
        log_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s - %(message)s"))
        root_logger = logging.getLogger()
        root_logger.addHandler(log_handler)

        try:
            with redirect_stdout(stdout_buf):
                result = pipeline.ask(
                    user_input=user_input,
                    verbose=verbose_switch,
                    force_mode=force_mode,
                    return_retrieval=show_retrieval
                )
                if show_retrieval and isinstance(result, tuple):
                    raw_reply, retrieve_docs = result
                else:
                    raw_reply = result
                    retrieve_docs = []
        except Exception as e:
            raw_reply = f"（RAG链路发生异常：{str(e)}，上下文已自动重置，请重新提问）"
            retrieve_docs = []
            logger.error(f"RAG链路执行失败: {e}", exc_info=True)
            # 关键：不修改rag_pipeline内部代码，出现异常直接整体重置pipeline状态，切断脏历史，防止连环报错
            reset_pipeline()
        finally:
            if log_handler is not None:
                logging.getLogger().removeHandler(log_handler)
            # 合并print输出 + logger输出
            print_content = stdout_buf.getvalue()
            logger_content = log_buf.getvalue()
            full_verbose_log = f"==== Print输出 ====\n{print_content}\n==== Logger链路日志 ====\n{logger_content}"
            stdout_buf.close()
            log_buf.close()
    else:
        # 未开启Verbose，直接执行，不捕获日志
        try:
            result = pipeline.ask(
                user_input=user_input,
                verbose=False,
                force_mode=force_mode,
                return_retrieval=show_retrieval
            )
            if show_retrieval and isinstance(result, tuple):
                raw_reply, retrieve_docs = result
            else:
                raw_reply = result
                retrieve_docs = []
        except Exception as e:
            raw_reply = f"（RAG链路发生异常：{str(e)}，上下文已自动重置，请重新提问）"
            retrieve_docs = []
            logger.error(f"RAG链路执行失败: {e}", exc_info=True)
            # 异常直接重置整个pipeline状态，规避脏历史连环报错
            reset_pipeline()
        full_verbose_log = "💡未开启Verbose调试开关，无链路日志，请勾选「开启Verbose调试」查看全流程决策"

    # ==========【关键修复】兼容两种返回格式：纯字符串 / {"success":True,"content":"xxx"} 结构体 ==========
    if isinstance(raw_reply, dict):
        # 如果返回的是完整返回包，提取content字段
        reply = raw_reply.get("content", str(raw_reply))
    else:
        reply = str(raw_reply)

    # Gradio4 Chatbot messages格式：content必须是字符串，禁止传入dict
    new_history = history.copy()
    new_history.append({"role": "user", "content": message})
    new_history.append({"role": "assistant", "content": reply})

    # =========【BUG修复】字段对齐：retrieve_docs主键是 chunk_id，不是 id ==========
    retrieve_text = ""
    if show_retrieval and retrieve_docs:
        retrieve_text += "=====🔍检索知识库片段（Top4）=====\n"
        for idx, item in enumerate(retrieve_docs[:4]):
            doc_id = item.get("chunk_id", "")
            doc_type = item.get("type", "")
            doc_title = item.get("title", "")
            doc_content = item.get("content", "")
            doc_source = item.get("source", "未知资料来源")
            doc_year = item.get("year", "")
            tags_str = item.get("tags", "")

            retrieve_text += f"\n———— 片段 {idx + 1} ————\n"
            retrieve_text += f"chunk_id: {doc_id} | 类型:{doc_type} | 年份:{doc_year}\n"
            retrieve_text += f"标题：{doc_title}\n"
            retrieve_text += f"资料出处：{doc_source}\n"
            if tags_str:
                retrieve_text += f"标签：{tags_str}\n"
            retrieve_text += f"\n{doc_content.strip()}\n"

    return new_history, current_mode_label(), retrieve_text, full_verbose_log


def reset_chat():
    """一键重置对话、模式、检索面板、日志面板"""
    reset_pipeline()
    # 返回顺序：chatbot,mode_display,retrieve_output,verbose_log_output,manual_mode_radio
    return [], "🔄 对话已重置", "", "", "自动识别模式"


# ---------- Gradio UI布局（新增Verbose调试日志面板，对应demo脚本全链路演示需求） ----------
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
    .log-panel textarea {
        background: #1a1a1a;
        color: #89ddff;
        font-family: monospace;
    }
    """

    with gr.Blocks(title="鲁迅数字人 · 端到端RAG全链路演示系统") as demo:
        gr.Markdown(
            """
            # 🏛️ 鲁迅数字人 · 端到端RAG全链路演示系统
            **完整链路：意图分类 → 4层越界防护 → 多子查询检索合并 → LLM生成 → 后置质量守卫重试兜底**
            讲解员模式（场馆问答）｜鲁迅数字人模式（角色对话）｜Verbose调试可查看每一步决策日志
            """
        )
        with gr.Row():
            # 左侧对话主面板
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    value=[],
                    height=420,
                    label="对话窗口",
                    placeholder="输入你的问题开始对话...",

                )
                retrieve_output = gr.Textbox(
                    label="🔍知识库检索召回片段",
                    interactive=False,
                    lines=6,
                    placeholder="勾选「展示检索结果」后显示向量库召回原文"
                )
                verbose_log_output = gr.Textbox(
                    label="📜Verbose全链路执行日志（对应demo.py演示脚本）",
                    interactive=False,
                    lines=7,
                    placeholder="勾选「开启Verbose调试」查看意图/改写/检索/质量校验每一步决策",
                    elem_classes="log-panel"
                )

                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="输入问题，如「鲁迅先生，您怎么看待现代互联网？」",
                        label="",
                        scale=5,
                        container=False,
                    )
                    submit_btn = gr.Button("发送", variant="primary", scale=1)

                with gr.Row():
                    reset_btn = gr.Button("🔄 重置对话", variant="secondary", size="sm")
                    gr.Markdown(
                        "自动识别/手动强制模式切换 · 支持完整链路调试演示",
                        elem_classes="guide-box",
                    )
            # 右侧控制面板
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
                    label="手动切换模式（手动优先级高于自动意图识别）"
                )
                show_retrieval_check = gr.Checkbox(
                    value=False,
                    label="输出附带知识库检索结果"
                )
                verbose_switch = gr.Checkbox(
                    value=False,
                    label="开启Verbose调试（展示全链路每一步决策）",
                    info="对应需求：可演示完整链路日志，包含质量守卫/熔断/兜底逻辑"
                )
                gr.Markdown("---")
                gr.Markdown("### 📁上传文档扩充向量知识库")
                upload_file = gr.File(label="上传txt/json文档", file_types=[".txt", ".json"])
                upload_info = gr.Textbox(label="文件上传状态", interactive=False)
                gr.Markdown(
                    """
                    **💡 功能说明（对应需求端到端Demo脚本）**
                    1. Verbose调试开启后，日志面板输出全链路每一步决策：
                       意图分类置信度、越界概念检测、查询改写子查询、多检索合并、质量守卫校验结果、重试兜底逻辑
                    2. 四层越界防护自动执行：意图拦截→查询改写→多子查询检索→系统Prompt时间边界约束
                    3. 后置质量守卫自动重试，回答空洞/过短时触发二次生成兜底文本
                    4. 向量库初始化：`python scripts/gradio_app.py --build-db`
                    5. 动态上传txt/json无需重启服务，实时追加知识库

                    > ⚠️注意：底层rag_pipeline未修改，一旦链路异常会自动重置全部上下文，之前对话会丢失
                    **🗣️ 测试示例（触发越界改写演示）**
                    - 鲁迅先生，您怎么看待手机短视频？
                    - 鲁迅纪念馆的展厅开放时间是？
                    """,
                    elem_classes="guide-box",
                )

        # 事件绑定：提交消息输出 对话/模式标签/检索片段/verbose日志
        submit_inputs = [msg, chatbot, manual_mode_radio, show_retrieval_check, verbose_switch]
        submit_outputs = [chatbot, mode_display, retrieve_output, verbose_log_output]

        msg.submit(respond, submit_inputs, submit_outputs).then(lambda: "", None, [msg])
        submit_btn.click(respond, submit_inputs, submit_outputs).then(lambda: "", None, [msg])
        # 重置按钮：清空对话、模式、检索面板、日志面板、恢复默认自动模式
        reset_btn.click(
            reset_chat,
            None,
            [chatbot, mode_display, retrieve_output, verbose_log_output, manual_mode_radio]
        )
        # 文件上传事件
        upload_file.upload(upload_knowledge_file, inputs=[upload_file], outputs=[upload_info])
        # 页面加载初始化状态
        demo.load(lambda: "✅ RAG管线就绪 · 等待输入", None, [mode_display])
    return demo


# ---------- 程序入口（支持向量库构建参数/端口/公网分享） ----------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="鲁迅数字人 Gradio端到端RAG演示系统")
    parser.add_argument("--build-db", action="store_true",
                        help="构建初始化Chroma向量库，构建完成直接退出程序，不启动Web界面")
    parser.add_argument("--port", type=int, default=7860, help="Web服务端口（默认7860）")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务绑定地址，0.0.0.0允许局域网访问")
    parser.add_argument("--share", action="store_true", help="生成gradio公网临时访问链接")
    args = parser.parse_args()

    # 向量库构建分支
    if args.build_db:
        print("===== 开始初始化构建向量知识库 =====")
        build_vector_db_main()
        print("===== 向量库构建完成，程序退出 =====")
        sys.exit(0)

    # 预加载RAG管线，避免首次对话卡顿
    print("预加载RAG全链路管线...")
    get_pipeline()
    print(f"Gradio Web服务启动地址：http://{args.host}:{args.port}")

    demo = create_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True
    )