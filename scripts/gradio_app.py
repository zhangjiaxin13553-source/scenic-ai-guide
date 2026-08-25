"""
Gradio 聊天界面（鲁迅数字人智能对话系统 V1.0）
==============
对接 rag_pipeline.py，提供双模式（讲解/数字人）对话界面 + 调试面板。

用法：
  cd scenic-ai-guide
  python scripts/gradio_app.py

功能：
  - 输入框 + 对话区
  - 实时显示当前模式（讲解员 / 鲁迅数字人）
  - 调试面板：意图判定、检索结果、守卫状态、耗时
  - 错误状态的鲁迅口吻展示
  - 重置对话按钮
  - 自动意图识别，无需手动切换模式
"""

import sys
import os
import time

import gradio as gr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

# ---------- 延迟导入：启动时给用户友好的加载提示 ----------

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
        _pipeline.last_debug = {}


# ---------- 意图 → 中文标签 ----------

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


# ---------- 调试面板内容生成 ----------

def build_debug_html(debug: dict, mode_label: str) -> str:
    """根据 last_debug 构建调试面板 HTML。"""
    if not debug or not debug.get("intent"):
        return "<div style='color:#888; padding:12px; font-style:italic;'>发送一条消息后，调试信息将在此显示。</div>"

    intent = debug["intent"]
    confidence = debug.get("confidence", 0)
    reason = debug.get("reason", "")
    matched = debug.get("matched", [])
    time_boundary = debug.get("time_boundary", False)
    time_warning = debug.get("time_warning", "")
    rewrite_used = debug.get("rewrite_used", False)
    rewrite_concepts = debug.get("rewrite_concepts", [])
    rewrite_queries = debug.get("rewrite_queries", [])
    rewrite_fallback = debug.get("rewrite_fallback", "")
    chunks = debug.get("chunks", [])
    chunk_count = debug.get("chunk_count", 0)
    guard_status = debug.get("guard_status", "PASS")
    elapsed_ms = debug.get("elapsed_ms", 0)
    reply_len = debug.get("reply_len", 0)
    error = debug.get("error")

    guard_color = {"PASS": "#2da44e", "WARN": "#d4a72c", "FAIL": "#cf222e"}

    lines = []

    # ── 错误/警告横幅（鲁迅口吻）──
    if guard_status == "FAIL":
        error_luxun = build_error_luxun(error or "unknown")
        lines.append(
            f"<div style='margin:0 0 10px 0; padding:10px 12px; "
            f"background:#fff1f0; border:1px solid #cf222e; border-radius:6px;'>"
            f"<div style='font-weight:700; color:#cf222e; margin-bottom:6px;'>"
            f"⚠ 检测到异常 · 守卫判定 {guard_status}</div>"
            f"<div style='font-size:13px; color:#1f2328; font-style:italic; line-height:1.6;'>"
            f"「{error_luxun}」</div>"
            f"<div style='font-size:11px; color:#cf222e; margin-top:6px; "
            f"font-family:monospace;'>detail: {error}</div>"
            f"</div>"
        )
    elif guard_status == "WARN":
        warn_msg = ""
        if elapsed_ms and (elapsed_ms / 1000) > 30:
            warn_msg = "回答耗时超过30秒，已触发超时熔断。"
        elif error:
            warn_msg = f"注意：{error}"
        else:
            warn_msg = "守卫检测到潜在问题，建议人工复查。"

        lines.append(
            f"<div style='margin:0 0 10px 0; padding:10px 12px; "
            f"background:#fff8c5; border:1px solid #d4a72c; border-radius:6px;'>"
            f"<div style='font-weight:700; color:#9a6700; margin-bottom:4px;'>"
            f"⚠ 守卫提示 · {guard_status}</div>"
            f"<div style='font-size:13px; color:#1f2328; line-height:1.5;'>"
            f"{warn_msg}</div>"
            f"</div>"
        )

    # ── 意图判定 ──
    lines.append("<details open>")
    lines.append("<summary style='font-weight:600; cursor:pointer;'>🎯 意图判定</summary>")
    lines.append(f"<div style='padding:4px 0 0 12px; font-size:13px;'>")
    lines.append(f"<b>意图：</b>{intent}<br>")
    lines.append(f"<b>置信度：</b>{confidence:.2f}<br>")
    if reason:
        lines.append(f"<b>理由：</b>{reason}<br>")
    if matched:
        lines.append(f"<b>命中词：</b>{', '.join(matched)}<br>")
    lines.append(f"<b>当前模式：</b>{mode_label}")
    lines.append("</div></details>")

    # ── 时间边界 ──
    if time_boundary:
        lines.append("<details>")
        lines.append("<summary style='font-weight:600; cursor:pointer; color:#d4a72c;'>⏳ 时间边界检测</summary>")
        lines.append(f"<div style='padding:4px 0 0 12px; font-size:13px;'>")
        lines.append(f"<b>警告：</b>{time_warning}<br>")
        lines.append(f"<b>改写：</b>{'是' if rewrite_used else '否'}<br>")
        if rewrite_concepts:
            lines.append(f"<b>现代概念：</b>{', '.join(rewrite_concepts)}<br>")
        if rewrite_queries:
            lines.append(f"<b>改写查询：</b>{'; '.join(rewrite_queries)}<br>")
        if rewrite_fallback:
            lines.append(f"<b>回退：</b>{rewrite_fallback}<br>")
        lines.append("</div></details>")

    # ── 检索结果 ──
    if chunk_count > 0:
        lines.append("<details>")
        lines.append(f"<summary style='font-weight:600; cursor:pointer;'>📚 检索结果 ({chunk_count} 条)</summary>")
        lines.append("<div style='padding:4px 0 0 8px; font-size:12px;'>")
        for i, c in enumerate(chunks):
            type_color = "#0969da" if c['type'] == 'venue' else "#8250df" if c['type'] == 'work' else "#1a7f37"
            lines.append(
                f"<div style='margin:4px 0; padding:6px; background:#f6f8fa; "
                f"border-radius:4px; border-left:3px solid {type_color};'>"
                f"<b>[{i+1}]</b> {c['title']}<br>"
                f"<span style='color:#57606a;'>type={c['type']} | dist={c['distance']}</span><br>"
                f"<span style='color:#1f2328;'>{c['snippet'][:80]}...</span>"
                f"</div>"
            )
        lines.append("</div></details>")

    # ── 守卫状态 ──
    guard_color_hex = guard_color.get(guard_status, "#57606a")
    lines.append("<details>")
    lines.append(
        f"<summary style='font-weight:600; cursor:pointer;'>🛡 质量守卫："
        f"<span style='color:{guard_color_hex};'>{guard_status}</span></summary>"
    )
    lines.append(f"<div style='padding:4px 0 0 12px; font-size:13px;'>")
    if guard_status == "PASS":
        lines.append("未触发守卫规则")
    elif guard_status == "WARN":
        lines.append("疑似问题，请人工复查")
    elif guard_status == "FAIL":
        lines.append("守卫拦截，回答可能存在问题")
    if error:
        lines.append(f"<br><b>错误：</b>{error}")
    lines.append("</div></details>")

    # ── 性能 ──
    lines.append("<details>")
    lines.append("<summary style='font-weight:600; cursor:pointer;'>⚡ 性能</summary>")
    lines.append(f"<div style='padding:4px 0 0 12px; font-size:13px;'>")
    elapsed_s = elapsed_ms / 1000 if elapsed_ms else 0
    color = "#2da44e" if elapsed_s < 3 else "#d4a72c" if elapsed_s < 10 else "#cf222e"
    lines.append(f"<b>耗时：</b><span style='color:{color};'>{elapsed_s:.2f}s</span><br>")
    lines.append(f"<b>回答长度：</b>{reply_len} 字<br>")
    if elapsed_s > 30:
        lines.append("<span style='color:#cf222e;'>已触发超时熔断</span>")
    lines.append("</div></details>")

    return "\n".join(lines)


def build_error_luxun(error_msg: str) -> str:
    """将错误信息转为鲁迅口吻的友好提示。"""
    luxun_errors = {
        "timeout": (
            "大约是这问题太过艰深，连我也要想上好一阵子。"
            "不妨换一个简短些的问法，或者待我喘口气再答罢。"
        ),
        "connection": (
            "我与外间世界的联络，似乎出了些故障。"
            "这并非你的过错——大抵是电线那一头睡过去了。稍后再试，如何？"
        ),
        "rate_limit": (
            "今日来问的人实在太多，我纵有三头六臂，也应付不暇。"
            "请稍待片刻再来——我倒是不急，只是怕你等得心焦。"
        ),
    }
    for key, msg in luxun_errors.items():
        if key in error_msg.lower():
            return msg
    return (
        "出了一点意外，我也不知是什么缘故。"
        "大约是我这把老骨头跟不上新式机器的脚步了。请再试一次罢。"
    )


# ---------- 核心回调 ----------

def respond(message, history):
    """处理用户输入，调用 RAG 管线，返回聊天记录、模式标签和调试信息。"""
    if not message or not message.strip():
        return history, current_mode_label(), build_debug_html({}, "")

    pipeline = get_pipeline()

    try:
        reply = pipeline.ask(message.strip())
    except Exception as e:
        err_str = str(e)
        reply = build_error_luxun(err_str)
        pipeline.last_debug["error"] = err_str
        pipeline.last_debug["guard_status"] = "FAIL"

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})

    mode_label = current_mode_label()
    debug_html = build_debug_html(pipeline.last_debug, mode_label)

    return history, mode_label, debug_html


def reset_chat():
    """清空对话历史和管线状态。"""
    reset_pipeline()
    return [], "🔄 对话已重置", build_debug_html({}, "")


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
    .debug-panel {
        font-size: 12px;
        font-family: 'SF Mono', 'Menlo', 'Consolas', monospace;
    }
    .debug-panel details {
        margin-bottom: 4px;
    }
    footer { display: none !important; }
    """

    with gr.Blocks(title="鲁迅数字人智能对话系统 V1.0", css=css) as demo:
        gr.Markdown(
            """
            # 🏛️ 鲁迅数字人智能对话系统 V1.0
            **讲解员模式** · 回答场馆、展品、参观信息 ｜ **数字人模式** · 以鲁迅口吻与你对话
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    value=[],
                    height=520,
                    label="对话",
                    placeholder="输入你的问题开始对话...",
                )
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="输入问题，如「鲁迅先生，您为什么要弃医从文？」",
                        label="", scale=5, container=False,
                    )
                    submit_btn = gr.Button("发送", variant="primary", scale=1)
                with gr.Row():
                    reset_btn = gr.Button("🔄 重置对话", variant="secondary", size="sm")
                    gr.Markdown(
                        "直接输入即可 · 系统自动识别讲解/数字人模式",
                        elem_classes="guide-box",
                    )

            with gr.Column(scale=2):
                gr.Markdown("### 当前模式")
                mode_display = gr.Textbox(
                    value="⏳ 正在加载模型...",
                    label="", interactive=False,
                    elem_classes="mode-display", container=True,
                )
                gr.Markdown("---")
                gr.Markdown("### 🔍 调试面板")
                debug_html = gr.HTML(
                    value="<div style='color:#888; padding:12px; font-style:italic;'>发送一条消息后，调试信息将在此显示。</div>",
                    elem_classes="debug-panel",
                )
                gr.Markdown("---")
                gr.Markdown(
                    """
                    **💡 使用提示**

                    系统会根据你的问题
                    自动判断应该用哪种
                    模式回答你：

                    - 问场馆/展品/参观
                      → 讲解员模式
                    - 问鲁迅生平/作品/
                      思想 → 数字人模式
                    - 既有场馆又有对话
                      → 自动选择

                    **🗣️ 试试这样问**
                    - 这个展厅主要展什么？
                    - 鲁迅先生，您怎么看
                      现在的年轻人？
                    - 您和许广平是怎么
                      认识的？
                    """,
                    elem_classes="guide-box",
                )

        msg.submit(
            respond, [msg, chatbot], [chatbot, mode_display, debug_html]
        ).then(lambda: "", None, [msg])
        submit_btn.click(
            respond, [msg, chatbot], [chatbot, mode_display, debug_html]
        ).then(lambda: "", None, [msg])
        reset_btn.click(reset_chat, None, [chatbot, mode_display, debug_html])
        demo.load(lambda: "✅ 就绪 · 等待输入", None, [mode_display])

    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="鲁迅数字人智能对话系统 V1.0 — Gradio 聊天界面")
    parser.add_argument("--port", type=int, default=7860, help="服务端口（默认 7860）")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="绑定地址")
    parser.add_argument("--share", action="store_true", help="生成公网链接")
    args = parser.parse_args()

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
