import os
import queue
import sys
import threading
import time
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from src.agent.llm import OpenAICompatLLM
from src.agent.loop import run_agent
from src.agent.session import load_history, save_history
from src.agent.findings import write_findings

SESSIONS_DIR = ROOT / "artifacts" / "sessions"
EXAMPLES = [
    "Forecast KO and DELL each day for 7 days with kronos_mini, 95% interval, line chart",
    "Compare naive, drift, arima and kronos_mini for KO and DELL",
    "Compare across kronos variants for DELL",
    "How risky is KO over the next 5 days?",
]
CSS = """
.gradio-container {max-width: 1400px !important}
#title {margin-bottom: 0}
#subtitle {color: #888; margin-top: 0}
"""


# ---------- helpers ----------
def list_sessions():
    names = {"default"}
    if SESSIONS_DIR.exists():
        names |= {p.stem for p in SESSIONS_DIR.iterdir() if p.is_file() or p.is_dir()}
    return sorted(names)


def history_to_chat(session):
    """Rebuild the visible chat from saved history (user and assistant text only)."""
    try:
        msgs = load_history((session or "default").strip() or "default") or []
    except Exception:
        return []
    return [{"role": m["role"], "content": m["content"]}
            for m in msgs
            if m.get("role") in ("user", "assistant")
            and isinstance(m.get("content"), str) and m["content"].strip()]


def tool_msg(tool, body, status):
    return {"role": "assistant", "content": body,
            "metadata": {"title": f"🔧 {tool}", "status": status}}


def status_msg(text):
    return {"role": "assistant", "content": text,
            "metadata": {"title": "🧠 Working", "status": "pending"}}


# ---------- main handler ----------
def respond(message, chat, session):
    """Outputs: chatbot, textbox, gallery, tables, run info, session dropdown."""
    keep = gr.update()
    message = (message or "").strip()
    chat = chat or []
    if not message:
        yield chat, "", keep, keep, keep, keep
        return

    session = (session or "").strip() or "default"
    base = chat + [{"role": "user", "content": message}]
    steps, status = [], status_msg("Starting...")
    render = lambda: base + steps + ([status] if status else [])

    # clear the side panels for the new run
    yield render(), "", gr.update(value=[]), "", "Running...", keep

    events: queue.Queue = queue.Queue()
    out: dict = {}

    def worker():
        try:
            out["r"] = run_agent(
                message, OpenAICompatLLM(),
                on_event=lambda event, **k: events.put((event, k)),
                history=load_history(session),
            )
        except Exception as e:
            out["err"] = e
        finally:
            events.put(None)

    threading.Thread(target=worker, daemon=True).start()

    t0 = time.time()
    open_steps = {}                       # tool name -> (index in steps, start time)
    step_text = "Starting..."
    open_steps = {}
    while True:
        try:
            item = events.get(timeout=1)
        except queue.Empty:  # no event: refresh the timer
            status = status_msg(f"{step_text} ({time.time() - t0:.0f}s)")
            yield render(), "", keep, keep, f"Running... {time.time() - t0:.0f}s", keep
            continue
        if item is None:
            break
        ev, k = item
        if ev == "thinking":
            step_text = f"Step {k.get('step')}/{k.get('max_steps')}: thinking..."
            status = status_msg(f"{step_text} ({time.time() - t0:.0f}s)")
        elif ev == "tool_start":
            tool = k.get("tool")
            steps.append(tool_msg(tool, "running...", "pending"))
            open_steps[tool] = (len(steps) - 1, time.time())
        elif ev == "tool_end":
            tool = k.get("tool")
            ok = k.get("ok") in (True, None)
            if tool in open_steps:
                i, ts = open_steps.pop(tool)
                steps[i] = tool_msg(tool, f"{'ok' if ok else 'FAILED'} in {time.time() - ts:.1f}s", "done")
                if not ok:
                    steps[i]["metadata"]["title"] = f"❌ {tool}"
        else:
            continue
        yield render(), "", keep, keep, f"Running... {time.time() - t0:.0f}s", keep

    status = None
    if "err" in out:
        base.append({"role": "assistant", "content": f"**Error:** {out['err']}",
                     "metadata": {"title": "❌ Error"}})
        yield render(), "", keep, keep, "Failed", keep
        return

    r = out["r"]
    save_history(session, r["messages"])
    base.append({"role": "assistant",
                 "content": r["answer"] or "*The agent returned an empty answer. Try rephrasing.*"})
    info = (f"**Status:** {r['status']}  \n**Run:** `{r['run_id']}`  \n"
            f"**Steps:** {r['steps']}  \n**Session:** `{session}`  \n"
            f"**Time:** {time.time() - t0:.0f}s")
    yield render(), "", keep, keep, info, gr.update(choices=list_sessions(), value=session)

    try:
        md, tables_md, charts = write_findings(r, message)
        gallery = [(str(p), p.stem.replace("_", " ")) for p in charts]
        info += f"  \n**Saved:** `{md.relative_to(ROOT)}`"
        yield render(), "", gr.update(value=gallery), tables_md or "*No tables this turn.*", info, keep
    except Exception as e:
        yield render(), "", keep, f"Findings failed: {e}", info, keep


def shutdown():
    threading.Timer(1.0, lambda: os._exit(0)).start()


CLOSE_TAB_JS = """
() => {
    window.close();
    document.body.innerHTML =
        '<h2 style="font-family:sans-serif;text-align:center;margin-top:20vh">' +
        'Session ended. Server stopped. You can close this tab.</h2>';
}
"""

# ---------- layout ----------
with gr.Blocks(title="Quant Research Agent") as demo:
    gr.Markdown("# Quant Research Agent", elem_id="title")
    gr.Markdown("Daily stock return forecasts and model comparison. "
                "Research only, not investment advice.", elem_id="subtitle")

    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=620, show_label=False, label="Chat")
            with gr.Row():
                msg = gr.Textbox(placeholder="Ask a question, then press Enter",
                                 show_label=False, lines=1, scale=8, autofocus=True)
                send = gr.Button("Send", variant="primary", scale=1)
            gr.Examples(examples=[[e] for e in EXAMPLES], inputs=msg, label="Try these")

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("Charts"):
                    gallery = gr.Gallery(show_label=False, columns=2, height=620,
                                         object_fit="contain", preview=False)
                with gr.Tab("Tables"):
                    tables = gr.Markdown("*Tables appear here after a run.*")
                with gr.Tab("Run info"):
                    info = gr.Markdown("*No run yet.*")
            session = gr.Dropdown(choices=list_sessions(), value="default",
                                  allow_custom_value=True,
                                  label="Session (type a new name for a new conversation)")
            with gr.Row():
                clear = gr.Button("Clear display")
                quit_btn = gr.Button("End & close", variant="stop")

    inputs = [msg, chatbot, session]
    outputs = [chatbot, msg, gallery, tables, info, session]
    msg.submit(respond, inputs, outputs)
    send.click(respond, inputs, outputs)
    session.change(history_to_chat, session, chatbot)
    clear.click(lambda: ([], [], "*Tables appear here after a run.*", "*No run yet.*"),
                None, [chatbot, gallery, tables, info])
    quit_btn.click(shutdown, None, None).then(None, None, None, js=CLOSE_TAB_JS)

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True,
                        allowed_paths=[str(ROOT / "results")],
                        theme=gr.themes.Soft(), css=CSS)

