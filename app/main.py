from __future__ import annotations

import argparse
import json
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from app.agent.graph import build_graph, run_agent
from app.agent.nodes.answer import answer_stream
from app.agent.nodes.chat import chat_stream
from app.agent.nodes.decide import decide
from app.agent.nodes.retrieve import retrieve
from app.agent.nodes.tool import tool
from app.config import settings
from app.memory.store import get_session


app = FastAPI(title="Agentic RAG System")
_graph = build_graph()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str
    session_id: str | None = None
    debug: bool = False


class AskResponse(BaseModel):
    answer: str
    trace: list[str] = []
    decision: dict | None = None
    documents: list[dict] = []
    session_id: str = "default"
    llm_stats: dict = {}


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": settings.groq_model,
        "use_real_llm": settings.use_real_llm,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    from app.llm.client import get_llm_client
    session = get_session(req.session_id)
    result = run_agent(
        _graph,
        req.question,
        history=session.history_text(),
        memory_summary=session.summary_text(),
    )
    answer_text = result.get("answer", "")
    session.add_turn(req.question, answer_text)

    llm_stats = get_llm_client().stats_dict()

    return AskResponse(
        answer=answer_text,
        trace=result.get("trace", []),
        decision=result.get("decision"),
        documents=result.get("documents", []) if req.debug else [],
        session_id=req.session_id or "default",
        llm_stats=llm_stats,
    )


@app.post("/ask/stream")
async def ask_stream(req: AskRequest) -> StreamingResponse:
    session = get_session(req.session_id)
    history = session.history_text()
    summary = session.summary_text()

    async def event_gen() -> AsyncGenerator[str, None]:
        from app.llm.client import get_llm_client
        from app.agent.state import AgentState

        state: AgentState = {
            "question": req.question,
            "trace": [],
            "history": history,
            "memory_summary": summary,
        }

        # 1. Decide
        dec = decide(state)
        state.update(dec)
        decision = state.get("decision") or {}
        action = decision.get("action", "retrieve")

        if req.debug:
            yield _sse("debug", {
                "type": "decision",
                "decision": decision,
                "llm_stats": get_llm_client().stats_dict(),
            })

        # 2. Retrieve / tool / short-circuit paths
        stream_fn = answer_stream
        if action == "retrieve":
            ret = retrieve(state)
            state.update(ret)
            if req.debug:
                yield _sse("debug", {
                    "type": "retrieval",
                    "n_docs": len(state.get("documents") or []),
                    "docs": [
                        {
                            "source": d.get("source"),
                            "score": round(float(d.get("score", 0)), 3),
                            "snippet": (d.get("text") or "")[:220],
                        }
                        for d in (state.get("documents") or [])
                    ],
                })

        elif action == "tool":
            t_result = tool(state)
            state.update(t_result)
            if req.debug:
                yield _sse("debug", {
                    "type": "tool",
                    "tool_name": decision.get("tool_name"),
                    "tool_args": decision.get("tool_args"),
                    "result_snippet": (
                        (state.get("documents") or [{}])[0].get("text", "")[:400]
                    ),
                })

        elif action == "answer":
          stream_fn = chat_stream

        elif action == "clarify":
            msg = (
                "I need a bit more detail to answer well — could you clarify the topic, "
                "the specific aspect you care about, or some context?"
            )
            session.add_turn(req.question, msg)
            yield _sse("token", {"text": msg})
            yield _sse("done", {"trace": state.get("trace", []), "llm_stats": {}})
            return

        elif action == "refuse":
            msg = (
                "I can't help with that request — it touches on sensitive credentials "
                "or content outside this assistant's scope. Please rephrase if you meant something else."
            )
            session.add_turn(req.question, msg)
            yield _sse("token", {"text": msg})
            yield _sse("done", {"trace": state.get("trace", []), "llm_stats": {}})
            return

        # 3. Stream answer (stream_fn handles trace append)
        full_answer: list[str] = []
        for chunk in stream_fn(state):
            full_answer.append(chunk)
            yield _sse("token", {"text": chunk})

        final = "".join(full_answer)
        session.add_turn(req.question, final)
        yield _sse("done", {
            "trace": state.get("trace", []),
            "llm_stats": get_llm_client().stats_dict(),
        })

    return StreamingResponse(event_gen(), media_type="text/event-stream")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


# ---------------------------------------------------------------------------
# Chat UI
# ---------------------------------------------------------------------------

_UI = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agentic RAG — cs.AI</title>
<style>
:root{
  --bg:#0b0d14;--surface:#13151f;--surface2:#1c1f2e;--border:#252840;
  --accent:#7c3aed;--accent2:#a78bfa;--green:#10b981;--red:#ef4444;
  --yellow:#f59e0b;--blue:#3b82f6;--text:#e2e8f0;--muted:#64748b;--muted2:#94a3b8;
  --radius:10px;--font:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:var(--bg);color:var(--text);font-family:var(--font);font-size:14px}
body{display:flex;flex-direction:column;height:100vh;overflow:hidden}

/* ── TOP BAR ── */
#topbar{
  display:flex;align-items:center;gap:10px;padding:10px 18px;
  background:var(--surface);border-bottom:1px solid var(--border);flex-shrink:0;
}
#topbar .logo{font-weight:700;font-size:15px;color:var(--accent2);letter-spacing:-.3px}
.chip{
  font-size:11px;padding:2px 9px;border-radius:99px;font-weight:500;
  background:var(--surface2);border:1px solid var(--border);color:var(--muted2);
}
.chip.live{border-color:var(--green);color:var(--green)}
.chip.model{border-color:#4f46e5;color:var(--accent2)}
#topbar-right{margin-left:auto;display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.toggle-row{display:flex;align-items:center;gap:5px;font-size:12px;color:var(--muted2);cursor:pointer}
.toggle-row input[type=checkbox]{width:14px;height:14px;accent-color:var(--accent);cursor:pointer}
.session-input{
  background:var(--surface2);border:1px solid var(--border);color:var(--text);
  border-radius:6px;padding:3px 9px;font-size:12px;width:110px;outline:none;
}
.session-input:focus{border-color:var(--accent)}

/* ── LAYOUT ── */
#main{display:flex;flex:1;overflow:hidden}

/* ── CHAT COLUMN ── */
#chat-col{flex:1;display:flex;flex-direction:column;overflow:hidden;min-width:0}
#chat-area{flex:1;overflow-y:auto;padding:20px 24px;display:flex;flex-direction:column;gap:14px;scroll-behavior:smooth}

/* ── MESSAGES ── */
.msg-wrap{display:flex;flex-direction:column;gap:4px}
.msg-wrap.user{align-items:flex-end}
.msg-wrap.agent{align-items:flex-start}
.bubble{
  max-width:680px;padding:11px 15px;border-radius:var(--radius);
  line-height:1.6;white-space:pre-wrap;word-break:break-word;font-size:13.5px;
}
.bubble.user{background:var(--accent);color:#fff}
.bubble.agent{background:var(--surface2);border:1px solid var(--border)}
.bubble.agent.streaming::after{content:'▍';animation:blink .7s steps(1) infinite}
@keyframes blink{50%{opacity:0}}
.msg-meta{font-size:11px;color:var(--muted);padding:0 4px}

/* ── INPUT ROW ── */
#input-row{
  display:flex;align-items:flex-end;gap:8px;
  padding:12px 18px;background:var(--surface);border-top:1px solid var(--border);flex-shrink:0;
}
#q{
  flex:1;background:var(--surface2);border:1px solid var(--border);color:var(--text);
  border-radius:var(--radius);padding:11px 14px;font-size:13.5px;resize:none;
  height:46px;max-height:140px;font-family:var(--font);outline:none;line-height:1.5;
  transition:border .15s;
}
#q:focus{border-color:var(--accent)}
#send{
  background:var(--accent);color:#fff;border:none;border-radius:var(--radius);
  padding:0 20px;height:46px;cursor:pointer;font-size:13.5px;font-weight:600;
  white-space:nowrap;transition:background .15s;flex-shrink:0;
}
#send:hover{background:#6d28d9}
#send:disabled{background:var(--surface2);color:var(--muted);cursor:default}

/* ── DEBUG PANEL (right side) ── */
#debug-col{
  width:340px;flex-shrink:0;display:flex;flex-direction:column;
  border-left:1px solid var(--border);background:var(--surface);overflow:hidden;
  transition:width .2s;
}
#debug-col.hidden{width:0;border:none}
#debug-header{
  padding:10px 14px;font-size:12px;font-weight:600;color:var(--accent2);
  border-bottom:1px solid var(--border);letter-spacing:.5px;text-transform:uppercase;
  display:flex;align-items:center;justify-content:space-between;flex-shrink:0;
}
#debug-body{flex:1;overflow-y:auto;padding:10px 12px;display:flex;flex-direction:column;gap:10px}

/* ── DEBUG CARDS ── */
.dcard{
  background:var(--surface2);border:1px solid var(--border);border-radius:8px;
  padding:10px 12px;font-family:'Menlo','Consolas',monospace;font-size:11.5px;
}
.dcard-title{
  font-size:10px;font-weight:700;letter-spacing:.8px;text-transform:uppercase;
  margin-bottom:6px;
}
.dcard-title.decision{color:var(--accent2)}
.dcard-title.retrieval{color:var(--blue)}
.dcard-title.tool{color:var(--yellow)}
.dcard-title.trace{color:var(--green)}
.dcard-title.stats{color:var(--muted2)}
.kv{display:flex;gap:6px;margin-bottom:3px;align-items:flex-start}
.kv .k{color:var(--muted);flex-shrink:0;min-width:80px}
.kv .v{color:var(--text);word-break:break-all}
.action-badge{
  display:inline-block;padding:1px 8px;border-radius:4px;font-weight:700;
  font-size:10px;text-transform:uppercase;letter-spacing:.5px;
}
.a-retrieve{background:#1e3a5f;color:#93c5fd}
.a-tool{background:#3b2200;color:#fcd34d}
.a-clarify{background:#1a2e1a;color:#86efac}
.a-refuse{background:#3b0f0f;color:#fca5a5}
.a-answer{background:#2d1b4e;color:#c4b5fd}
.doc-item{margin-top:6px;padding-top:6px;border-top:1px solid var(--border)}
.doc-score{color:var(--green);font-weight:600}
.doc-source{color:var(--muted)}
.doc-snippet{color:var(--muted2);margin-top:2px;line-height:1.45}
.trace-pills{display:flex;flex-wrap:wrap;gap:4px;margin-top:4px}
.tpill{
  background:var(--surface);border:1px solid var(--border);
  color:var(--muted2);border-radius:4px;padding:1px 7px;font-size:10.5px;
}
.strict-badge{
  display:inline-block;padding:1px 6px;border-radius:4px;font-size:10px;
  font-weight:700;background:#14532d;color:#86efac;margin-left:4px;
}
.mock-badge{
  display:inline-block;padding:1px 6px;border-radius:4px;font-size:10px;
  font-weight:700;background:#3b0f0f;color:#fca5a5;margin-left:4px;
}

/* ── REASONING BOX ── */
.reasoning{
  margin-top:6px;padding:6px 8px;background:var(--bg);border-radius:6px;
  color:var(--muted2);font-style:italic;font-size:11px;line-height:1.5;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}

/* ── WELCOME CARDS ── */
.welcome{display:flex;flex-direction:column;gap:10px;max-width:560px;margin:auto;padding:30px 0}
.welcome h2{font-size:22px;font-weight:700;color:var(--accent2)}
.welcome p{color:var(--muted2);line-height:1.6}
.example-chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:4px}
.echip{
  background:var(--surface2);border:1px solid var(--border);border-radius:6px;
  padding:5px 12px;font-size:12px;color:var(--muted2);cursor:pointer;
  transition:border-color .15s,color .15s;
}
.echip:hover{border-color:var(--accent);color:var(--text)}

/* ── STATUS BAR (bottom) ── */
#statusbar{
  font-size:11px;color:var(--muted);padding:4px 18px;
  background:var(--surface);border-top:1px solid var(--border);
  display:flex;gap:16px;flex-shrink:0;
}
#statusbar span{color:var(--muted2)}
</style>
</head>
<body>

<!-- TOP BAR -->
<div id="topbar">
  <span class="logo">Agentic RAG</span>
  <span class="chip model" id="model-chip">…</span>
  <span class="chip" id="mode-chip">mock</span>
  <span class="chip">arXiv cs.AI</span>
  <div id="topbar-right">
    <label class="toggle-row">
      <input type="checkbox" id="debug-toggle"> Debug panel
    </label>
    <label class="toggle-row">
      <input type="checkbox" id="stream-toggle" checked> Streaming
    </label>
    <label class="toggle-row" style="gap:6px">
      <span>Session</span>
      <input type="text" class="session-input" id="session-id" value="default" placeholder="session id">
    </label>
  </div>
</div>

<!-- MAIN -->
<div id="main">
  <!-- CHAT -->
  <div id="chat-col">
    <div id="chat-area">
      <div class="welcome" id="welcome">
        <h2>Agentic RAG System</h2>
        <p>Answers questions over <strong>arXiv cs.AI papers</strong> using a LangGraph agent that decides when to retrieve, use a tool, clarify, or refuse.</p>
        <p style="margin-top:8px">Try an example:</p>
        <div class="example-chips">
          <span class="echip" onclick="fillQ('What is retrieval augmented generation?')">What is RAG?</span>
          <span class="echip" onclick="fillQ('How does transformer attention work?')">Transformer attention</span>
          <span class="echip" onclick="fillQ('Search arxiv for diffusion model papers')">arXiv search</span>
          <span class="echip" onclick="fillQ('What is (128 * 3) + 42?')">Calculator</span>
          <span class="echip" onclick="fillQ('tell me more')">Clarify trigger</span>
          <span class="echip" onclick="fillQ('What is my password?')">Refuse trigger</span>
        </div>
        <p style="margin-top:10px;font-size:12px;color:var(--muted)">
          Enable <strong>Debug panel</strong> to see routing decisions, retrieved docs, LLM stats, and trace.
        </p>
      </div>
    </div>
    <div id="input-row">
      <textarea id="q" placeholder="Ask about cs.AI papers…" rows="1"></textarea>
      <button id="send">Send</button>
    </div>
  </div>

  <!-- DEBUG PANEL -->
  <div id="debug-col" class="hidden">
    <div id="debug-header">
      <span>Debug</span>
      <span id="debug-turn" style="font-weight:400;color:var(--muted)">—</span>
    </div>
    <div id="debug-body"></div>
  </div>
</div>

<!-- STATUS BAR -->
<div id="statusbar">
  <span>Model: <span id="sb-model">—</span></span>
  <span>Mode: <span id="sb-mode">—</span></span>
  <span>Latency: <span id="sb-latency">—</span></span>
  <span>Tokens: <span id="sb-tokens">—</span></span>
  <span>Structured output: <span id="sb-structured">—</span></span>
</div>

<script>
// ── State ──────────────────────────────────────────────────────────────────
let turnCount = 0;
const qEl = document.getElementById('q');
const sendBtn = document.getElementById('send');
const chatArea = document.getElementById('chat-area');
const debugCol = document.getElementById('debug-col');
const debugBody = document.getElementById('debug-body');
const debugToggle = document.getElementById('debug-toggle');
const streamToggle = document.getElementById('stream-toggle');
const sessionEl = document.getElementById('session-id');
const welcomeEl = document.getElementById('welcome');

// ── Init: fetch server info ────────────────────────────────────────────────
fetch('/health').then(r=>r.json()).then(d=>{
  document.getElementById('model-chip').textContent = (d.model||'').split('/').pop();
  document.getElementById('sb-model').textContent = d.model||'—';
  const live = d.use_real_llm;
  const modeChip = document.getElementById('mode-chip');
  modeChip.textContent = live ? 'live API' : 'mock mode';
  if(live) modeChip.classList.add('live');
  document.getElementById('sb-mode').textContent = live ? 'real LLM' : 'mock (set USE_REAL_LLM=true)';
}).catch(()=>{});

// ── Debug panel toggle ─────────────────────────────────────────────────────
debugToggle.addEventListener('change', ()=>{
  debugCol.classList.toggle('hidden', !debugToggle.checked);
});

// ── Input auto-resize ──────────────────────────────────────────────────────
qEl.addEventListener('input', ()=>{
  qEl.style.height = '46px';
  qEl.style.height = Math.min(qEl.scrollHeight, 140) + 'px';
});
qEl.addEventListener('keydown', e=>{
  if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); send(); }
});

function fillQ(text){ qEl.value=text; qEl.focus(); }

// ── DOM helpers ─────────────────────────────────────────────────────────────
function addMsg(role, text, extra=''){
  const wrap = document.createElement('div');
  wrap.className = `msg-wrap ${role}`;
  const bubble = document.createElement('div');
  bubble.className = `bubble ${role} ${extra}`;
  bubble.textContent = text;
  wrap.appendChild(bubble);
  chatArea.appendChild(wrap);
  chatArea.scrollTop = chatArea.scrollHeight;
  return bubble;
}

function addDCard(html){
  const el = document.createElement('div');
  el.className = 'dcard';
  el.innerHTML = html;
  debugBody.appendChild(el);
  debugBody.scrollTop = debugBody.scrollHeight;
  return el;
}

function clearDebug(){
  debugBody.innerHTML = '';
}

function actionBadge(action){
  const cls = {retrieve:'a-retrieve',tool:'a-tool',clarify:'a-clarify',
                refuse:'a-refuse',answer:'a-answer'}[action]||'a-answer';
  return `<span class="action-badge ${cls}">${action}</span>`;
}

function updateStatusBar(stats){
  if(!stats||!stats.model) return;
  document.getElementById('sb-latency').textContent =
    stats.latency_ms ? stats.latency_ms.toFixed(0)+'ms' : '—';
  const tok = stats.tokens||{};
  document.getElementById('sb-tokens').textContent =
    tok.total ? `${tok.total} (↑${tok.prompt} ↓${tok.completion})` : '—';
  document.getElementById('sb-structured').textContent =
    stats.structured_output ? '✓ strict schema' : (stats.streaming ? '— (streaming)' : '—');
}

function renderDecisionCard(dec){
  if(!dec) return;
  const strict = dec._strict_schema
    ? '<span class="strict-badge">STRICT SCHEMA</span>'
    : (dec._mock ? '<span class="mock-badge">MOCK</span>' : '');
  let html = `<div class="dcard-title decision">⚡ Decision ${strict}</div>`;
  html += `<div class="kv"><span class="k">action</span><span class="v">${actionBadge(dec.action)}</span></div>`;
  if(dec.query) html += `<div class="kv"><span class="k">query</span><span class="v">${esc(dec.query)}</span></div>`;
  if(dec.tool_name){
    html += `<div class="kv"><span class="k">tool</span><span class="v">${esc(dec.tool_name)}</span></div>`;
    if(dec.tool_args && Object.keys(dec.tool_args).length)
      html += `<div class="kv"><span class="k">args</span><span class="v">${esc(JSON.stringify(dec.tool_args))}</span></div>`;
  }
  if(dec._reasoning)
    html += `<div class="reasoning">💭 ${esc(dec._reasoning)}</div>`;
  addDCard(html);
}

function renderRetrievalCard(data){
  let html = `<div class="dcard-title retrieval">🔍 Retrieval — ${data.n_docs} doc${data.n_docs!==1?'s':''}</div>`;
  if(!data.docs||!data.docs.length){
    html += `<div style="color:var(--muted);font-size:11px">No documents found — run ingestion first.</div>`;
  } else {
    data.docs.forEach((d,i)=>{
      html += `<div class="doc-item">
        <div><span class="doc-score">score ${d.score}</span> &nbsp;
             <span class="doc-source">${esc(d.source||'')}</span></div>
        <div class="doc-snippet">${esc((d.snippet||'').trim())}</div>
      </div>`;
    });
  }
  addDCard(html);
}

function renderToolCard(data){
  let html = `<div class="dcard-title tool">🔧 Tool — ${esc(data.tool_name||'')}</div>`;
  if(data.tool_args) html += `<div class="kv"><span class="k">args</span><span class="v">${esc(JSON.stringify(data.tool_args))}</span></div>`;
  if(data.result_snippet)
    html += `<div class="reasoning">${esc(data.result_snippet.trim())}</div>`;
  addDCard(html);
}

function renderTraceCard(trace, llmStats){
  let html = `<div class="dcard-title trace">🗺 Trace</div>`;
  html += `<div class="trace-pills">${(trace||[]).map(s=>`<span class="tpill">${s}</span>`).join('')}</div>`;
  if(llmStats && llmStats.model){
    html += `<div style="margin-top:8px">`;
    if(llmStats.latency_ms)
      html += `<div class="kv"><span class="k">latency</span><span class="v">${llmStats.latency_ms}ms</span></div>`;
    if(llmStats.tokens && llmStats.tokens.total)
      html += `<div class="kv"><span class="k">tokens</span><span class="v">↑${llmStats.tokens.prompt} ↓${llmStats.tokens.completion}</span></div>`;
    html += `<div class="kv"><span class="k">structured</span><span class="v">${llmStats.structured_output?'✓ strict schema':'—'}</span></div>`;
    html += `<div class="kv"><span class="k">streaming</span><span class="v">${llmStats.streaming?'✓':llmStats.latency_ms?'—':'n/a'}</span></div>`;
    html += `</div>`;
    updateStatusBar(llmStats);
  }
  addDCard(html);
}

function esc(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

// ── Send ───────────────────────────────────────────────────────────────────
async function send(){
  const question = qEl.value.trim();
  if(!question) return;
  qEl.value = ''; qEl.style.height = '46px';
  sendBtn.disabled = true;

  if(welcomeEl){ welcomeEl.remove(); }

  addMsg('user', question);
  clearDebug();
  turnCount++;
  document.getElementById('debug-turn').textContent = `Turn ${turnCount}`;

  try {
    if(streamToggle.checked){
      await sendStream(question);
    } else {
      await sendBlocking(question);
    }
  } finally {
    sendBtn.disabled = false;
    qEl.focus();
  }
}

// ── Streaming ──────────────────────────────────────────────────────────────
async function sendStream(question){
  const bubble = addMsg('agent', '', 'streaming');
  let full = '';

  const resp = await fetch('/ask/stream', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({
      question,
      session_id: sessionEl.value||'default',
      debug: debugToggle.checked,
    }),
  });

  if(!resp.ok){ bubble.textContent='Error '+resp.status; bubble.classList.remove('streaming'); return; }

  const reader = resp.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  let lastEvent = '';

  while(true){
    const {done,value} = await reader.read();
    if(done) break;
    buf += dec.decode(value, {stream:true});
    const lines = buf.split('\n');
    buf = lines.pop();

    for(const line of lines){
      if(line.startsWith('event: ')){ lastEvent = line.slice(7).trim(); continue; }
      if(!line.startsWith('data: ')) continue;
      let payload;
      try{ payload = JSON.parse(line.slice(6)); } catch{ continue; }

      if(lastEvent==='token'){
        full += payload.text||'';
        bubble.textContent = full;
        chatArea.scrollTop = chatArea.scrollHeight;
      } else if(lastEvent==='debug'){
        if(payload.type==='decision') renderDecisionCard(payload.decision);
        else if(payload.type==='retrieval') renderRetrievalCard(payload);
        else if(payload.type==='tool') renderToolCard(payload);
      } else if(lastEvent==='done'){
        bubble.classList.remove('streaming');
        if(debugToggle.checked) renderTraceCard(payload.trace, payload.llm_stats);
        updateStatusBar(payload.llm_stats||{});
      }
    }
  }
  bubble.classList.remove('streaming');
}

// ── Blocking ───────────────────────────────────────────────────────────────
async function sendBlocking(question){
  const bubble = addMsg('agent', '…');
  const resp = await fetch('/ask', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({
      question,
      session_id: sessionEl.value||'default',
      debug: true,
    }),
  });
  const data = await resp.json();
  bubble.textContent = data.answer||'(no answer)';

  if(debugToggle.checked){
    renderDecisionCard(data.decision);
    if(data.documents&&data.documents.length){
      renderRetrievalCard({
        n_docs: data.documents.length,
        docs: data.documents.map(d=>({
          source: d.source, score: d.score,
          snippet: (d.text||'').slice(0,220),
        })),
      });
    }
    renderTraceCard(data.trace, data.llm_stats);
  }
  updateStatusBar(data.llm_stats||{});
}

sendBtn.addEventListener('click', send);
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def ui() -> HTMLResponse:
    return HTMLResponse(_UI)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agentic RAG CLI")
    parser.add_argument("question", nargs="?", help="One-shot question")
    parser.add_argument("--session", default="cli")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--show-trace", action="store_true")
    args = parser.parse_args(argv)

    session = get_session(args.session)

    def _ask(q: str) -> dict:
        r = run_agent(_graph, q, history=session.history_text(), memory_summary=session.summary_text())
        session.add_turn(q, r.get("answer", ""))
        return r

    if args.chat or not args.question:
        print("Agentic RAG chat — type 'exit' to quit.")
        while True:
            try:
                q = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print(); return 0
            if not q: continue
            if q.lower() in {"exit","quit"}: return 0
            r = _ask(q)
            print(f"agent> {r.get('answer','')}")
            if args.show_trace:
                print(f"  trace:    {r.get('trace')}")
                print(f"  decision: {r.get('decision')}")
        return 0

    r = _ask(args.question)
    print(r["answer"])
    if args.show_trace:
        print(f"trace: {r.get('trace')}")
        print(f"decision: {r.get('decision')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
