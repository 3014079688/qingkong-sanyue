"""私人 AI 助手 —— 应用层，全部手写，可以逐行改。

  浏览器界面 (static/index.html)
        |  HTTP + SSE
  本文件  <-- 你正在看的这一层：上下文拼接、记忆、知识库、流式输出
        |  HTTP
  Ollama (127.0.0.1:11434)  <-- 推理引擎，负责把模型权重跑起来

想脱离 Ollama、用 transformers 自己写 generate 循环？
只需要替换 ollama_chat() 和 ollama_embed() 两个函数，其余代码一行都不用动。

启动：python server.py
"""

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "_data"
DATA.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA / "chat.db"
SETTINGS_PATH = DATA / "settings.json"

# Ollama 的地址。改了端口就改这里，或者设环境变量 OLLAMA_HOST
OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")

DEFAULT_SYSTEM = """你是一个运行在用户本机的私人 AI 助手。

要求：
- 用中文回答，直接给结论，不铺垫、不复述问题。
- 不确定就说不确定，不要编造事实、数据或引用。
- 涉及代码时给可运行的完整片段，并说明放在哪个文件里。
- 回答简短优先；用户明确要求详细时才展开。
- 你能看到对话历史，也会收到从本地知识库检索到的资料。若资料与问题相关就以资料为准，并指出出自哪个文件。"""

DEFAULT_SETTINGS = {
    "model": "qwen2.5:3b",
    "embed_model": "nomic-embed-text",
    "system": DEFAULT_SYSTEM,
    "use_kb": True,
    "temperature": 0.7,
    "history_msgs": 24,   # 最多带多少条历史进上下文
    "top_k": 4,           # 知识库检索返回几块
    "chunk_size": 400,    # 文档切块大小（字符）
    "chunk_overlap": 100,
}


# ---------------------------------------------------------------- 存储

@contextmanager
def conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init_db():
    with conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                created REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                ts REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS kb(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                idx INTEGER NOT NULL,
                text TEXT NOT NULL,
                vec TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_msg_sess ON messages(session_id);
            """
        )


init_db()


def load_settings() -> dict:
    if SETTINGS_PATH.exists():
        try:
            saved = json.loads(SETTINGS_PATH.read_text("utf-8"))
            return {**DEFAULT_SETTINGS, **saved}
        except Exception:
            pass
    return dict(DEFAULT_SETTINGS)


def save_settings(s: dict) -> None:
    settings = load_settings()
    for k in DEFAULT_SETTINGS:
        if k in s and s[k] is not None:
            settings[k] = s[k]
    SETTINGS_PATH.write_text(json.dumps(settings, ensure_ascii=False, indent=2), "utf-8")


# ---------------------------------------------------------------- 推理层（可替换）

async def ollama_models():
    """返回 Ollama 已下载的模型名列表；连不上返回 None。"""
    try:
        async with httpx.AsyncClient(timeout=5) as cli:
            r = await cli.get(f"{OLLAMA}/api/tags")
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return None


async def ollama_chat(messages, model, temperature) -> AsyncIterator[tuple[str, str]]:
    """调 Ollama 聊天接口，逐个 token 吐出 (类型, 文本)。

    这就是「推理」的全部：把消息喂进去，把 token 一个个接出来。
    换成 transformers 的话，这里是 model.generate() + TextIteratorStreamer。
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"temperature": temperature, "num_ctx": 4096},
    }
    # qwen3 默认会先输出一段思考过程，这里关掉，速度更快
    if model.startswith("qwen3"):
        payload["think"] = False

    async with httpx.AsyncClient(timeout=None) as cli:
        async with cli.stream("POST", f"{OLLAMA}/api/chat", json=payload) as r:
            if r.status_code >= 400:
                body = await r.aread()
                raise RuntimeError(f"Ollama 返回 {r.status_code}：{body[:300].decode('utf-8', 'ignore')}")
            async for line in r.aiter_lines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = obj.get("message") or {}
                if msg.get("thinking"):
                    yield "thinking", msg["thinking"]
                if msg.get("content"):
                    yield "content", msg["content"]
                if obj.get("done"):
                    return


async def ollama_embed(texts: list[str], model: str) -> list[list[float]]:
    async with httpx.AsyncClient(timeout=180) as cli:
        r = await cli.post(f"{OLLAMA}/api/embed", json={"model": model, "input": texts})
        r.raise_for_status()
        return r.json()["embeddings"]


# ---------------------------------------------------------------- 知识库

def extract_text(filename: str, raw: bytes) -> str:
    low = filename.lower()
    if low.endswith(".pdf"):
        try:
            import pypdf  # 可选依赖：pip install pypdf
        except ImportError:
            raise HTTPException(400, "读 PDF 需要先装 pypdf：pip install pypdf")
        import io

        reader = pypdf.PdfReader(io.BytesIO(raw))
        return "\n\n".join((p.extract_text() or "") for p in reader.pages)
    for enc in ("utf-8", "gbk", "utf-16"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "ignore")


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """按段落切块，块之间留一点重叠，避免一句话被拦腰切断。"""
    text = "\n".join(line.strip() for line in text.splitlines())
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 1 <= size:
            buf = f"{buf}\n{p}" if buf else p
        else:
            if buf:
                chunks.append(buf)
            if len(p) <= size:
                buf = p
            else:
                for i in range(0, len(p), size - overlap):
                    piece = p[i:i + size]
                    if piece.strip():
                        chunks.append(piece)
                buf = ""
    if buf:
        chunks.append(buf)
    return [c for c in chunks if len(c.strip()) > 10]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb + 1e-9)


async def retrieve(query: str, k: int, embed_model: str) -> list[dict]:
    with conn() as c:
        rows = c.execute("SELECT source, text, vec FROM kb").fetchall()
    if not rows:
        return []
    try:
        qv = (await ollama_embed([query], embed_model))[0]
    except Exception:
        return []  # 向量模型没装就先退化成不检索，不阻断对话
    scored = []
    for r in rows:
        try:
            v = json.loads(r["vec"])
        except Exception:
            continue
        if len(v) != len(qv):
            continue
        scored.append((cosine(qv, v), r["source"], r["text"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"score": round(s, 3), "source": src, "text": txt} for s, src, txt in scored[:k]]


# ---------------------------------------------------------------- 服务

app = FastAPI(title="私人 AI 助手")


class ChatIn(BaseModel):
    session_id: int
    text: str


class SessionIn(BaseModel):
    title: str = "新对话"


class SettingsIn(BaseModel):
    model: str | None = None
    embed_model: str | None = None
    system: str | None = None
    use_kb: bool | None = None
    temperature: float | None = None
    history_msgs: int | None = None
    top_k: int | None = None


@app.get("/")
async def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
async def health():
    models = await ollama_models()
    s = load_settings()
    return {
        "ollama": models is not None,
        "models": models or [],
        "active": s["model"],
        "embed_ready": bool(models and any(m.startswith(s["embed_model"]) for m in models)),
    }


@app.get("/api/settings")
async def get_settings():
    return load_settings()


@app.post("/api/settings")
async def post_settings(body: SettingsIn):
    save_settings(body.model_dump(exclude_none=True))
    return load_settings()


@app.get("/api/sessions")
async def list_sessions():
    with conn() as c:
        rows = c.execute(
            "SELECT s.id, s.title, s.created, "
            "(SELECT COUNT(*) FROM messages m WHERE m.session_id=s.id) AS n "
            "FROM sessions s ORDER BY s.id DESC"
        ).fetchall()
    return [dict(r) for r in rows]


@app.post("/api/sessions")
async def new_session(body: SessionIn):
    with conn() as c:
        cur = c.execute("INSERT INTO sessions(title, created) VALUES(?,?)", (body.title, time.time()))
        sid = cur.lastrowid
    return {"id": sid, "title": body.title}


@app.delete("/api/sessions/{sid}")
async def del_session(sid: int):
    with conn() as c:
        c.execute("DELETE FROM messages WHERE session_id=?", (sid,))
        c.execute("DELETE FROM sessions WHERE id=?", (sid,))
    return {"ok": True}


@app.get("/api/sessions/{sid}/messages")
async def get_messages(sid: int):
    with conn() as c:
        rows = c.execute(
            "SELECT role, content, ts FROM messages WHERE session_id=? ORDER BY id", (sid,)
        ).fetchall()
    return [dict(r) for r in rows]


@app.post("/api/chat")
async def chat(body: ChatIn):
    s = load_settings()
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "消息是空的")

    with conn() as c:
        sess = c.execute("SELECT id FROM sessions WHERE id=?", (body.session_id,)).fetchone()
        if not sess:
            raise HTTPException(404, "会话不存在")
        history = c.execute(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (body.session_id, s["history_msgs"]),
        ).fetchall()[::-1]
        c.execute(
            "INSERT INTO messages(session_id, role, content, ts) VALUES(?,?,?,?)",
            (body.session_id, "user", text, time.time()),
        )
        if not history:
            c.execute(
                "UPDATE sessions SET title=? WHERE id=?",
                (text.replace("\n", " ")[:24], body.session_id),
            )

    # 1) 知识库检索：只在开了开关、并且库里有东西时才做
    hits = []
    if s["use_kb"]:
        hits = await retrieve(text, s["top_k"], s["embed_model"])

    # 2) 拼上下文：系统人格 + 检索结果 + 历史 + 本次问题
    messages = [{"role": "system", "content": s["system"]}]
    if hits:
        ctx = "\n\n".join(f"【{h['source']}】\n{h['text']}" for h in hits)
        messages.append({"role": "system", "content": f"从本地知识库检索到的资料：\n\n{ctx}"})
    messages += [{"role": h["role"], "content": h["content"]} for h in history]
    messages.append({"role": "user", "content": text})

    async def gen():
        answer = ""
        try:
            yield _sse({"type": "sources", "items": hits})
            async for kind, piece in ollama_chat(messages, s["model"], s["temperature"]):
                if kind == "content":
                    answer += piece
                yield _sse({"type": kind, "text": piece})
        except Exception as e:
            yield _sse({"type": "error", "text": str(e)})
        finally:
            if answer:
                with conn() as c:
                    c.execute(
                        "INSERT INTO messages(session_id, role, content, ts) VALUES(?,?,?,?)",
                        (body.session_id, "assistant", answer, time.time()),
                    )
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _sse(obj) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------- 知识库接口

@app.get("/api/kb")
async def kb_list():
    with conn() as c:
        rows = c.execute(
            "SELECT source, COUNT(*) AS n FROM kb GROUP BY source ORDER BY source"
        ).fetchall()
    return [dict(r) for r in rows]


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    s = load_settings()
    raw = await file.read()
    name = file.filename or "untitled.txt"
    text = extract_text(name, raw)
    if not text.strip():
        raise HTTPException(400, "没能从这个文件里提取到文字")
    chunks = split_text(text, s["chunk_size"], s["chunk_overlap"])
    if not chunks:
        raise HTTPException(400, "文件太短或没有有效内容")

    vectors = []
    for i in range(0, len(chunks), 8):
        vectors += await ollama_embed(chunks[i:i + 8], s["embed_model"])

    with conn() as c:
        c.execute("DELETE FROM kb WHERE source=?", (name,))
        for i, (t, v) in enumerate(zip(chunks, vectors)):
            c.execute("INSERT INTO kb(source, idx, text, vec) VALUES(?,?,?,?)",
                      (name, i, t, json.dumps(v)))
    return {"source": name, "chunks": len(chunks)}


@app.delete("/api/kb")
async def kb_delete(source: str):
    with conn() as c:
        c.execute("DELETE FROM kb WHERE source=?", (source,))
    return {"ok": True}


if __name__ == "__main__":
    import threading

    import uvicorn

    def warmup():
        """启动时先把两个模型载入显存，否则第一次提问要等半分钟。"""
        s = load_settings()
        try:
            with httpx.Client(timeout=600) as cli:
                cli.post(f"{OLLAMA}/api/generate", json={
                    "model": s["model"], "prompt": "hi", "stream": False,
                    "options": {"num_predict": 1}, "keep_alive": "30m"})
                cli.post(f"{OLLAMA}/api/embed", json={
                    "model": s["embed_model"], "input": "hi", "keep_alive": "30m"})
            print("  模型已载入显存")
        except Exception:
            print("  预热失败，第一次对话会慢一些（不影响使用）")

    port = int(os.environ.get("PORT", "8420"))
    print(f"\n  私人助手已启动  ->  http://127.0.0.1:{port}\n")
    threading.Thread(target=warmup, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
