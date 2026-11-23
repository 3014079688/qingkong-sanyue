# 私人 AI 助手

一个跑在你自己笔记本上的对话助手。模型推理交给 Ollama，**应用层全部是手写代码**——没有框架、没有黑盒，每个文件都能读懂、能改。

- 界面：浏览器打开 `http://127.0.0.1:8420`
- 数据：全部在本地 `_data/`，对话记录在 SQLite 里，不联网、不外传
- 模型：Qwen2.5-3B（中文，直答不啰嗦），另装了 Qwen3-4B 可随时切换

---

## 快速开始

双击 `start.bat` 就行。它做三件事：启动推理服务 → 打开浏览器 → 启动助手服务。

手动启动等价于：

```bat
C:\Users\ASUS\ollama-portable\bin\ollama.exe serve
python server.py
```

第一次用之前确认模型在本地：

```bat
ollama list          # 应该能看到 qwen3:4b 和 nomic-embed-text
```

---

## 目录结构

```
ai-assistant/
├── server.py           后台服务：上下文拼接、会话记忆、知识库检索、流式输出
├── static/index.html   界面：一个文件，无构建、无依赖
├── start.bat           一键启动
├── _data/
│   ├── chat.db         会话和消息（SQLite，可用任意工具打开查看）
│   ├── settings.json   人格设定、模型、温度等
│   └── kb 向量         存在 chat.db 的 kb 表里
└── README.md
```

---

## 它到底是怎么工作的

一次对话的完整数据流，全部在 `server.py` 的 `chat()` 里，不到 40 行：

```
你打字
  ↓
1. 从 kb 表取所有文本块，把你的问题也向量化，算余弦相似度，取最像的 4 块
  ↓
2. 拼成一个 messages 数组：
     [system: 人格设定]
     [system: 检索到的资料]
     [user: 上一轮问题] [assistant: 上一轮回答]   ← 历史，最多 24 条
     [user: 这次的问题]
  ↓
3. POST 给 Ollama，它把 token 一个个吐回来
  ↓
4. 每收到一个 token 就 SSE 推给浏览器，界面实时追加
  ↓
5. 完整回答存进 messages 表
```

**没有别的魔法了。** 所有「AI 助手」的差别——记不记得住上下文、会不会引用资料、回答风格——全都在这 40 行里，不在模型里。

---

## 你可以改什么

| 想改的东西 | 改哪里 |
|---|---|
| 助手的性格、口吻、边界 | 界面「设置」里的系统提示词，或 `_data/settings.json` 的 `system` |
| 换模型 | 设置里下拉选择，或 `ollama pull qwen3:8b` 后切换 |
| 上下文能记多久 | `settings.json` 的 `history_msgs`（越大越慢、越占显存） |
| 检索几块资料 | `top_k` |
| 切块大小 | `chunk_size` / `chunk_overlap` |
| 界面配色 | `index.html` 顶部 `:root` 里的 CSS 变量 |

---

## 想继续往下钻（第二层：自己写推理）

现在 `server.py` 里 `ollama_chat()` 这个函数，就是「推理」的全部。想看懂它在干什么，把它换成下面这段——用 transformers 直接跑权重，你会亲眼看到 token 是怎么一个个冒出来的：

```python
import threading, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B", torch_dtype=torch.float16, device_map="auto")

text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt").to(model.device)

streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
threading.Thread(target=model.generate, kwargs=dict(
    input_ids=ids.input_ids, max_new_tokens=512,
    streamer=streamer, do_sample=True, temperature=0.7,
)).start()

for piece in streamer:      # 这里就是「一个 token 一个 token 往外吐」
    yield "content", piece
```

差别：Ollama 用 llama.cpp + GGUF 量化，4G 显存能跑；transformers 用 fp16，同样模型要 8G 以上显存，或者加 `load_in_4bit=True`（需要 bitsandbytes）。**理解用前者，学习用后者。**

再往下一层（预训练）就不用碰了。

---

## 这台机器上的实际表现

RTX 3050 Ti Laptop = **4GB 显存**，这是硬约束，决定了模型上限。以下是实测（i5-12500H + 3050Ti）：

| 模型 | 大小 | 实测 | 评价 |
|---|---|---|---|
| **qwen2.5:3b** | 1.9G | 首字 2.3s，一句话回答 2.5s | **默认**。直答、不废话，日常最舒服 |
| qwen3:4b | 2.5G | 问「光速是多少」耗时 51 秒 | 先思考再回答，逻辑题更强，但太啰嗦 |
| qwen3:1.7b | 1.2G | 更快 | 更轻，能力也更弱 |
| qwen3:8b | ~5G | 3–5 tok/s（一半跑 CPU） | 质量好一截，需要耐心 |

### 一个必须知道的坑：思考型模型

Qwen3 全系是「混合思考」模型，回答前会先在内部生成一大段推理。实测问一句「光速是多少」，答案只有 18 个字，模型内部却生成了 **1184 个 token、跑了 51 秒**。

`/no_think` 和 `think: false` 只能让思考过程不显示，**并不能让它少算**。所以在 4GB 显存这种算力紧张的机器上，日常对话用 Qwen2.5 更合适；真遇到需要推理的题，再去设置里切成 Qwen3-4B。

这一点也解释了为什么「本地跑大模型」的体验差异极大——不是模型越新越好，是要看它的推理模式跟你的显存匹不匹配。

---

## 常见问题

**卡在「Ollama 未启动」**
推理服务没跑。单独执行一次 `ollama serve`，或在任务管理器里看有没有 `ollama.exe`。

**回答特别慢 / 硬盘狂转**
显存不够，模型被切到 CPU 了。换小一号模型，或把 `history_msgs` 调到 8。

**上传 PDF 报错**
需要 `pip install pypdf`。扫描版 PDF（图片）抽不出文字，得先 OCR。

**端口被占**
`set PORT=8500` 后再启动，界面地址同步改。

**想清空全部对话**
删掉 `_data/chat.db` 即可，重启后自动重建。
