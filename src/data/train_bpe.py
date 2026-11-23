# -*- coding: utf-8 -*-
"""训练中文 BPE 分词器，并与商用分词器对比。

实验：20261123-token-bpe
用法：
    cp <下载的 poet.tang.0.json> data/raw/tang0.json
    python src/data/train_bpe.py

产物写入 models/tokenizer/：两个分词器权重 + 指标 + 切分明细。
原始语料不入库，见 configs/tokenizer/bpe-20261123.json 里的 source 与 sha256。
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

REPO = Path(__file__).resolve().parents[2]


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    return cfg


def verify_corpus(path, expected_sha256, expected_bytes):
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_bytes and len(raw) != expected_bytes:
        raise SystemExit(f"语料字节数不符：期望 {expected_bytes}，实际 {len(raw)}")
    if expected_sha256 and digest != expected_sha256:
        raise SystemExit(f"语料 sha256 不符：期望 {expected_sha256}，实际 {digest}")
    return len(raw)


def load_lines(corpus_path):
    """全唐诗 JSON -> 每首诗拆成标题 + 各行，去重后返回。"""
    data = json.loads(corpus_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("语料格式不符：期望 JSON 数组")
    raw = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if item.get("title"):
            raw.append(str(item["title"]).strip())
        for line in item.get("paragraphs") or []:
            raw.append(str(line).strip())
    seen, lines = set(), []
    for line in raw:
        if line and line not in seen:
            seen.add(line)
            lines.append(line)
    return lines


def train_char_bpe(lines, vocab_size, min_freq, special_tokens, out_path):
    alphabet = sorted({ch for line in lines for ch in line})
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.decoder = decoders.Fuse()
    trainer = trainers.BpeTrainer(
        vocab_size=max(vocab_size, len(alphabet) + 64),
        special_tokens=special_tokens,
        initial_alphabet=alphabet,
        min_frequency=min_freq,
        show_progress=False,
    )
    tok.train_from_iterator(lines, trainer=trainer)
    tok.save(str(out_path))
    return tok, len(alphabet)


def train_byte_bpe(lines, vocab_size, min_freq, special_tokens, out_path):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=min_freq,
        show_progress=False,
    )
    tok.train_from_iterator(lines, trainer=trainer)
    tok.save(str(out_path))
    return tok


def get_ids(tok, text):
    enc = tok.encode(text)
    return enc if isinstance(enc, list) else enc.ids


def segment_units(tok, text):
    """累积解码取增量，得到「这段文字花了几个 token」的对照表。

    部分 token 只是某个汉字的一个字节片段，单独解码是乱码，所以按增量归属：
    这类片段并入它最终拼成的那个字，并累加消耗的 token 数。
    """
    ids = get_ids(tok, text)
    units, prev = [], ""
    for i in range(1, len(ids) + 1):
        cur = tok.decode(ids[:i]).rstrip("\ufffd")
        added = cur[len(prev):]
        prev = cur
        if added:
            units.append({"text": added, "n": 1})
        elif units:
            units[-1]["n"] += 1
        else:
            units.append({"text": "", "n": 1})
    merged, carry = [], 0
    for u in units:
        if u["text"] == "":
            carry += u["n"]
        else:
            merged.append({"text": u["text"], "n": u["n"] + carry})
            carry = 0
    if carry:
        if merged:
            merged[-1]["n"] += carry
        else:
            merged.append({"text": "?", "n": carry})
    return merged


def load_tiktoken():
    try:
        import tiktoken
    except Exception:
        return {}
    out = {}
    for name in ("cl100k_base", "o200k_base"):
        try:
            out[name] = tiktoken.get_encoding(name)
        except Exception:
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "tokenizer" / "bpe-20261123.json"))
    ap.add_argument("--out", default=str(REPO / "models" / "tokenizer"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_cfg = cfg["corpus"]["primary"]
    corpus_path = REPO / corpus_cfg["path"]
    if not corpus_path.exists():
        raise SystemExit(
            f"缺少语料 {corpus_path}\n"
            f"请从 {corpus_cfg['source']} 下载，并确认 sha256 = {corpus_cfg['sha256']}"
        )
    nbytes = verify_corpus(corpus_path, corpus_cfg.get("sha256"), corpus_cfg.get("bytes"))
    lines = load_lines(corpus_path)
    total_chars = sum(len(x) for x in lines)
    alphabet = sorted({ch for line in lines for ch in line})

    random.Random(cfg["seed"]).shuffle(lines)
    split = int(len(lines) * cfg["split"]["train_ratio"])
    train_lines, held_lines = lines[:split], lines[split:]

    print(f"语料 {corpus_cfg['path']}  {nbytes:,} 字节  sha256 校验通过")
    print(f"文本 {len(lines):,} 条 / {total_chars:,} 字符 / {len(alphabet):,} 个不同字符")
    print(f"划分 训练 {len(train_lines):,} 条，留出 {len(held_lines):,} 条")

    ccfg = cfg["tokenizer"]["char_start"]
    bcfg = cfg["tokenizer"]["byte_start"]
    tok_char, alpha = train_char_bpe(
        train_lines, ccfg["vocab_size"], ccfg["min_frequency"], ccfg["special_tokens"],
        out_dir / "tokenizer_char.json",
    )
    tok_byte = train_byte_bpe(
        train_lines, bcfg["vocab_size"], bcfg["min_frequency"], bcfg["special_tokens"],
        out_dir / "tokenizer_byte.json",
    )
    print(f"词表 字符表起步 {len(tok_char.get_vocab()):,}（初始 {alpha:,} 字符）")
    print(f"词表 字节起步   {len(tok_byte.get_vocab()):,}")

    encs = load_tiktoken()
    models_all = [("我的 BPE（字符表起步）", tok_char), ("我的 BPE（字节起步）", tok_byte)]
    models_all += list(encs.items())

    held_chars = sum(len(x) for x in held_lines)
    rows = [{"name": "字符级基线", "vocab": alpha, "tokens": held_chars, "ratio": 1.0, "oov_rate": 0.0}]
    print(f"\n留出集压缩率（{held_chars:,} 字符，越低越省）")
    print(f"  {'字符级基线':<22} {held_chars:>9,}  {1.00:.2f} token/字")
    for name, tok in models_all:
        n = sum(len(get_ids(tok, x)) for x in held_lines)
        rows.append({
            "name": name,
            "vocab": len(tok.get_vocab()) if hasattr(tok, "get_vocab") else None,
            "tokens": n,
            "ratio": round(n / held_chars, 3),
            "oov_rate": None,
        })
        print(f"  {name:<22} {n:>9,}  {n / held_chars:.2f} token/字")

    probe = cfg["eval"]["oov_probe_sentences"]
    probe_chars = sum(len(s) for s in probe)
    print(f"\n生僻字覆盖（{len(probe)} 句现代中文，{probe_chars} 字）")
    for name, tok in models_all:
        unk_id = tok.token_to_id("<unk>") if hasattr(tok, "token_to_id") else None
        oov = 0
        if unk_id is not None:
            for s in probe:
                oov += sum(1 for t in get_ids(tok, s) if t == unk_id)
        row = next(r for r in rows if r["name"] == name)
        row["oov_rate"] = round(oov / probe_chars, 4)
        print(f"  {name:<22} 无法识别 {oov}/{probe_chars}")

    print("\n无损还原")
    roundtrip = {}
    for name, tok in models_all:
        ok = all(tok.decode(get_ids(tok, s)) == s for s in probe)
        roundtrip[name] = ok
        print(f"  {name:<22} {'通过' if ok else '失败'}")

    samples = []
    for s in probe[: cfg["eval"]["segmentation_samples"]]:
        samples.append({
            "text": s,
            "models": [
                {"name": name, "pieces": segment_units(tok, s)} for name, tok in models_all
            ],
        })

    saved = json.loads((out_dir / "tokenizer_char.json").read_text(encoding="utf-8"))
    merges = ["".join(pair) for pair in saved["model"]["merges"][:2000]]
    merges = [m for m in merges if len(m) >= 2 and "\n" not in m and " " not in m][:24]

    metrics = {
        "experiment": cfg["experiment"],
        "seed": cfg["seed"],
        "corpus": {
            "path": corpus_cfg["path"],
            "sha256": corpus_cfg["sha256"],
            "bytes": nbytes,
            "lines": len(lines),
            "chars": total_chars,
            "unique_chars": len(alphabet),
            "train_lines": len(train_lines),
            "held_lines": len(held_lines),
            "held_chars": held_chars,
        },
        "vocab": {"char_start": len(tok_char.get_vocab()), "byte_start": len(tok_byte.get_vocab())},
        "rows": rows,
        "roundtrip": roundtrip,
        "learned_merges": merges,
        "loss_example": {
            "original": probe[0],
            "decoded_char_start": tok_char.decode(get_ids(tok_char, probe[0])),
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "segments.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n产出一：{out_dir}/tokenizer_char.json、tokenizer_byte.json、metrics.json、segments.json")
    print("学到的合并规则（前 12）：", " / ".join(merges[:12]))


if __name__ == "__main__":
    main()
