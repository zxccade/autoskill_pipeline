#!/usr/bin/env python3
"""
Stage 2: Rewrite Training Set — 用 Qwen2.5-VL-7B 将训练集 query 改写为
benchmark 测试集风格, 实现分布对齐.

每个 benchmark 的 query 风格:
  - LongVideoBench: "In the video, ..." + 多选题
  - MLVU:          直接问题 + (A)(B)(C)(D) 格式
  - LVBench:       "Select the best answer..." + A. B. C. D.
  - VideoMME:      "Select the best answer..." + A. B. C. D.

策略:
  1. 从测试集每个类别采样 2-3 个 query 作为 style reference
  2. 用 Qwen2.5-VL-7B 把训练 query rewrite 成测试集风格
  3. 保留原始 category 和 skill_results 标签

输入:
  --train-json      训练集 JSON (含 question, options, category, skill_results)
  --bench-queries   测试集 query 列表 JSON (含 input, benchmark, category, doc_id)
  --model-path      Qwen2.5-VL-7B 模型路径
  --output          输出 rewrite 后的训练集 JSON

输出:
  train_rewritten.json — rewrite 后的训练集 (保留原始标签 + rewritten_question)
"""
import json
import argparse
import os
import random
import time
from collections import defaultdict


def build_rewrite_prompt(train_q, train_opts, ref_qs):
    """构建 rewrite prompt."""
    ref_text = "\n".join(f"Example {i+1}: {r}" for i, r in enumerate(ref_qs[:2]))
    opts_text = "\n".join(train_opts) if train_opts else ""
    return (
        f"Rewrite the following video QA question to match the style of the examples below. "
        f"Keep the same meaning and category, only change the wording/style.\n\n"
        f"Style examples:\n{ref_text}\n\n"
        f"Original question: {train_q}\n"
        f"Original options:\n{opts_text}\n\n"
        f"Rewritten question (with options, match the style exactly):"
    )


def main():
    ap = argparse.ArgumentParser(description="Stage 2: Rewrite training set to match benchmark style")
    ap.add_argument("--train-json", required=True, help="Training set JSON")
    ap.add_argument("--bench-queries", required=True, help="Benchmark query list JSON")
    ap.add_argument("--model-path", required=True, help="Qwen2.5-VL-7B model path")
    ap.add_argument("--output", default="train_rewritten.json", help="Output JSON path")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-refs", type=int, default=3, help="Style reference queries per category")
    args = ap.parse_args()

    # 加载训练集
    train = json.load(open(args.train_json))
    print(f"Training set: {len(train)} samples")

    # 加载测试集 query 作为 style reference
    cls_data = json.load(open(args.bench_queries))
    test_by_cat = defaultdict(list)
    for item in cls_data:
        test_by_cat[item["category"]].append(item["input"])

    # 每个类别采样 style reference
    style_refs = {}
    for cat in test_by_cat:
        refs = test_by_cat[cat]
        random.seed(42)
        style_refs[cat] = random.sample(refs, min(args.n_refs, len(refs)))

    print(f"Style references: {sum(len(v) for v in style_refs.values())} samples from test set")

    # Qwen2.5-VL 推理
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    import torch
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    def gen_batch(prompts):
        msgs = [[{"role": "user", "content": p}] for p in prompts]
        texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=256, do_sample=False, pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        return tok.batch_decode(gen, skip_special_tokens=True)

    # 批量 rewrite
    rewritten = []
    B = args.batch_size
    t0 = time.time()
    for i in range(0, len(train), B):
        chunk = train[i : i + B]
        prompts = []
        for s in chunk:
            cat = s.get("category", s.get("category_19class", "other"))
            refs = style_refs.get(cat, [])
            prompts.append(
                build_rewrite_prompt(s.get("question", ""), s.get("options", []), refs)
            )

        raws = gen_batch(prompts)
        for s, raw in zip(chunk, raws):
            new_s = dict(s)
            new_s["rewritten_question"] = raw.strip()
            new_s["original_question"] = s.get("question", "")
            rewritten.append(new_s)

        if i % 320 == 0 or i + B >= len(train):
            done = min(i + B, len(train))
            print(f"  {done}/{len(train)}  {time.time()-t0:.0f}s", flush=True)

    json.dump(rewritten, open(args.output, "w"), ensure_ascii=False, indent=2)
    print(f"\nSaved: {args.output} ({len(rewritten)} samples)")


if __name__ == "__main__":
    main()
