#!/usr/bin/env python3
"""
Stage 5: Classify Benchmark Samples — 用 Qwen2.5-VL-7B 对 benchmark 上的
valuable 样本 (6 skill 答对情况不一致) 做 19 类分类预测.

Valuable 样本定义: 6 候选 skill 中既有答对的也有答错的 (有分歧).
Non-valuable 样本 (全对/全错) 不需要分类.

分类 prompt (v3): 19 类 + 描述, 要求只输出类别名.

输入:
  --bench-dir       benchmark per-sample 结果目录 (含 6 skill × 5 benchmark)
  --skills          候选 skill 列表 (逗号分隔)
  --benchmarks      benchmark 列表 (逗号分隔, 默认 mlvu,mlvu_test,longvideobench,videomme,lvbench)
  --model-path      Qwen2.5-VL-7B 模型路径
  --output          输出 {sample_id: category} JSON

输出:
  cls_pred_v3.json — {benchmark::benchmark_doc_id: predicted_category}
"""
import json
import argparse
import glob
import re
from collections import defaultdict


# 19 类
CATEGORIES = [
    "action_recognition", "anomaly_detection", "appearance", "causal_reasoning",
    "counting", "emotion_state", "event_identification", "fact_verification",
    "general_qa", "narrative_plot", "negative_qa", "object_identification",
    "ocr_text", "other", "person_attribute", "spatial_location",
    "temporal_ordering", "timestamp_specific", "yes_no",
]

# v3 prompt 的类别描述
DESC_V3 = {
    "ocr_text": "the question asks what text, words, subtitles, captions, signs, or labels are shown or written",
    "object_identification": "the question asks what a specific physical object, tool, food, or animal is",
    "person_attribute": "the question asks who a person is, or what they wear, carry, or hold",
    "appearance": "the question asks about color, shape, size, pattern, or visual look of something",
    "action_recognition": "the question asks what someone or something is doing",
    "counting": "the question asks how many or the number of something",
    "temporal_ordering": "the question asks about the order or sequence of events",
    "spatial_location": "the question asks where something is located",
    "timestamp_specific": "the question asks about a specific time or moment in the video",
    "causal_reasoning": "the question asks why something happens or the reason",
    "narrative_plot": "the question asks for a summary, main topic, or overall content of the video",
    "anomaly_detection": "the question asks about abnormal, unusual, or unexpected events",
    "emotion_state": "the question asks about emotion, mood, feeling, or expression",
    "event_identification": "the question asks to identify a named event, competition, award, film, or match",
    "fact_verification": "the question asks which statement or description is correct",
    "general_qa": "the question is a generic factual question not fitting other categories",
    "negative_qa": "the question asks what does NOT appear or is NOT included",
    "yes_no": "the question can be answered with yes or no",
    "other": "the question does not fit any of the above categories",
}

# benchmark 输入的前缀/后缀 (需要剥离提取纯问题)
PREFIXES = [
    "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option.\n",
    "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.\n",
]
SUFFIX_PAT = re.compile(r"Answer with the option's letter from the given choices directly\.?\s*$", re.I)


def strip_prefix(inp):
    for p in PREFIXES:
        if inp.startswith(p):
            return inp[len(p):].strip()
    return inp.strip()


def question_only(inp):
    """从 benchmark input 提取纯问题 (去掉前缀/后缀/选项)."""
    q = strip_prefix(inp)
    q = SUFFIX_PAT.sub("", q).strip()
    m = re.search(r"\n[ ]*A[.)] ", q)
    if m:
        q = q[: m.start()].strip()
    return q


def build_prompt(question):
    """构建 v3 分类 prompt."""
    cat_lines = "\n".join(f"- {c}: {DESC_V3[c]}" for c in CATEGORIES)
    return (
        "Classify this video question-answering query into exactly ONE category.\n"
        f"Categories:\n{cat_lines}\n\n"
        f"Question: {question}\n\n"
        "Reply with ONLY the category name (lowercase, exactly as listed), nothing else."
    )


def parse_category(raw):
    """解析模型输出为类别名."""
    t = raw.strip().lower()
    for c in CATEGORIES:
        if t == c:
            return c
    best, best_pos = None, 1e9
    for c in CATEGORIES:
        p = t.find(c)
        if p != -1 and p < best_pos:
            best_pos, best = p, c
    return best or "other"


def is_correct(item, bench):
    """判定单样本是否答对."""
    if bench == "longvideobench":
        s = item.get("lvb_acc", {}) or {}
        ans, pred = s.get("answer"), s.get("parsed_pred")
        if ans is not None and pred is not None:
            return str(ans) == str(pred)
        p = str(item.get("filtered_resps", "")).strip()
        t = str(item.get("target", "")).strip()
        if p and p[0] in "ABCDE":
            return str(ord(p[0]) - ord("A")) == t
        return p == t
    if bench in ("mlvu", "mlvu_test"):
        s = item.get("mlvu_percetion_score", {}) or {}
        ans, pa = s.get("answer"), s.get("pred_answer")
        if ans is not None and pa is not None:
            return str(ans) == str(pa)
        p = str(item.get("filtered_resps", ""))
        t = str(item.get("target", ""))
        pl = next((c.upper() for c in p if c.upper() in "ABCDE"), "")
        return pl == (t.upper()[:1] if t else "")
    if bench == "videomme":
        s = item.get("videomme_perception_score", {}) or {}
        ans, pa = s.get("answer"), s.get("pred_answer")
        if ans is not None and pa is not None:
            return str(ans) == str(pa)
        p = str(item.get("filtered_resps", ""))
        t = str(item.get("target", ""))
        pl = next((c.upper() for c in p if c.upper() in "ABCDE"), "")
        return pl == (t.upper()[:1] if t else "")
    if bench == "lvbench":
        return bool(item.get("lvbench_score"))
    return False


def find_valuable(bench_dir, bench, skills):
    """找出 benchmark 上的 valuable 样本 (6 skill 有分歧)."""
    bsp = {}
    sid2question = {}
    for sk in skills:
        pattern = f"{bench_dir}/{bench}/{sk}/Qwen__Qwen2.5-VL-7B-Instruct/*_samples_*.jsonl"
        files = glob.glob(pattern)
        if not files:
            continue
        ps = {}
        with open(files[0]) as fh:
            for line in fh:
                item = json.loads(fh)
                doc_id = str(item.get("doc_id", ""))
                ps[doc_id] = is_correct(item, bench)
                if doc_id not in sid2question:
                    sid2question[doc_id] = item.get("input", "")
        bsp[sk] = ps

    all_sids = set()
    for sk in skills:
        all_sids |= set(bsp.get(sk, {}).keys())

    valuable = {}
    for sid in all_sids:
        results = [bsp[sk][sid] for sk in skills if sid in bsp.get(sk, {})]
        if not results:
            continue
        if all(results) or not any(results):
            continue  # non-valuable
        valuable[sid] = sid2question.get(sid, "")
    return valuable


def main():
    ap = argparse.ArgumentParser(description="Stage 5: Classify benchmark valuable samples")
    ap.add_argument("--bench-dir", required=True, help="Benchmark per-sample results directory")
    ap.add_argument("--skills", required=True, help="Comma-separated skill list")
    ap.add_argument("--benchmarks", default="mlvu,mlvu_test,longvideobench,videomme,lvbench")
    ap.add_argument("--model-path", required=True, help="Qwen2.5-VL-7B model path")
    ap.add_argument("--output", default="cls_pred_v3.json")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    skills = args.skills.split(",")
    benchmarks = args.benchmarks.split(",")

    # 1. 找 valuable 样本
    print("=== Step 1: Find valuable samples ===")
    all_valuable = {}
    for bench in benchmarks:
        valuable = find_valuable(args.bench_dir, bench, skills)
        for sid, q in valuable.items():
            full_sid = f"{bench}::{bench}_{sid}"
            all_valuable[full_sid] = q
        print(f"  {bench}: {len(valuable)} valuable")
    print(f"Total valuable: {len(all_valuable)}")

    if not all_valuable:
        print("No valuable samples!")
        return

    # 2. Qwen2.5-VL-7B 分类
    print("\n=== Step 2: Qwen2.5-VL-7B classification ===")
    import torch
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    sids = list(all_valuable.keys())
    questions = [question_only(all_valuable[sid]) for sid in sids]
    prompts = [build_prompt(q) for q in questions]

    results = {}
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i : i + args.batch_size]
        batch_sids = sids[i : i + args.batch_size]
        msgs = [[{"role": "user", "content": p}] for p in batch]
        texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**enc, max_new_tokens=20, do_sample=False, temperature=1.0)
        for j, out in enumerate(outputs):
            raw = tok.decode(out[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            cat = parse_category(raw)
            results[batch_sids[j]] = cat
        if (i // args.batch_size + 1) % 10 == 0 or i + args.batch_size >= len(prompts):
            print(f"  {min(i + args.batch_size, len(prompts))}/{len(prompts)}")

    # 3. 保存
    json.dump(results, open(args.output, "w"), indent=2)
    print(f"\nSaved: {args.output} ({len(results)} samples)")

    from collections import Counter
    print("\nCategory distribution:")
    for cat, n in Counter(results.values()).most_common():
        print(f"  {cat:30s}: {n}")


if __name__ == "__main__":
    main()
