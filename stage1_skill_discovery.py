#!/usr/bin/env python3
"""
Stage 1: Skill Discovery — 在 N1800+supplement 训练集上评估所有候选 skill,
按整体准确率排名选出 Top-5 + baseline (uniform_128_direct).

输入:
  --n1800-results   N1800 的 results.json (skill -> {sample_id: {correct: bool}})
  --supp-results    supplement 的 results.json (同上)
  --n1800-meta      N1800 metadata (含 category_19class, problem_id)
  --supp-meta       supplement metadata
  --output          输出 JSON: {top5: [...], baseline: "uniform_128_direct", all_skills: [...]}

输出:
  top5_skills.json — Top-5 skill 列表 + baseline
"""
import json
import argparse
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser(description="Stage 1: Skill Discovery — rank skills on N1800+supplement")
    ap.add_argument("--n1800-results", required=True, help="N1800 results.json")
    ap.add_argument("--supp-results", required=True, help="Supplement results.json")
    ap.add_argument("--n1800-meta", required=True, help="N1800 metadata JSON")
    ap.add_argument("--supp-meta", required=True, help="Supplement metadata JSON")
    ap.add_argument("--output", default="top5_skills.json", help="Output JSON path")
    ap.add_argument("--top-k", type=int, default=5, help="Number of top skills to select")
    args = ap.parse_args()

    # 加载结果
    n1800 = json.load(open(args.n1800_results))
    supp = json.load(open(args.supp_results))

    # 合并 N1800 + supplement
    all_skills = sorted(set(list(n1800.keys()) + list(supp.keys())))
    print(f"Total skills: {len(all_skills)}")
    for s in all_skills:
        print(f"  {s}")

    # 合并 per-sample 结果
    skill_correct = defaultdict(dict)  # skill -> {sample_id: 0/1}
    all_sample_ids = set()
    for sk in all_skills:
        for sid, v in n1800.get(sk, {}).items():
            if v.get("correct") is not None:
                skill_correct[sk][sid] = 1 if v["correct"] else 0
                all_sample_ids.add(sid)
        for sid, v in supp.get(sk, {}).items():
            if v.get("correct") is not None:
                skill_correct[sk][sid] = 1 if v["correct"] else 0
                all_sample_ids.add(sid)

    print(f"\nTotal samples: {len(all_sample_ids)}")

    # 计算每个 skill 的整体准确率
    skill_acc = {}
    for sk in all_skills:
        if sk == "uniform_128_direct":
            continue  # baseline 不参与排名
        vals = [skill_correct[sk][s] for s in all_sample_ids if s in skill_correct[sk]]
        if vals:
            skill_acc[sk] = sum(vals) / len(vals)

    # 排名
    ranked = sorted(skill_acc.items(), key=lambda x: -x[1])
    print(f"\n=== Skill Ranking (by accuracy on N1800+supplement) ===")
    print(f"{'Rank':>4s}  {'Skill':45s} {'Acc':>8s}")
    print("-" * 60)
    for i, (sk, acc) in enumerate(ranked, 1):
        print(f"{i:>4d}  {sk:45s} {acc:>7.1%}")

    # 选 Top-5
    top5 = [sk for sk, _ in ranked[: args.top_k]]
    print(f"\n=== Top-{args.top_k} Skills ===")
    for i, sk in enumerate(top5, 1):
        print(f"  Skill-{i}: {sk}")

    # 输出
    out = {
        "top5": top5,
        "baseline": "uniform_128_direct",
        "candidates": top5 + ["uniform_128_direct"],
        "ranking": [
            {"rank": i + 1, "skill": sk, "accuracy": acc} for i, (sk, acc) in enumerate(ranked)
        ],
    }
    json.dump(out, open(args.output, "w"), indent=2)
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
