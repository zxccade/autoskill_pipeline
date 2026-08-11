#!/usr/bin/env python3
"""
Stage 3: Build Router Table — 从训练样本 (1500 个, 含 skill_results) 中
按 19 个类别统计每个候选 skill 的准确率, 每类选最优 skill, 构建路由表.

注意: 不使用 recommended_skill (那是 GT, 推理时没有), 只从 skill_results 选 max.

输入:
  --train-samples   训练样本 JSONL (含 category_19class, skill_results, 每行一个样本)
  --top5-skills      Top-5 skill 列表 JSON (含 candidates)
  --output           输出路由表 JSON (category -> skill)

输出:
  router_table.json — {category: best_skill} (19 类)
"""
import json
import argparse
from collections import defaultdict


CATEGORIES_19 = [
    "action_recognition", "anomaly_detection", "appearance", "causal_reasoning",
    "counting", "emotion_state", "event_identification", "fact_verification",
    "general_qa", "narrative_plot", "negative_qa", "object_identification",
    "ocr_text", "other", "person_attribute", "spatial_location",
    "temporal_ordering", "timestamp_specific", "yes_no",
]


def main():
    ap = argparse.ArgumentParser(description="Stage 3: Build router table from training samples")
    ap.add_argument("--train-samples", required=True, help="Training samples JSONL (with category_19class, skill_results)")
    ap.add_argument("--top5-skills", required=True, help="Top-5 skills JSON (contains candidates)")
    ap.add_argument("--output", default="router_table.json")
    args = ap.parse_args()

    # 加载候选 skill
    top5_data = json.load(open(args.top5_skills))
    candidate_skills = top5_data["candidates"]  # top5 + baseline
    print(f"Candidate skills: {candidate_skills}")

    # 加载训练样本
    samples = []
    with open(args.train_samples) as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"Training samples: {len(samples)}")

    # 按类别统计每个 skill 的准确率
    cat_skill_correct = defaultdict(lambda: defaultdict(int))  # cat -> skill -> correct
    cat_skill_total = defaultdict(lambda: defaultdict(int))    # cat -> skill -> total

    for item in samples:
        cat = item.get("category_19class", item.get("category", "unknown"))
        sr = item.get("skill_results", {})
        for sk in candidate_skills:
            if sk in sr:
                cat_skill_correct[cat][sk] += int(sr[sk])
                cat_skill_total[cat][sk] += 1

    # 每类选最优 skill
    router_table = {}
    print(f"\n{'Category':30s} {'N':>5s} {'Best Skill':35s} {'Acc':>7s}   All skills")
    print("-" * 140)

    for cat in CATEGORIES_19:
        if cat not in cat_skill_total:
            router_table[cat] = "uniform_128_direct"
            print(f"{cat:30s} {'—':>5s} {'uniform_128_direct (fallback)':35s}")
            continue

        # 计算每个 skill 的准确率
        acc_by_sk = {}
        for sk in candidate_skills:
            if cat_skill_total[cat][sk] > 0:
                acc_by_sk[sk] = cat_skill_correct[cat][sk] / cat_skill_total[cat][sk]

        if not acc_by_sk:
            router_table[cat] = "uniform_128_direct"
            continue

        # 选最优
        best_sk = max(acc_by_sk, key=acc_by_sk.get)
        best_acc = acc_by_sk[best_sk]
        router_table[cat] = best_sk

        n = sum(cat_skill_total[cat].values()) // len(candidate_skills)
        all_acc_str = " | ".join(f"{sk[:15]}={acc_by_sk.get(sk, 0):.1%}" for sk in candidate_skills if sk in acc_by_sk)
        print(f"{cat:30s} {n:>5d} {best_sk:35s} {best_acc:>6.1%}  {all_acc_str}")

    # 统计路由表分布
    from collections import Counter
    dist = Counter(router_table.values())
    print(f"\n=== Router Table Distribution ===")
    for sk, n in dist.most_common():
        print(f"  {sk:35s}: {n} categories")

    # 保存
    json.dump(router_table, open(args.output, "w"), indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
