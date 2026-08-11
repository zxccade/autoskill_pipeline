#!/usr/bin/env python3
"""
Stage 6: Evaluate AutoSkill — 用路由表 + 分类预测在 5 个 benchmark 上评估.

Inference 逻辑 (对每个 benchmark 样本):
  1. 收集 6 候选 skill 的答对情况
  2. 全对 (non-valuable) → 判对
  3. 全错 (non-valuable) → 判错
  4. 有分歧 (valuable) → Qwen 分类预测 category → 查路由表 → 用路由 skill 结果

评估:
  - mlvu / mlvu_test: task_type macro-average
  - videomme / longvideobench / lvbench: sample-level average
  - 最终 = 5 benchmark 平均

输入:
  --bench-dir       benchmark per-sample 结果目录
  --skills          候选 skill 列表 (逗号分隔)
  --router-table    路由表 JSON (category -> skill)
  --cls-pred        分类预测 JSON (sample_id -> category)
  --benchmarks      benchmark 列表 (逗号分隔)

输出: 打印评估结果表格
"""
import json
import argparse
import glob
from collections import defaultdict


BENCHMARKS = ["mlvu", "mlvu_test", "longvideobench", "videomme", "lvbench"]


def is_correct_sample(item, bench):
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


def load_per_sample(bench_dir, bench, skill):
    """加载某 benchmark + skill 的 per-sample 结果."""
    pattern = f"{bench_dir}/{bench}/{skill}/Qwen__Qwen2.5-VL-7B-Instruct/*_samples_*.jsonl"
    files = glob.glob(pattern)
    if not files:
        return {}, {}
    ps = {}
    task_types = {}
    with open(files[0]) as f:
        for line in f:
            item = json.loads(line)
            sid = f"{bench}::{bench}_{item.get('doc_id', '')}"
            ps[sid] = is_correct_sample(item, bench)
            if bench in ("mlvu", "mlvu_test"):
                task_types[sid] = item.get("mlvu_percetion_score", {}).get("task_type", "unknown")
    return ps, task_types


def route_sample(sid, skill_correct, router_table, cls_pred):
    """对单样本执行路由."""
    if not skill_correct:
        return False, "all_wrong"
    if all(skill_correct.values()):
        return True, "all_correct"
    if not any(skill_correct.values()):
        return False, "all_wrong"
    # valuable: 用分类预测 + 路由表
    pred_cls = cls_pred.get(sid, "other")
    routed_skill = router_table.get(pred_cls, "uniform_128_direct")
    routed_correct = skill_correct.get(routed_skill, skill_correct.get("uniform_128_direct", False))
    return routed_correct, "valuable"


def macro_avg(correct_map, task_types):
    """计算 macro-average (MLVU/MLVU-test 按 task_type)."""
    if not task_types:
        return sum(int(c) for c in correct_map.values()) / max(len(correct_map), 1) * 100
    tc, tn = defaultdict(int), defaultdict(int)
    for sid, c in correct_map.items():
        tt = task_types.get(sid, "unknown")
        tc[tt] += int(c)
        tn[tt] += 1
    return sum(tc[t] / max(tn[t], 1) * 100 for t in tn) / max(len(tn), 1)


def eval_single_skill(bench_dir, skills, skill, benchmarks):
    """评估单个 skill 的准确率."""
    results = {}
    for bench in benchmarks:
        ps, tt = load_per_sample(bench_dir, bench, skill)
        results[bench] = macro_avg(ps, tt)
    return results


def eval_autoskill(bench_dir, skills, router_table, cls_pred, benchmarks):
    """评估 AutoSkill 路由效果."""
    results = {}
    for bench in benchmarks:
        # 加载所有 skill 的 per-sample 结果
        bsp = {}
        for sk in skills:
            ps, _ = load_per_sample(bench_dir, bench, sk)
            bsp[sk] = ps

        # task_types (for MLVU macro)
        _, task_types = load_per_sample(bench_dir, bench, skills[-1])  # baseline

        all_sids = set()
        for sk in skills:
            all_sids |= set(bsp[sk].keys())

        correct_map = {}
        n_val, n_ac, n_aw = 0, 0, 0
        for sid in all_sids:
            ck = {}
            for sk in skills:
                if sid in bsp[sk]:
                    ck[sk] = bsp[sk][sid]
            if not ck:
                continue
            routed_correct, stype = route_sample(sid, ck, router_table, cls_pred)
            correct_map[sid] = routed_correct
            if stype == "all_correct":
                n_ac += 1
            elif stype == "all_wrong":
                n_aw += 1
            else:
                n_val += 1

        macro = macro_avg(correct_map, task_types)
        results[bench] = {
            "macro": macro,
            "sample": sum(int(c) for c in correct_map.values()) / max(len(correct_map), 1) * 100,
            "n": len(correct_map),
            "n_valuable": n_val,
            "n_all_correct": n_ac,
            "n_all_wrong": n_aw,
        }
    return results


def main():
    ap = argparse.ArgumentParser(description="Stage 6: Evaluate AutoSkill on 5 benchmarks")
    ap.add_argument("--bench-dir", required=True, help="Benchmark per-sample results directory")
    ap.add_argument("--skills", required=True, help="Comma-separated candidate skills")
    ap.add_argument("--router-table", required=True, help="Router table JSON")
    ap.add_argument("--cls-pred", required=True, help="Classification predictions JSON")
    ap.add_argument("--benchmarks", default="mlvu,mlvu_test,longvideobench,videomme,lvbench")
    args = ap.parse_args()

    skills = args.skills.split(",")
    benchmarks = args.benchmarks.split(",")
    router_table = json.load(open(args.router_table))
    cls_pred = json.load(open(args.cls_pred))

    print("=" * 80)
    print("AutoSkill Evaluation")
    print("=" * 80)

    # 打印路由表
    print("\n=== Router Table ===")
    for cls, sk in sorted(router_table.items()):
        print(f"  {cls:30s} -> {sk}")

    # 评估 baseline
    print("\n=== Baseline (uniform_128_direct) ===")
    bl = eval_single_skill(args.bench_dir, skills, "uniform_128_direct", benchmarks)
    for b in benchmarks:
        print(f"  {b:20s} {bl[b]:.2f}%")

    # 评估 single best (top5 第一个)
    best_skill = skills[0] if skills[0] != "uniform_128_direct" else skills[1]
    print(f"\n=== Single Best ({best_skill}) ===")
    sb = eval_single_skill(args.bench_dir, skills, best_skill, benchmarks)
    for b in benchmarks:
        print(f"  {b:20s} {sb[b]:.2f}%")

    # 评估 AutoSkill
    print("\n=== AutoSkill (router + classification) ===")
    rt = eval_autoskill(args.bench_dir, skills, router_table, cls_pred, benchmarks)
    for b in benchmarks:
        r = rt[b]
        print(f"  {b:20s} macro={r['macro']:.2f}%  (n={r['n']}, valuable={r['n_valuable']})")

    # 汇总
    print(f"\n{'='*80}")
    print(f"{'Benchmark':20s} {'Baseline':>10s} {'SingleBest':>12s} {'AutoSkill':>10s} {'Δ(vs BL)':>10s}")
    print(f"{'='*80}")
    for b in benchmarks:
        print(f"{b:20s} {bl[b]:>9.1f}% {sb[b]:>11.1f}% {rt[b]['macro']:>9.1f}% {rt[b]['macro']-bl[b]:>+9.1f}%")
    bl_avg = sum(bl.values()) / len(benchmarks)
    sb_avg = sum(sb.values()) / len(benchmarks)
    rt_avg = sum(rt[b]["macro"] for b in benchmarks) / len(benchmarks)
    print(f"{'-'*80}")
    print(f"{'Average':20s} {bl_avg:>9.1f}% {sb_avg:>11.1f}% {rt_avg:>9.1f}% {rt_avg-bl_avg:>+9.1f}%")


if __name__ == "__main__":
    main()
