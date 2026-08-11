"""
Skill Registry for Video VQA.

A Skill = (frame_strategy, prompt_strategy).
  frame_strategy : (vpath: str[, question: str]) -> (frames: List[PIL.Image], duration: float)
  prompt_strategy: (frames, question, duration) -> messages list (Qwen2.5-VL format)

Baseline: uniform_128_direct (128 uniformly sampled frames, direct prompt).
New skills are added below SKILL_REGISTRY by each round's implementation.
"""

from __future__ import annotations

import base64
import hashlib
import inspect
import io
import re as _re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


MAX_PIXELS = 602112
MIN_PIXELS = 256 * 28 * 28
BASELINE = "uniform_128_direct"

CLIP_CACHE_DIR  = "/gpfs/scratch/acw761/hf_home"
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
CLIP_MODEL_NAME = "google/siglip-so400m-patch14-384"  # CLIP not in cache; use SigLIP
CLIP_POOL_N = 256
CLIP_TOP_K = 16
CLIP_BUDGET = 128
DECORD_CHUNK = 64

SPATIAL_KEYWORDS = ("left", "right", "facing", "standing", "back", "appear", "order")
LONGVIDEO_BLOCKLIST = ("left", "right", "facing", "standing", "order", "appear")

_CLIP_STATE: Dict[str, Any] | None = None


# ─────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────

def _read_video(vpath: str):
    import decord
    vr = decord.VideoReader(vpath, ctx=decord.cpu(0), num_threads=4)
    fps = vr.get_avg_fps()
    dur = len(vr) / max(fps, 1e-6)
    return vr, fps, dur


def _get_frames(vr, indices: List[int]) -> List[Image.Image]:
    indices = [min(max(0, i), len(vr) - 1) for i in indices]
    arr = vr.get_batch(indices).asnumpy()
    return [Image.fromarray(f.astype("uint8"), "RGB") for f in arr]


def _enc(img: Image.Image) -> dict:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {
        "type": "image",
        "image": f"data:image/jpeg;base64,{b64}",
        "max_pixels": MAX_PIXELS,
        "min_pixels": MIN_PIXELS,
    }


def _get_frames_chunked(vr, indices: List[int], chunk_size: int = DECORD_CHUNK) -> List[Image.Image]:
    frames: List[Image.Image] = []
    if not indices:
        return frames
    n_frames = len(vr)
    for start in range(0, len(indices), chunk_size):
        chunk = [min(max(0, int(i)), n_frames - 1) for i in indices[start:start + chunk_size]]
        arr = vr.get_batch(chunk).asnumpy()
        frames.extend(Image.fromarray(f.astype("uint8"), "RGB") for f in arr)
    return frames


def _contains_any_keyword(question: str, keywords: Tuple[str, ...]) -> bool:
    q = question.lower()
    return any(keyword in q for keyword in keywords)


def _matched_keywords(question: str, keywords: Tuple[str, ...]) -> List[str]:
    q = question.lower()
    return [keyword for keyword in keywords if keyword in q]


def _invoke_frame_strategy(frame_fn: Callable, vpath: str, question: str) -> Tuple[List[Image.Image], float]:
    sig = inspect.signature(frame_fn)
    if len(sig.parameters) >= 2:
        return frame_fn(vpath, question)
    return frame_fn(vpath)


def _unique_preserve_order(indices: List[int]) -> List[int]:
    seen = set()
    ordered: List[int] = []
    for idx in indices:
        idx = int(idx)
        if idx in seen:
            continue
        seen.add(idx)
        ordered.append(idx)
    return ordered


def _budget_counts(total: int, slots: int) -> List[int]:
    if slots <= 0:
        return []
    base = total // slots
    remainder = total % slots
    return [base + (1 if i < remainder else 0) for i in range(slots)]


def _pool_indices(num_frames: int, pool_n: int) -> List[int]:
    if num_frames <= 0:
        return []
    n = max(1, min(pool_n, num_frames))
    return np.unique(np.linspace(0, num_frames - 1, n, dtype=int)).tolist()


def _load_clip_state() -> Dict[str, Any]:
    """Load SigLIP (google/siglip-so400m-patch14-384) from local HF cache."""
    global _CLIP_STATE
    if _CLIP_STATE is not None:
        return _CLIP_STATE

    import torch
    from transformers import AutoProcessor, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[skill] Loading SigLIP ({CLIP_MODEL_NAME}) on {device} ...", flush=True)
    model = AutoModel.from_pretrained(
        CLIP_MODEL_NAME,
    ).to(device=device, dtype=dtype).eval()
    processor = AutoProcessor.from_pretrained(
        CLIP_MODEL_NAME,
    )
    _CLIP_STATE = {
        "model": model,
        "processor": processor,
        "device": device,
        "dtype": dtype,
    }
    print(f"[skill] SigLIP loaded OK on {device}", flush=True)
    return _CLIP_STATE


def _score_frames_with_clip(frames: List[Image.Image], question: str, batch_size: int = 16) -> np.ndarray:
    """Score frames against question text using SigLIP cosine similarity."""
    import torch

    state = _load_clip_state()
    model = state["model"]
    processor = state["processor"]
    device = state["device"]
    dtype = state["dtype"]

    # SigLIP text encoding (truncate to model max length)
    text_inputs = processor(
        text=[question[:512]], padding="max_length", truncation=True, return_tensors="pt"
    )
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with torch.no_grad():
        text_out = model.get_text_features(**text_inputs)
        # transformers 5.x returns BaseModelOutputWithPooling; extract pooler_output
        text_features = text_out.pooler_output if hasattr(text_out, 'pooler_output') else text_out
        text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    scores: List[float] = []
    for start in range(0, len(frames), batch_size):
        batch = frames[start:start + batch_size]
        image_inputs = processor(images=batch, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].to(device=device, dtype=dtype)
        with torch.no_grad():
            img_out = model.get_image_features(pixel_values=pixel_values)
            image_features = img_out.pooler_output if hasattr(img_out, 'pooler_output') else img_out
            image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            batch_scores = (image_features * text_features).sum(dim=-1)
        scores.extend(batch_scores.float().cpu().tolist())
    return np.asarray(scores, dtype=np.float32)


def _expand_anchor(anchor_idx: int, num_frames: int, count: int, pool_stride: float) -> List[int]:
    anchor_idx = int(min(max(0, anchor_idx), max(num_frames - 1, 0)))
    count = max(1, int(count))
    if num_frames <= 1 or count == 1:
        return [anchor_idx]
    radius = max(int(round(pool_stride * max(count - 1, 1) / 4.0)), 1)
    left = max(0, anchor_idx - radius)
    right = min(num_frames - 1, anchor_idx + radius)
    if left == right:
        return [left] * count
    return np.linspace(left, right, count, dtype=int).tolist()


def _build_f2c_indices(
    num_frames: int,
    pool_indices: List[int],
    scores: np.ndarray,
    top_k: int,
    budget: int,
) -> List[int]:
    if num_frames <= 0:
        return []
    if not pool_indices or len(scores) == 0:
        return np.linspace(0, num_frames - 1, budget, dtype=int).tolist()

    top_k = max(1, min(top_k, len(pool_indices), len(scores)))
    anchor_positions = sorted(int(i) for i in np.argsort(-scores)[:top_k])
    clip_sizes = _budget_counts(budget, len(anchor_positions))
    if len(pool_indices) > 1:
        pool_stride = max((pool_indices[-1] - pool_indices[0]) / (len(pool_indices) - 1), 1.0)
    else:
        pool_stride = 1.0

    selected: List[int] = []
    for anchor_pos, clip_size in zip(anchor_positions, clip_sizes):
        selected.extend(_expand_anchor(pool_indices[anchor_pos], num_frames, clip_size, pool_stride))
    selected = sorted(_unique_preserve_order(selected))

    if len(selected) < budget:
        seen = set(selected)
        for pos in np.argsort(-scores):
            idx = int(pool_indices[int(pos)])
            if idx in seen:
                continue
            seen.add(idx)
            selected.append(idx)
            if len(selected) >= budget:
                break
        selected = sorted(selected)

    if len(selected) < budget:
        seen = set(selected)
        for idx in np.linspace(0, num_frames - 1, budget, dtype=int).tolist():
            idx = int(idx)
            if idx in seen:
                continue
            seen.add(idx)
            selected.append(idx)
            if len(selected) >= budget:
                break
        selected = sorted(selected)

    return selected[:budget]


# ─────────────────────────────────────────────
# Frame strategies
# ─────────────────────────────────────────────

def uniform(vpath: str, n: int) -> Tuple[List[Image.Image], float]:
    vr, _, dur = _read_video(vpath)
    idx = np.linspace(0, len(vr) - 1, n, dtype=int).tolist()
    return _get_frames(vr, idx), dur


def f2c_pool256_top16_expand128(
    vpath: str,
    question: str,
    pool_n: int = CLIP_POOL_N,
    top_k: int = CLIP_TOP_K,
    budget: int = CLIP_BUDGET,
) -> Tuple[List[Image.Image], float]:
    """
    F2C: pool 256 uniform frames, score by CLIP cosine sim with question,
    select top-16 anchors, expand each into a temporal clip (budget=128 total).
    Literature: From Frames to Clips (arXiv 2510.02262, Oct 2025)
    """
    vr, _, dur = _read_video(vpath)
    pool_idx = _pool_indices(len(vr), pool_n)
    if not pool_idx:
        return uniform(vpath, budget)

    candidate_frames = _get_frames_chunked(vr, pool_idx, chunk_size=DECORD_CHUNK)
    try:
        scores = _score_frames_with_clip(candidate_frames, question)
    except Exception:
        return uniform(vpath, budget)

    final_idx = _build_f2c_indices(len(vr), pool_idx, scores, top_k=top_k, budget=budget)
    if not final_idx:
        return uniform(vpath, budget)
    return _get_frames_chunked(vr, final_idx, chunk_size=DECORD_CHUNK), dur


FRAME_STRATEGIES: Dict[str, Callable] = {
    "uniform_8":   lambda v: uniform(v, 8),
    "uniform_16":  lambda v: uniform(v, 16),
    "uniform_32":  lambda v: uniform(v, 32),
    "uniform_64":  lambda v: uniform(v, 64),
    "uniform_128": lambda v: uniform(v, 128),
    # New frame strategies added by each round go here
}


# ─────────────────────────────────────────────
# Prompt strategies
# ─────────────────────────────────────────────

_SYS = "You are a helpful assistant analyzing video frames to answer questions accurately."


def direct(frames: List[Image.Image], question: str, duration: float) -> list:
    user = [_enc(f) for f in frames]
    user.append({
        "type": "text",
        "text": f"Question: {question}\nAnswer with the single letter only (A/B/C/D/E).",
    })
    return [{"role": "system", "content": _SYS}, {"role": "user", "content": user}]

PROMPT_STRATEGIES: Dict[str, Callable] = {
    "direct": direct,
    # New prompt strategies added by each round go here
}


# ─────────────────────────────────────────────
# Skill base class
# ─────────────────────────────────────────────

@dataclass
class Skill:
    name:       str
    frame_key:  str
    prompt_key: str
    tags:       List[str] = field(default_factory=list)

    def run(self, vpath: str, question: str) -> dict:
        frame_fn = FRAME_STRATEGIES[self.frame_key]
        frames, duration = _invoke_frame_strategy(frame_fn, vpath, question)
        messages = PROMPT_STRATEGIES[self.prompt_key](frames, question, duration)
        return {
            "frames":   frames,
            "messages": messages,
            "duration": duration,
            "meta": {
                "skill":      self.name,
                "frame_key":  self.frame_key,
                "prompt_key": self.prompt_key,
                "n_frames":   len(frames),
                "question":   question,
                "tags":       list(self.tags),
            },
        }


def _fallback_to_baseline(skill_name: str, vpath: str, question: str, error: Exception) -> dict:
    baseline = SKILL_REGISTRY.get(BASELINE)
    if baseline is None or skill_name == BASELINE:
        raise error
    out = baseline.run(vpath, question)
    out["meta"] = {
        **out.get("meta", {}),
        "skill": skill_name,
        "fallback_to": BASELINE,
        "fallback_reason": str(error),
    }
    return out


# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# Round N skill classes (added below each round)
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# Round 1 — H1: siglip_mmr_diverse
#            H2: siglip_spatial_cooccur
#            H3: dino_boundary_dense
# ─────────────────────────────────────────────

_DINO_STATE: Dict[str, Any] | None = None

_SPATIAL_KWS: Tuple[str, ...] = (
    "left", "right", "facing", "standing", "front", "behind",
    "next to", "beside", "appear", "order", "first-time",
    "clockwise", "counterclockwise", "opposite", "adjacent",
)
_TEMPORAL_KWS: Tuple[str, ...] = (
    "sequence", "order", "after", "before", "first", "last", "then",
    "triggered", "caused", "steps", "happens next", "chronological",
    "transition", "confronted", "leaving", "simultaneous",
)


_DINO_LOAD_FAILED: bool = False   # sentinel: don't retry after failure

_DINO_SNAPSHOT = "facebook/dinov3-vitl16-pretrain-lvd1689m"


def _load_dino_state() -> Dict[str, Any]:
    global _DINO_STATE, _DINO_LOAD_FAILED
    if _DINO_STATE is not None:
        return _DINO_STATE
    if _DINO_LOAD_FAILED:
        raise RuntimeError("DINOv3 load already failed — not retrying")
    import torch
    from transformers import AutoImageProcessor, AutoModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[skill] Loading DINOv3-ViT-L/16 on {device} ...", flush=True)
    try:
        model = AutoModel.from_pretrained(
            _DINO_SNAPSHOT, trust_remote_code=True,
        ).to(device=device, dtype=dtype).eval()
        processor = AutoImageProcessor.from_pretrained(
            _DINO_SNAPSHOT, trust_remote_code=True,
        )
    except Exception as e:
        _DINO_LOAD_FAILED = True
        raise RuntimeError(f"DINOv3 load failed: {e}") from e
    _DINO_STATE = {"model": model, "processor": processor, "device": device, "dtype": dtype}
    print(f"[skill] DINOv3-ViT-L/16 loaded OK on {device} ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)", flush=True)
    return _DINO_STATE


def _score_frames_and_feats(
    frames: List[Image.Image], question: str, batch_size: int = 16
) -> Tuple[np.ndarray, np.ndarray]:
    """SigLIP: return (scores [N], image_feats [N, D]) in one pass."""
    import torch
    state = _load_clip_state()
    model = state["model"]
    processor = state["processor"]
    device = state["device"]
    dtype = state["dtype"]
    text_inputs = processor(
        text=[question[:512]], padding="max_length", truncation=True, return_tensors="pt"
    )
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with torch.no_grad():
        text_out = model.get_text_features(**text_inputs)
        tf = text_out.pooler_output if hasattr(text_out, "pooler_output") else text_out
        tf = tf / tf.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    scores: List[float] = []
    feat_chunks: List[np.ndarray] = []
    for start in range(0, len(frames), batch_size):
        batch = frames[start : start + batch_size]
        img_inputs = processor(images=batch, return_tensors="pt")
        pv = img_inputs["pixel_values"].to(device=device, dtype=dtype)
        with torch.no_grad():
            img_out = model.get_image_features(pixel_values=pv)
            imf = img_out.pooler_output if hasattr(img_out, "pooler_output") else img_out
            imf = imf / imf.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            scores.extend((imf * tf).sum(dim=-1).float().cpu().tolist())
            feat_chunks.append(imf.float().cpu().numpy())
    scores_arr = np.asarray(scores, dtype=np.float32)
    feats_arr = np.concatenate(feat_chunks, axis=0) if feat_chunks else np.zeros((len(frames), 1))
    return scores_arr, feats_arr


def _mmr_select(
    scores: np.ndarray, feats_norm: np.ndarray, k: int, alpha: float = 0.5
) -> List[int]:
    """Maximal Marginal Relevance: iteratively pick argmax(alpha*score - (1-alpha)*max_sim_to_selected).
    Approximates MDP3 DPP (ICCV 2025, arXiv 2501.02885)."""
    n = len(scores)
    if n <= k:
        return list(range(n))
    scores_n = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    first = int(np.argmax(scores_n))
    selected = [first]
    remaining = [i for i in range(n) if i != first]
    while len(selected) < k and remaining:
        sel_feats = feats_norm[selected]
        cand_feats = feats_norm[remaining]
        max_sim = (cand_feats @ sel_feats.T).max(axis=1)
        cand_sc = scores_n[remaining]
        mmr = alpha * cand_sc - (1.0 - alpha) * max_sim
        best_local = int(np.argmax(mmr))
        selected.append(remaining[best_local])
        remaining.pop(best_local)
    return sorted(selected)


def _extract_spatial_objects(question: str) -> List[str]:
    """Extract object names from spatial-relationship questions."""
    objs: List[str] = []
    for m in _re.finditer(
        r"\b(?:by|near|next to|facing|between|beside|toward|from)\s+(?:the\s+)?"
        r"([a-z][a-z\s]{1,24}?)(?=\s+(?:and|or|,|\?|\band\b)|$)",
        question, _re.IGNORECASE,
    ):
        obj = m.group(1).strip().rstrip(".,;:")
        if 1 < len(obj) < 25:
            objs.append(obj)
    m2 = _re.search(
        r"(?:following|categories|objects)\s*[:\-]\s*([^\?\.]+)",
        question, _re.IGNORECASE,
    )
    if m2:
        objs.extend(
            x.strip() for x in _re.split(r"[,;]", m2.group(1)) if x.strip()
        )
    # R4 fix: also extract the directional TARGET object ("backpack" in "is the backpack to my left?")
    target_m = _re.search(
        r"is\s+(?:the\s+)?([a-zA-Z][a-zA-Z\s]{1,20}?)\s+to\s+(?:my\s+)?(?:left|right|back|front|behind)",
        question,
        _re.IGNORECASE,
    )
    if target_m:
        target_obj = target_m.group(1).strip().rstrip(".,;:")
        if 1 < len(target_obj) < 25:
            objs.insert(0, target_obj)

    seen: set = set()
    result: List[str] = []
    for o in objs:
        if o.lower() not in seen and len(o) > 1:
            seen.add(o.lower())
            result.append(o)
    return result[:4]


def _compute_signals(vpath: str, question: str) -> Dict[str, float]:
    """Compute 5 routing signals from (video, question). Fast: uses 32 uniform frames.
    Literature basis: Signal Library defined in video-skill-loop/SKILL.md."""
    import torch

    defaults = {
        "relevance_peakedness": 1.0,
        "relevance_entropy": 3.0,
        "frame_diversity": 0.5,
        "duration_s": 60.0,
        "boundary_density": 1.0,
    }
    try:
        vr, _, dur = _read_video(vpath)
    except Exception:
        return defaults

    duration_s = float(dur)

    # boundary_density: pixel-L1 coefficient of variation on ~1fps subsample; no model needed
    try:
        n_bnd = min(64, max(8, int(dur)))
        bnd_idx = np.linspace(0, len(vr) - 1, n_bnd, dtype=int).tolist()
        bnd_frames = _get_frames_chunked(vr, bnd_idx)
        bnd_arr = np.stack([np.asarray(f, dtype=np.float32) for f in bnd_frames])
        diffs = np.mean(np.abs(bnd_arr[1:] - bnd_arr[:-1]), axis=(1, 2, 3))
        boundary_density = float(np.std(diffs) / (np.mean(diffs) + 1e-6))
    except Exception:
        boundary_density = defaults["boundary_density"]

    # SigLIP-based signals on 32 uniform frames
    text_probe_sparsity = 0.5  # default: neutral
    try:
        sig_idx = np.linspace(0, len(vr) - 1, 32, dtype=int).tolist()
        sig_frames = _get_frames_chunked(vr, sig_idx)
        scores, feats = _score_frames_and_feats(sig_frames, question)

        peakedness = float(scores.max() / (scores.mean() + 1e-6))

        sm = np.exp(scores - scores.max())
        sm /= sm.sum() + 1e-8
        entropy = float(-np.sum(sm * np.log(sm + 1e-9)))

        sim_mat = feats @ feats.T
        mask = ~np.eye(len(feats), dtype=bool)
        frame_diversity = float(1.0 - sim_mat[mask].mean())

        # text_probe_sparsity: fraction of text-probe SigLIP score in top-10% of frames.
        # High value = text is concentrated in few frames → uniform sampling likely misses it.
        # Literature: HiMu (arXiv 2603.18558) OCR predicate branch insight.
        try:
            TEXT_PROBE = "text written on screen caption subtitle logo label sign banner"
            txt_scores, _ = _score_frames_and_feats(sig_frames, TEXT_PROBE)
            k = max(1, int(0.1 * len(txt_scores)))
            text_probe_sparsity = float(
                np.sort(txt_scores)[-k:].sum() / (txt_scores.sum() + 1e-8)
            )
        except Exception:
            pass
    except Exception:
        peakedness = defaults["relevance_peakedness"]
        entropy = defaults["relevance_entropy"]
        frame_diversity = defaults["frame_diversity"]

    # Text-level routing signals (question-only, zero model inference cost)
    # Calibration (300 samples): ocr p50=6 p75=10 p90=16 | spatial p90=1 | counting p75=1
    _SPATIAL_KW_SET = frozenset([
        "left", "right", "facing", "standing", "front", "behind",
        "next to", "beside", "above", "below", "between", "closest", "farthest",
    ])
    _COUNT_KW_SET = frozenset([
        "how many", "times", "count", "sequence", "order",
        "before", "after", "steps", "number of",
    ])
    q_lower = question.lower()
    ocr_query_score      = float(len(_re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", question)))
    spatial_query_score  = float(sum(1 for kw in _SPATIAL_KW_SET if kw in q_lower))
    counting_query_score = float(sum(1 for kw in _COUNT_KW_SET if kw in q_lower))
    # Compound: counting keyword + long video + boundary-rich video.
    # boundary_density (CV of inter-frame L1 diffs at ~1fps) > p50 means the video has
    # meaningful scene transitions → boundary-dense sampling can find event instances.
    # Routes ~10-15% (counting questions in long, visually dynamic videos).
    # Literature: WFS-SB (arXiv 2603.00512); Codex review Round 2 (Signal C redesign).
    # Threshold boundary_density > 0.57 = p75 from 60-sample calibration (to be refined R3).
    # p50=0.420, p75=0.573, p90=0.811 — routes top ~25% most dynamic videos.
    counting_rich_score = (
        counting_query_score
        if (duration_s > 600.0 and boundary_density > 0.57)
        else 0.0
    )
    # Legacy alias kept for compatibility
    counting_long_score  = counting_query_score if duration_s > 600.0 else 0.0
    # Compound: spatial keyword + static scene (frame_diversity < p50=0.4) → reliable routing
    spatial_static_score = spatial_query_score if frame_diversity < 0.4 else 0.0
    # Compound: OCR routing — question signals text need AND video has sparse text frames.
    # text_probe_sparsity > 0.5 means top-10% frames hold >50% of text-probe SigLIP score
    # → text is concentrated in few frames → uniform sampling likely misses them.
    # Literature: HiMu (arXiv 2603.18558), Codex review Round 2.
    _TEXT_KW_SET = frozenset([
        "text", "caption", "subtitle", "label", "logo", "written", "sign",
        "banner", "title", "word", "read", "says", "writes", "screen",
    ])
    has_text_kw = any(kw in q_lower for kw in _TEXT_KW_SET)
    question_wants_text = (ocr_query_score >= 10.0 or (has_text_kw and ocr_query_score >= 4.0))
    # Valid route only when question needs text AND text is sparse in video frames
    ocr_route_score = float(question_wants_text and text_probe_sparsity > 0.5)

    # temporal_order_score: binary flag for event-ordering questions.
    # Calibration (300 samples): p75=1.0, 31% of samples active.
    # Routes to savgol_boundary_count which provides dense boundary frames.
    # Fixes temporal_localization×medium −18pp regression caused by siglip_mmr routing.
    # Literature: SemVID (arXiv 2603.05663, 2025).
    _TEMPORAL_ORDER_KW = [
        "sequence", "after", "before", "first", "then", "what happens",
        "order of events", "next", "following", "prior", "subsequently",
        "when does", "at what point", "chronological",
    ]
    temporal_order_score = 1.0 if any(k in q_lower for k in _TEMPORAL_ORDER_KW) else 0.0
    # R4: tighter temporal gate — routes only medium/long videos (not short where siglip_mmr is better,
    # not very_long > 5000s where siglip_mmr also helps). Estimated routing: ~16% vs 31% raw.
    # Literature: EFS (arXiv:2603.00983, 2026) — event-boundary frame selection.
    temporal_medium_score = temporal_order_score if (155 < duration_s < 5000) else 0.0

    return {
        "relevance_peakedness":  peakedness,
        "relevance_entropy":     entropy,
        "frame_diversity":       frame_diversity,
        "duration_s":            duration_s,
        "boundary_density":      boundary_density,
        "text_probe_sparsity":   text_probe_sparsity,
        "ocr_query_score":       ocr_query_score,
        "spatial_query_score":   spatial_query_score,
        "counting_query_score":  counting_query_score,
        "counting_long_score":   counting_long_score,   # legacy
        "counting_rich_score":   counting_rich_score,   # validated: boundary_density > 0.7
        "spatial_static_score":  spatial_static_score,
        "ocr_route_score":       ocr_route_score,       # validated: text_probe_sparsity > 0.5
        "temporal_order_score":  temporal_order_score,  # Round 3: event-ordering questions
        "temporal_medium_score": temporal_medium_score, # Round 4: temporal in medium/long only
        # Round 5 signals:
        # temporal_loc_medium: temporal questions in medium/long videos (300-8000s)
        # discriminability: 45% of temporal_localization (13/29), 13% overall
        # Literature: F2C (arXiv:2510.02262, Oct 2025)
        "temporal_loc_medium":   temporal_order_score if (300 < duration_s < 8000) else 0.0,
        # counting_short_clean_score: counting without temporal-keyword contamination
        # R4 lesson: "before/after/first" appear in 59% of temporal_loc questions
        # Fix: require NO temporal keywords AND short video (<600s)
        "counting_short_clean_score": (
            counting_query_score
            if (duration_s < 600.0 and temporal_order_score <= 0.0)
            else 0.0
        ),
        # Round 6: video_summary_score — broad comprehension/description questions
        # Fires on 14/14 video_summarization samples (100% recall), 22/300 overall (7%)
        # False positives: 5 longform, 1 emotion, 1 counting, 1 spatial — mostly neutral
        # Route to clip_count_temporal: +14pp on video_summarization (57.1%→71.4%)
        "video_summary_score": (
            1.0 if any(kw in q_lower for kw in (
                "main content", "main topic", "main activity", "main plot", "main theme",
                "climax", "throughout the video", "genre", "visual style",
                "primarily engaged", "what activity",
            )) else 0.0
        ),
        # Round 6: pure_reasoning_medium_score — non-spatial/temporal/counting medium-length questions
        # Fires on 41% of longform_reasoning × medium (target cell), 18% overall
        # Route to clip_options_mmr: option-wise CLIP max scoring for pure reasoning
        # Literature: A.I.R. (arXiv:2510.04428, Oct 2025)
        "pure_reasoning_medium_score": (
            1.0 if (155 < duration_s < 5000
                    and temporal_order_score == 0.0
                    and counting_query_score == 0.0
                    and spatial_query_score == 0.0)
            else 0.0
        ),
    }


# ── Round 1 frame strategies ──────────────────

def siglip_mmr_diverse_128(
    vpath: str, question: str, pool_n: int = CLIP_POOL_N, budget: int = CLIP_BUDGET,
    alpha: float = 0.5,
) -> Tuple[List[Image.Image], float]:
    """MMR-based diverse+relevant frame selection.
    Approximates MDP3 DPP (ICCV 2025, arXiv 2501.02885):
    pool 256 → score by SigLIP → MMR(alpha=0.5) to balance relevance + diversity → 128 frames."""
    try:
        vr, _, dur = _read_video(vpath)
        pool_idx = _pool_indices(len(vr), pool_n)
        if not pool_idx:
            return uniform(vpath, budget)
        candidates = _get_frames_chunked(vr, pool_idx)
        scores, feats = _score_frames_and_feats(candidates, question)
        selected_local = _mmr_select(scores, feats, budget, alpha=alpha)
        final_idx = sorted(pool_idx[i] for i in selected_local)
        if not final_idx:
            return uniform(vpath, budget)
        return _get_frames_chunked(vr, final_idx), dur
    except Exception as _e:
        print(f"[skill] siglip_mmr_diverse_128 FALLBACK: {_e}", flush=True)
        return uniform(vpath, budget)


def siglip_spatial_cooccur_128(
    vpath: str, question: str, pool_n: int = CLIP_POOL_N, budget: int = CLIP_BUDGET,
) -> Tuple[List[Image.Image], float]:
    """Co-occurrence frame selection for spatial reasoning questions.
    Finds frames where BOTH queried objects are relevant (min of per-object SigLIP scores).
    Literature: ObjectMLLM (arXiv 2504.07454), Disjoint-3DQA (EMNLP 2025)."""
    try:
        vr, _, dur = _read_video(vpath)
        pool_idx = _pool_indices(len(vr), pool_n)
        if not pool_idx:
            return uniform(vpath, budget)
        candidates = _get_frames_chunked(vr, pool_idx)

        obj_names = _extract_spatial_objects(question)
        if len(obj_names) >= 2:
            s0, _ = _score_frames_and_feats(candidates, obj_names[0])
            s1, _ = _score_frames_and_feats(candidates, obj_names[1])
            cooccur = np.minimum(s0, s1)
        else:
            cooccur, _ = _score_frames_and_feats(candidates, question)

        top_local = np.argsort(-cooccur)[:budget].tolist()
        final_idx = sorted(pool_idx[i] for i in top_local)
        if not final_idx:
            return uniform(vpath, budget)
        return _get_frames_chunked(vr, final_idx), dur
    except Exception as _e:
        print(f"[skill] siglip_spatial_cooccur_128 FALLBACK: {_e}", flush=True)
        return uniform(vpath, budget)


def dino_event_boundary_128(
    vpath: str, question: str,
    budget: int = CLIP_BUDGET,
    max_probe: int = 64,
    dense_window: int = 3,
) -> Tuple[List[Image.Image], float]:
    """Event-anchored dense frame sampling via DINOv2 boundary detection.
    Literature: EFS (arXiv 2603.00983, ICLR 2025) + FlowGEBD (WACV 2024).
    Steps: DINOv2 CLS features → consecutive cosine distances → peaks = event boundaries
           → dense ±window frames around each boundary → fill remainder uniformly."""
    try:
        import torch
        vr, _, dur = _read_video(vpath)

        probe_n = min(max_probe, max(8, int(dur)))
        probe_idx = np.linspace(0, len(vr) - 1, probe_n, dtype=int).tolist()
        probe_frames = _get_frames_chunked(vr, probe_idx)

        state = _load_dino_state()
        dino_model = state["model"]
        dino_proc = state["processor"]
        device = state["device"]
        dtype = state["dtype"]

        all_feats: List[np.ndarray] = []
        for start in range(0, len(probe_frames), 16):
            batch = probe_frames[start : start + 16]
            inputs = dino_proc(images=batch, return_tensors="pt")
            pv = inputs["pixel_values"].to(device=device, dtype=dtype)
            with torch.no_grad():
                out = dino_model(pixel_values=pv)
                cls = out.last_hidden_state[:, 0, :]
                cls = cls / cls.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            all_feats.append(cls.float().cpu().numpy())
        feats = np.concatenate(all_feats, axis=0)  # [probe_n, D]

        sim = np.sum(feats[:-1] * feats[1:], axis=1)  # consecutive cosine sim
        dists = 1.0 - sim                              # higher = more different
        threshold = dists.mean() + 1.0 * dists.std()
        boundary_pos = np.where(dists > threshold)[0]

        radius = max(1, int(dense_window * len(vr) / max(probe_n, 1)))
        boundary_set: set = set()
        for bp in boundary_pos:
            center = probe_idx[bp + 1]
            lo = max(0, center - radius)
            hi = min(len(vr) - 1, center + radius)
            boundary_set.update(int(x) for x in np.linspace(lo, hi, dense_window * 2 + 1, dtype=int))

        uniform_idx = np.linspace(0, len(vr) - 1, budget, dtype=int).tolist()
        combined = sorted(boundary_set | set(int(x) for x in uniform_idx))

        if len(combined) > budget:
            b_sorted = sorted(boundary_set)[:budget]
            fill = [i for i in uniform_idx if int(i) not in boundary_set][: budget - len(b_sorted)]
            combined = sorted(b_sorted + [int(x) for x in fill])
        elif len(combined) < budget:
            seen = set(combined)
            for fi in uniform_idx:
                if int(fi) not in seen:
                    combined.append(int(fi))
                    seen.add(int(fi))
                if len(combined) >= budget:
                    break
            combined = sorted(combined)

        return _get_frames_chunked(vr, combined[:budget]), dur
    except Exception as _e:
        print(f"[skill] dino_event_boundary_128 FALLBACK: {_e}", flush=True)
        return uniform(vpath, budget)


FRAME_STRATEGIES["siglip_mmr_diverse_128"]      = siglip_mmr_diverse_128
FRAME_STRATEGIES["siglip_spatial_cooccur_128"]  = siglip_spatial_cooccur_128
FRAME_STRATEGIES["dino_event_boundary_128"]     = dino_event_boundary_128


# ── Round 1 prompt strategies ─────────────────

def spatial_cooccur_prompt(frames: List[Image.Image], question: str, duration: float) -> list:
    """Spatial-aware prompt: instructs model to identify object positions from frames."""
    user = [_enc(f) for f in frames]
    user.append({
        "type": "text",
        "text": (
            f"Question: {question}\n"
            "Carefully examine the relative positions and spatial arrangement of all mentioned "
            "objects across these frames. Answer with the single letter only (A/B/C/D/E)."
        ),
    })
    return [{"role": "system", "content": _SYS}, {"role": "user", "content": user}]


PROMPT_STRATEGIES["spatial_cooccur"] = spatial_cooccur_prompt


# ── FallbackSkill wrapper (adds try/except to base Skill.run) ─────────────────

@dataclass
class FallbackSkill(Skill):
    """Skill with automatic try/except → fallback to uniform_128_direct on any error."""

    def run(self, vpath: str, question: str) -> dict:
        try:
            return super().run(vpath, question)
        except Exception as e:
            return _fallback_to_baseline(self.name, vpath, question, e)


# ── DispatcherSkill ───────────────────────────

@dataclass
class DispatcherSkill:
    """Routes (vpath, question) to a skill based on computed numeric signals.
    Every rule requires ≥1 numeric signal. Keywords are secondary only."""

    name: str
    rules: List[Dict[str, Any]]
    default_skill: str = BASELINE

    def run(self, vpath: str, question: str) -> dict:
        try:
            sigs = _compute_signals(vpath, question)
            q_lower = question.lower()
            for rule in self.rules:
                sig_val = sigs.get(rule["signal"], 0.0)
                op = rule.get("op", ">")
                thr = rule["threshold"]
                passes_numeric = (sig_val > thr) if op == ">" else (sig_val < thr)
                if not passes_numeric:
                    continue
                kws = rule.get("keywords", ())
                if kws and not any(k in q_lower for k in kws):
                    continue
                target = rule["skill"]
                skill = SKILL_REGISTRY.get(target) or SKILL_REGISTRY[BASELINE]
                out = skill.run(vpath, question)
                out.setdefault("meta", {}).update({
                    "dispatched_from": self.name,
                    "routing_signal": rule["signal"],
                    "routing_value": round(sig_val, 4),
                    "routing_rule": rule.get("label", ""),
                })
                return out
            skill = SKILL_REGISTRY[self.default_skill]
            out = skill.run(vpath, question)
            out.setdefault("meta", {}).update({
                "dispatched_from": self.name,
                "routing_signal": "default",
                "routing_value": 0.0,
                "routing_rule": "fallback",
            })
            return out
        except Exception as e:
            return _fallback_to_baseline(self.name, vpath, question, e)


SKILL_REGISTRY: Dict[str, Any] = {
    BASELINE: Skill(BASELINE, "uniform_128", "direct", ["baseline"]),
    # Round 1 skills
    "siglip_mmr_diverse": FallbackSkill(
        "siglip_mmr_diverse", "siglip_mmr_diverse_128", "direct",
        ["round1", "h1", "insufficient_context"],
    ),
    # siglip_spatial_cooccur: DEPRECATED Round 1 — −1.3% overall, H2 REFUTED.
    # SigLIP co-occurrence is too coarse; GroundingDINO needed. DO NOT ROUTE.
    "dino_boundary_dense": FallbackSkill(
        "dino_boundary_dense", "dino_event_boundary_128", "direct",
        ["round1", "h3", "event_boundary_missing"],
    ),
}

DISPATCHER_REGISTRY: Dict[str, Any] = {}



# ─────────────────────────────────────────────
# Round 2 — H1: siglip_ocr_text_aware
#            H2: gdino_spatial_cooccur
#            H3: savgol_boundary_count
# ─────────────────────────────────────────────

# ── Helpers ──────────────────────────────────


def _video_duration_s(vpath: str) -> float:
    """Return video duration in seconds without loading frames."""
    try:
        _, _, dur = _read_video(vpath)
        return float(dur)
    except Exception:
        return 0.0


_GDINO_STATE: Optional[Dict[str, Any]] = None
_GDINO_LOAD_FAILED: bool = False
_GDINO_SNAPSHOT = "IDEA-Research/grounding-dino-base"


def _load_gdino_state() -> Dict[str, Any]:
    global _GDINO_STATE, _GDINO_LOAD_FAILED
    if _GDINO_STATE is not None:
        return _GDINO_STATE
    if _GDINO_LOAD_FAILED:
        raise RuntimeError("GroundingDINO load already failed — not retrying")
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use float16 on GPU to save memory (Qwen 7B already loaded)
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[skill] Loading GroundingDINO-base on {device} ({dtype}) ...", flush=True)
    try:
        processor = AutoProcessor.from_pretrained(_GDINO_SNAPSHOT)
        model = (AutoModelForZeroShotObjectDetection
                 .from_pretrained(_GDINO_SNAPSHOT, torch_dtype=dtype)
                 .to(device).eval())
        _GDINO_STATE = {"model": model, "processor": processor, "device": device}
        n = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[skill] GroundingDINO-base loaded OK on {device} ({n:.0f}M params, {dtype})", flush=True)
    except Exception as e:
        _GDINO_LOAD_FAILED = True
        print(f"[skill] GroundingDINO load FAILED: {e}", flush=True)
        raise RuntimeError(f"GroundingDINO load failed: {e}") from e
    return _GDINO_STATE


def _extract_two_spatial_objects(question: str) -> Tuple[str, str]:
    """Extract two objects compared spatially from a question string."""
    _STOP = frozenset(
        "the a an is are was were what which where who how this that there here "
        "it its to of and or in on at by for with from as be been being has have "
        "had do does did will would could should may might shall can left right "
        "front behind facing next beside between above below closest farthest "
        "standing sitting positioned located side".split()
    )
    words = [w for w in _re.sub(r"[^\w\s]", "", question.lower()).split()
             if w not in _STOP and len(w) > 2]
    if len(words) >= 2:
        return words[0], words[-1]
    if len(words) == 1:
        return words[0], words[0]
    return "object", "object"


# ── H1: siglip_ocr_text_aware_128 ────────────

def siglip_ocr_text_aware_128(
    vpath: str, question: str, pool_n: int = CLIP_POOL_N, budget: int = CLIP_BUDGET,
) -> Tuple[List[Image.Image], float]:
    """H1: SigLIP dual-prompt selection — question relevance + text-density probe.
    Literature: HiMu (arXiv 2603.18558, 2026), VidText (arXiv 2505.22810, 2025).
    Detection signal: ocr_query_score > 15.5 (≈ p90 named-entity count in question).
    """
    try:
        import torch
        frames, duration = uniform(vpath, pool_n)
        TEXT_PROBE = "text written on screen caption subtitle logo label sign banner title"
        scores_rel, _ = _score_frames_and_feats(frames, question[:300])
        scores_txt, _ = _score_frames_and_feats(frames, TEXT_PROBE)
        combined = 0.6 * scores_rel + 0.4 * scores_txt
        k = min(budget, len(frames))
        indices = sorted(torch.topk(torch.from_numpy(combined).float(), k).indices.tolist())
        return [frames[i] for i in indices], duration
    except Exception as _e:
        print(f"[skill] siglip_ocr_text_aware_128 FALLBACK: {_e}", flush=True)
        return uniform(vpath, budget)


FRAME_STRATEGIES["siglip_ocr_text_aware_128"] = siglip_ocr_text_aware_128


# ── H3: savgol_boundary_count_128 ────────────

def savgol_boundary_count_128(
    vpath: str, question: str, budget: int = CLIP_BUDGET,
) -> Tuple[List[Image.Image], float]:
    """H3: Savitzky-Golay smoothed DINOv3 boundaries for counting questions.
    Literature: WFS-SB (arXiv 2603.00512, 2026), Q-Frame (ICCV 2025, arXiv 2506.22139).
    Detection signal: counting_long_score > 0.5 (counting keyword + duration_s > 600s).
    """
    try:
        import torch
        from scipy.signal import savgol_filter, find_peaks

        vr, _, dur = _read_video(vpath)
        probe_n = min(budget, max(32, int(dur / 5) + 1))
        probe_idx = np.linspace(0, len(vr) - 1, probe_n, dtype=int).tolist()
        probe_frames = _get_frames_chunked(vr, probe_idx)
        duration = float(dur)

        state = _load_dino_state()
        model, processor, device, dtype = (
            state["model"], state["processor"], state["device"], state["dtype"]
        )

        feats = []
        with torch.no_grad():
            for frame in probe_frames:
                inputs = processor(images=frame, return_tensors="pt").to(device)
                inputs = {k: v.to(dtype) if v.dtype.is_floating_point else v
                          for k, v in inputs.items()}
                out = model(**inputs)
                cls = out.last_hidden_state[:, 0, :].squeeze(0).float().cpu().numpy()
                feats.append(cls)

        feats = np.array(feats)
        norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
        feats_n = feats / norms

        if len(feats_n) < 4:
            return _get_frames_chunked(vr, probe_idx[:budget]), duration

        dists = 1.0 - np.sum(feats_n[:-1] * feats_n[1:], axis=1)

        # Savitzky-Golay denoising (WFS-SB style, no pywt needed)
        win = min(7, len(dists))
        win = win if win >= 3 and win % 2 == 1 else max(3, win - (1 - win % 2))
        smoothed = savgol_filter(dists, window_length=win, polyorder=2) if len(dists) >= win else dists

        mn, sd = smoothed.mean(), smoothed.std()
        peaks, _ = find_peaks(smoothed, height=mn + 0.5 * sd, distance=2)

        # Question-relevance weighting (WFS-SB + Q-Frame insight):
        # score each probe frame by SigLIP to bias budget toward counted events,
        # not just any scene-change boundary.
        try:
            rel_scores, _ = _score_frames_and_feats(probe_frames, question[:200])
            rel_arr = rel_scores  # numpy array [probe_n]
        except Exception:
            rel_arr = np.ones(len(probe_frames))

        # boundary_score[i] = peak proximity (1 if within ±3 of a peak, else 0)
        boundary_score = np.zeros(len(probe_frames))
        for pk in peaks:
            for off in range(-3, 4):
                idx = pk + off
                if 0 <= idx < len(probe_frames):
                    boundary_score[idx] = 1.0

        # Combined: 50% boundary proximity + 50% question relevance
        combined = 0.5 * boundary_score + 0.5 * (rel_arr / (rel_arr.max() + 1e-8))

        # Select top-budget frames by combined score (cap per-peak neighborhood to avoid cluster collapse)
        selected: set = set()
        # First pass: up to 6 frames from each peak neighborhood, highest combined first
        for pk in peaks:
            neighborhood = sorted(
                [i for i in range(max(0, pk - 3), min(len(probe_frames), pk + 4))],
                key=lambda i: -combined[i],
            )
            for idx in neighborhood[:6]:
                selected.add(idx)
                if len(selected) >= budget:
                    break
            if len(selected) >= budget:
                break

        # Fill remainder with highest combined-score frames not yet selected
        remaining = sorted(
            [i for i in range(len(probe_frames)) if i not in selected],
            key=lambda i: -combined[i],
        )
        for idx in remaining:
            selected.add(idx)
            if len(selected) >= budget:
                break

        indices = sorted(selected)[:budget]
        return [probe_frames[i] for i in indices], duration

    except Exception as _e:
        print(f"[skill] savgol_boundary_count_128 FALLBACK: {_e}", flush=True)
        return uniform(vpath, budget)


FRAME_STRATEGIES["savgol_boundary_count_128"] = savgol_boundary_count_128


# ── H2: SpatialBboxSkill ─────────────────────


@dataclass
class SpatialBboxSkill(FallbackSkill):
    """H2: GroundingDINO co-occurrence frame selection + bbox centroid text injection.
    Literature: GroundSight (arXiv 2509.25669, KDD 2025), SpatialPIN (NeurIPS 2024).
    Detection signal: spatial_query_score > 0.5 (any directional keyword in question).
    """

    def run(self, vpath: str, question: str) -> Dict[str, Any]:
        try:
            import torch
            probe_frames, duration = uniform(vpath, 64)
            obj1, obj2 = _extract_two_spatial_objects(question)

            state = _load_gdino_state()
            model, processor, device = state["model"], state["processor"], state["device"]

            cooccur_frames: List[Image.Image] = []
            # Track best anchor by highest combined detection confidence (not just first)
            best_anchor = ""
            best_conf   = -1.0

            for frame in probe_frames:
                text_prompt = f"{obj1} . {obj2} ."
                inputs = processor(
                    images=frame, text=text_prompt, return_tensors="pt"
                ).to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                h, w = frame.size[1], frame.size[0]
                result = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs["input_ids"],
                    threshold=0.3,
                    text_threshold=0.25,
                    target_sizes=[(h, w)],
                )[0]

                labels = [lb.lower() for lb in result["labels"]]
                boxes  = result["boxes"]
                scores = result["scores"]

                idx1 = next((i for i, lb in enumerate(labels) if obj1[:4] in lb), None)
                idx2 = next((i for i, lb in enumerate(labels) if obj2[:4] in lb), None)

                if idx1 is not None and idx2 is not None:
                    b1 = boxes[idx1].cpu().tolist()
                    b2 = boxes[idx2].cpu().tolist()
                    cx1 = int((b1[0] + b1[2]) / 2)
                    cy1 = int((b1[1] + b1[3]) / 2)
                    cx2 = int((b2[0] + b2[2]) / 2)
                    cy2 = int((b2[1] + b2[3]) / 2)
                    cooccur_frames.append(frame)
                    # Pick anchor with highest combined detection confidence
                    conf = float(scores[idx1]) + float(scores[idx2])
                    if conf > best_conf:
                        best_conf = conf
                        rel = "to the left of" if cx1 < cx2 else "to the right of"
                        best_anchor = (
                            f"\n[Spatial context: '{obj1}' at pixel ({cx1},{cy1}); "
                            f"'{obj2}' at pixel ({cx2},{cy2}). "
                            f"'{obj1}' appears {rel} '{obj2}'.]"
                        )

            if not cooccur_frames:
                print(
                    f"[skill] {self.name}: no co-occurrence for '{obj1}'/'{obj2}' — falling back",
                    flush=True,
                )
                return SKILL_REGISTRY[BASELINE].run(vpath, question)

            if len(cooccur_frames) < 128:
                extra, _ = uniform(vpath, 128 - len(cooccur_frames))
                cooccur_frames = cooccur_frames + extra
            final_frames = cooccur_frames[:128]

            augmented_q = question + best_anchor
            messages = PROMPT_STRATEGIES["direct"](final_frames, augmented_q, duration)
            return {
                "frames": final_frames,
                "messages": messages,
                "meta": {"skill": self.name, "spatial_anchor": best_anchor},
            }
        except Exception as _e:
            print(f"[skill] {self.name} FALLBACK: {_e}", flush=True)
            return SKILL_REGISTRY[BASELINE].run(vpath, question)


# ── Round 2 registry additions ───────────────

SKILL_REGISTRY["siglip_ocr_text_aware"] = FallbackSkill(
    "siglip_ocr_text_aware", "siglip_ocr_text_aware_128", "direct",
    ["round2", "h1", "ocr_text_required"],
)
SKILL_REGISTRY["savgol_boundary_count"] = FallbackSkill(
    "savgol_boundary_count", "savgol_boundary_count_128", "direct",
    ["round2", "h3", "counting_aggregation"],
)
SKILL_REGISTRY["gdino_spatial_cooccur"] = SpatialBboxSkill(
    "gdino_spatial_cooccur", "uniform_128", "direct",
    ["round2", "h2", "spatial_layout_unclear"],
)

# ── signal_dispatcher_v2 ──────────────────────
# v2.2 — signals redesigned after Codex review (Round 2):
#
# Signal A: ocr_route_score = (question_wants_text) AND (text_probe_sparsity > 0.5)
#   → text is concentrated in few frames; uniform sampling misses them
# Signal B: spatial_static_score = spatial_query_score if frame_diversity < 0.4
#   → directional question + static low-motion scene → objects stable across frames
# Signal C: counting_rich_score = counting_query_score if duration>600 AND boundary_density>0.7
#   → counting question in long video WITH meaningful scene transitions
#   (threshold 0.7 ≈ p50 of boundary_density; to be refined in Round 3 after validation)
#
# Rule 1: spatial_static_score > 0.5  → gdino_spatial_cooccur   (~10%)
# Rule 2: counting_rich_score > 0.5   → savgol_boundary_count   (~10%)
# Rule 3: ocr_route_score > 0.5       → siglip_ocr_text_aware   (~10-15%)
# Rule 4: duration_s > 3600           → siglip_mmr_diverse       (R1 very_long +9pp)
# Rule 5: duration_s < 155            → siglip_mmr_diverse       (R1 short +1pp)
# ELSE:                               → uniform_128_direct

_dispatcher_v2 = DispatcherSkill(
    name="signal_dispatcher_v2",
    rules=[
        {
            "label": "spatial_static_scene",
            "signal": "spatial_static_score",
            "op": ">",
            "threshold": 0.5,
            "skill": "gdino_spatial_cooccur",
        },
        {
            "label": "counting_boundary_rich",
            "signal": "counting_rich_score",
            "op": ">",
            "threshold": 0.5,
            "skill": "savgol_boundary_count",
        },
        {
            "label": "ocr_text_sparse_frames",
            "signal": "ocr_route_score",
            "op": ">",
            "threshold": 0.5,
            "skill": "siglip_ocr_text_aware",
        },
        {
            "label": "very_long_video",
            "signal": "duration_s",
            "op": ">",
            "threshold": 3600.0,
            "skill": "siglip_mmr_diverse",
        },
        {
            "label": "short_video_diverse",
            "signal": "duration_s",
            "op": "<",
            "threshold": 155.0,
            "skill": "siglip_mmr_diverse",
        },
    ],
    default_skill=BASELINE,
)
DISPATCHER_REGISTRY["signal_dispatcher_v2"] = _dispatcher_v2
SKILL_REGISTRY["signal_dispatcher_v2"] = _dispatcher_v2


# ── Round 3 skills ────────────────────────────

# ── H1: siglip_spatial_wide_128 ──────────────

def siglip_spatial_wide_128(
    vpath: str, question: str, budget: int = CLIP_BUDGET,
) -> Tuple[List[Image.Image], float]:
    """H1: SigLIP co-presence probe — find frames where both spatial objects are visible.
    Literature: Out of Sight, Not Out of Context? (EMNLP 2025, arXiv 2505.24257).
    Detection signal: spatial_query_score > 0.5 (directional keyword in question).
    """
    try:
        frames, duration = uniform(vpath, CLIP_POOL_N)
        obj1, obj2 = _extract_two_spatial_objects(question)
        probe_a = f"a scene showing {obj1} and {obj2} together in the same room"
        probe_b = "wide angle overview of a room showing objects in their positions"
        scores_a, _ = _score_frames_and_feats(frames, probe_a)
        scores_b, _ = _score_frames_and_feats(frames, probe_b)
        combined = 0.6 * np.asarray(scores_a) + 0.4 * np.asarray(scores_b)
        k = min(budget, len(frames))
        indices = sorted(np.argsort(-combined)[:k].tolist())
        return [frames[i] for i in indices], duration
    except Exception as _e:
        print(f"[skill] siglip_spatial_wide_128 FALLBACK: {_e}", flush=True)
        return uniform(vpath, budget)


FRAME_STRATEGIES["siglip_spatial_wide_128"] = siglip_spatial_wide_128
SKILL_REGISTRY["siglip_spatial_wide"] = FallbackSkill(
    "siglip_spatial_wide", "siglip_spatial_wide_128", "direct",
    ["round3", "h1", "spatial_layout_unclear"],
)


# ── H3: craft_ocr_dense_128 ──────────────────

def craft_ocr_dense_128(
    vpath: str, question: str, budget: int = CLIP_BUDGET,
) -> Tuple[List[Image.Image], float]:
    """H3: Laplacian-variance text-density frame selection.
    Literature: SFA (arXiv 2511.20190, Nov 2025).
    Detection signal: ocr_query_score > 16 (p90 named-entity count in question).
    """
    try:
        from PIL import ImageFilter

        try:
            import cv2 as _cv2
        except Exception:
            _cv2 = None

        def _text_density(frame: Image.Image) -> float:
            if _cv2 is not None:
                import cv2
                gray = np.asarray(frame.convert("L"), dtype=np.uint8)
                return float(cv2.Laplacian(gray, cv2.CV_64F).var())
            edges = frame.convert("L").filter(ImageFilter.FIND_EDGES)
            return float(np.asarray(edges, dtype=np.float32).mean())

        vr, _meta, duration = _read_video(vpath)
        total = len(vr)
        if total <= 0:
            return uniform(vpath, budget)

        probe_n = min(64, total)
        probe_idx = np.linspace(0, total - 1, probe_n, dtype=int)
        probe_idx = np.unique(probe_idx)
        probe_frames = _get_frames_chunked(vr, probe_idx.tolist())
        probe_idx = probe_idx[: len(probe_frames)]

        if not probe_frames:
            return uniform(vpath, budget)

        text_scores = np.array([_text_density(f) for f in probe_frames], dtype=np.float32)
        dense_count = max(1, int(np.ceil(len(probe_frames) * 0.4)))
        dense_pos = np.argsort(-text_scores)[:dense_count]
        selected = list(probe_idx[dense_pos].astype(int))
        selected_set = set(selected)

        target_k = min(budget, total)
        if len(selected) < target_k:
            fill_n = min(CLIP_POOL_N, total)
            fill_idx = np.unique(np.linspace(0, total - 1, fill_n, dtype=int))
            fill_frames = _get_frames_chunked(vr, fill_idx.tolist())
            fill_idx = fill_idx[: len(fill_frames)]
            if fill_frames:
                rel_scores, _ = _score_frames_and_feats(fill_frames, question)
                for pos in np.argsort(-np.asarray(rel_scores)):
                    idx = int(fill_idx[pos])
                    if idx not in selected_set:
                        selected.append(idx)
                        selected_set.add(idx)
                    if len(selected) >= target_k:
                        break

        if not selected:
            return uniform(vpath, budget)

        frames = _get_frames_chunked(vr, sorted(selected[:target_k]))
        return frames, duration
    except Exception as _e:
        print(f"[skill] craft_ocr_dense_128 FALLBACK: {_e}", flush=True)
        return uniform(vpath, budget)


FRAME_STRATEGIES["craft_ocr_dense_128"] = craft_ocr_dense_128
SKILL_REGISTRY["craft_ocr_dense"] = FallbackSkill(
    "craft_ocr_dense", "craft_ocr_dense_128", "direct",
    ["round3", "h3", "ocr_text_required"],
)




# ── H2: siglip_temporal_event_128 ────────────

def siglip_temporal_event_128(
    vpath: str, question: str, budget: int = CLIP_BUDGET,
) -> Tuple[List[Image.Image], float]:
    """H2: Query-guided event localization via SigLIP temporal peak detection.
    Parse MCQ option phrases as event descriptions, score all pooled frames against
    each event, sample densely around each event's temporal peak.
    Literature: VideoTemp-o3 (arXiv 2602.07801, 2025).
    Detection signal: temporal_order_score > 0.5 (temporal ordering keywords).

    Why better than savgol: savgol detects visual scene-change boundaries (DINOv2 L1
    distance peaks) which may not align with the specific events in the question.
    This skill locates each event directly via SigLIP semantic matching.
    """
    try:
        POOL = 256  # smaller pool for multiple event-phrase scoring

        # --- Parse unique event phrases from MCQ options (A/B/C/D lines) ---
        lines = question.split('\n')
        option_lines = [
            l for l in lines
            if l.strip() and l.strip()[0] in 'ABCDE' and '.' in l
        ]
        event_phrases: List[str] = []
        seen: set = set()
        for line in option_lines:
            text = line.split('.', 1)[-1].strip()
            for part in text.split(','):
                p = part.strip()
                if len(p) > 5 and p.lower() not in seen:
                    seen.add(p.lower())
                    event_phrases.append(p)
        event_phrases = event_phrases[:6]  # cap: at most 6 SigLIP calls

        if not event_phrases:
            # No parseable options — fall through to relevance-only selection
            frames, duration = uniform(vpath, budget)
            rel_scores, _ = _score_frames_and_feats(frames, question[:300])
            k = min(budget, len(frames))
            indices = sorted(np.argsort(-np.asarray(rel_scores))[:k].tolist())
            return [frames[i] for i in indices], duration

        frames, duration = uniform(vpath, POOL)
        n = len(frames)
        # Adaptive window: aim to cover ~budget/n_events frames per event
        window = max(2, budget // (max(len(event_phrases), 1) * 2))

        selected: set = set()
        for phrase in event_phrases:
            scores, _ = _score_frames_and_feats(frames, phrase[:200])
            peak = int(np.argmax(np.asarray(scores)))
            lo = max(0, peak - window)
            hi = min(n - 1, peak + window)
            selected.update(range(lo, hi + 1))

        # Fill remainder with question-relevance frames
        if len(selected) < budget:
            rel_scores, _ = _score_frames_and_feats(frames, question[:300])
            for idx in np.argsort(-np.asarray(rel_scores)):
                if len(selected) >= budget:
                    break
                selected.add(int(idx))

        # Return in temporal order so model sees events sequentially
        indices = sorted(selected)[:budget]
        return [frames[i] for i in indices], duration
    except Exception as _e:
        print(f"[skill] siglip_temporal_event_128 FALLBACK: {_e}", flush=True)
        return uniform(vpath, budget)


FRAME_STRATEGIES["siglip_temporal_event_128"] = siglip_temporal_event_128
SKILL_REGISTRY["siglip_temporal_event"] = FallbackSkill(
    "siglip_temporal_event", "siglip_temporal_event_128", "direct",
    ["round3", "h2", "event_ordering_unclear"],
)


# ══════════════════════════════════════════════
# CLIP ViT-L/14 versions of SigLIP skills
# Strictly follow paper model choice (CLIP used in MDP3, ObjectMLLM, etc.)
# ══════════════════════════════════════════════

_CLIP_VIT_STATE: Optional[Dict[str, Any]] = None
_CLIP_VIT_MODEL  = "openai/clip-vit-large-patch14"
_CLIP_VIT_CACHE  = CLIP_CACHE_DIR  # same HF cache dir


def _load_clip_vit_state() -> Dict[str, Any]:
    """Load CLIP ViT-L/14 (OpenAI). Max text length = 77 tokens."""
    global _CLIP_VIT_STATE
    if _CLIP_VIT_STATE is not None:
        return _CLIP_VIT_STATE
    import torch
    from transformers import AutoProcessor, AutoModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32
    print(f"[skill] Loading CLIP ViT-L/14 on {device} ...", flush=True)
    model = AutoModel.from_pretrained(
        _CLIP_VIT_MODEL,
    ).to(device=device, dtype=dtype).eval()
    processor = AutoProcessor.from_pretrained(
        _CLIP_VIT_MODEL,
    )
    _CLIP_VIT_STATE = {"model": model, "processor": processor,
                       "device": device, "dtype": dtype}
    print(f"[skill] CLIP ViT-L/14 loaded OK on {device}", flush=True)
    return _CLIP_VIT_STATE


def _score_frames_clip(
    frames: List[Image.Image], text: str, batch_size: int = 16
) -> Tuple[np.ndarray, np.ndarray]:
    """CLIP ViT-L/14 version of _score_frames_and_feats.
    Returns (scores [N], feats [N, D]). Text truncated to 77 tokens."""
    import torch
    state = _load_clip_vit_state()
    model, processor, device, dtype = (
        state["model"], state["processor"], state["device"], state["dtype"]
    )
    # CLIP max token length = 77
    text_inputs = processor(
        text=[text[:200]], padding=True, truncation=True,
        max_length=77, return_tensors="pt",
    )
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with torch.no_grad():
        text_out = model.get_text_features(**text_inputs)
        tf = text_out.pooler_output if hasattr(text_out, "pooler_output") else text_out
        tf = tf / tf.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    scores: List[float] = []
    feat_chunks: List[np.ndarray] = []
    for start in range(0, len(frames), batch_size):
        batch = frames[start: start + batch_size]
        img_inputs = processor(images=batch, return_tensors="pt")
        pv = img_inputs["pixel_values"].to(device=device, dtype=dtype)
        with torch.no_grad():
            img_out = model.get_image_features(pixel_values=pv)
            imf = img_out.pooler_output if hasattr(img_out, "pooler_output") else img_out
            imf = imf / imf.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            scores.extend((imf * tf).sum(dim=-1).float().cpu().tolist())
            feat_chunks.append(imf.float().cpu().numpy())
    return np.array(scores, dtype=np.float32), np.concatenate(feat_chunks, axis=0)


# ── clip_mmr_diverse_128 ──────────────────────

def clip_mmr_diverse_128(
    vpath: str, question: str, pool_n: int = CLIP_POOL_N, budget: int = CLIP_BUDGET,
    alpha: float = 0.5,
) -> Tuple[List[Image.Image], float]:
    """CLIP ViT-L/14 MMR frame selection (strict paper model: CLIP as in MDP3 ICCV 2025).
    Replaces siglip_mmr_diverse_128 with the model actually used in the source paper."""
    try:
        vr, _, dur = _read_video(vpath)
        pool_idx = _pool_indices(len(vr), pool_n)
        if not pool_idx:
            return uniform(vpath, budget)
        candidates = _get_frames_chunked(vr, pool_idx)
        scores, feats = _score_frames_clip(candidates, question)
        selected_local = _mmr_select(scores, feats, budget, alpha=alpha)
        final_idx = sorted(pool_idx[i] for i in selected_local)
        if not final_idx:
            return uniform(vpath, budget)
        return _get_frames_chunked(vr, final_idx), dur
    except Exception as _e:
        print(f"[skill] clip_mmr_diverse_128 FALLBACK: {_e}", flush=True)
        return uniform(vpath, budget)


FRAME_STRATEGIES["clip_mmr_diverse_128"] = clip_mmr_diverse_128
SKILL_REGISTRY["clip_mmr_diverse"] = FallbackSkill(
    "clip_mmr_diverse", "clip_mmr_diverse_128", "direct",
    ["clip", "r1_clip", "mmr"],
)


# ── clip_spatial_cooccur_128 ─────────────────

def clip_spatial_cooccur_128(
    vpath: str, question: str, pool_n: int = CLIP_POOL_N, budget: int = CLIP_BUDGET,
) -> Tuple[List[Image.Image], float]:
    """CLIP ViT-L/14 co-occurrence frame selection (strict paper model).
    Replaces siglip_spatial_cooccur_128."""
    try:
        vr, _, dur = _read_video(vpath)
        pool_idx = _pool_indices(len(vr), pool_n)
        if not pool_idx:
            return uniform(vpath, budget)
        candidates = _get_frames_chunked(vr, pool_idx)

        obj_names = _extract_spatial_objects(question)
        if len(obj_names) >= 2:
            s0, _ = _score_frames_clip(candidates, obj_names[0])
            s1, _ = _score_frames_clip(candidates, obj_names[1])
            cooccur = np.minimum(s0, s1)
        else:
            cooccur, _ = _score_frames_clip(candidates, question)

        top_local = np.argsort(-cooccur)[:budget].tolist()
        final_idx = sorted(pool_idx[i] for i in top_local)
        if not final_idx:
            return uniform(vpath, budget)
        return _get_frames_chunked(vr, final_idx), dur
    except Exception as _e:
        print(f"[skill] clip_spatial_cooccur_128 FALLBACK: {_e}", flush=True)
        return uniform(vpath, budget)


FRAME_STRATEGIES["clip_spatial_cooccur_128"] = clip_spatial_cooccur_128
SKILL_REGISTRY["clip_spatial_cooccur"] = FallbackSkill(
    "clip_spatial_cooccur", "clip_spatial_cooccur_128", "direct",
    ["clip", "r1_clip", "spatial_cooccur"],
)


# ── clip_ocr_text_aware_128 ──────────────────

def clip_ocr_text_aware_128(
    vpath: str, question: str, pool_n: int = CLIP_POOL_N, budget: int = CLIP_BUDGET,
) -> Tuple[List[Image.Image], float]:
    """CLIP ViT-L/14 dual-probe OCR text selection (strict paper model).
    Replaces siglip_ocr_text_aware_128."""
    try:
        TEXT_PROBE = "text written on screen caption subtitle logo label sign banner title"
        frames, duration = uniform(vpath, pool_n)
        scores_rel, _ = _score_frames_clip(frames, question)
        scores_txt, _ = _score_frames_clip(frames, TEXT_PROBE)
        combined = 0.6 * scores_rel + 0.4 * scores_txt
        k = min(budget, len(frames))
        indices = sorted(np.argsort(-combined)[:k].tolist())
        return [frames[i] for i in indices], duration
    except Exception as _e:
        print(f"[skill] clip_ocr_text_aware_128 FALLBACK: {_e}", flush=True)
        return uniform(vpath, budget)


FRAME_STRATEGIES["clip_ocr_text_aware_128"] = clip_ocr_text_aware_128
SKILL_REGISTRY["clip_ocr_text_aware"] = FallbackSkill(
    "clip_ocr_text_aware", "clip_ocr_text_aware_128", "direct",
    ["clip", "r2_clip", "ocr_text_required"],
)


# ── Round 4 skills ────────────────────────────

# ── clip_count_temporal_128 ──────────────────
# H1: counting_aggregation — counting×long=0%, counting overall=46.7%.
# CLIP ViT-L/14 query-relevant frame selection with strict temporal spread.
# Literature: Q-Frame (ICCV 2025, arXiv:2506.22139), AKS (CVPR 2025, arXiv:2502.21271).
# Detection signal: counting_rich_score > 0 (counting keywords + duration>600 + boundary_density>0.57).

def clip_count_temporal_128(
    vpath: str, question: str, pool_n: int = CLIP_POOL_N, budget: int = CLIP_BUDGET,
) -> Tuple[List[Image.Image], float]:
    """Counting selector for long/dynamic videos.

    Hypothesis: counting improves when frame selection combines CLIP object relevance
    with strict temporal spread — each temporal bucket yields its most object-relevant frame,
    ensuring all distinct object appearances are captured across the video timeline.
    Literature: Q-Frame (ICCV 2025, arXiv:2506.22139), AKS (CVPR 2025, arXiv:2502.21271).
    Detection signal: counting_rich_score > 0.
    """
    try:
        import re as _re_inner

        question = question or ""

        def _extract_count_target(q: str) -> str:
            patterns = [
                r"how\s+many\s+([a-zA-Z][a-zA-Z\s-]{0,40}?)(?:\s+(?:are|were|was|can|do|does|did|appear|appears|appearing|shown|visible|present|in|on|at|there)\b|[?.!,]|$)",
                r"count\s+(?:the\s+)?([a-zA-Z][a-zA-Z\s-]{0,40}?)(?:\s+(?:in|on|at|from|shown|visible|present)\b|[?.!,]|$)",
                r"number\s+of\s+([a-zA-Z][a-zA-Z\s-]{0,40}?)(?:\s+(?:in|on|at|shown|visible|present)\b|[?.!,]|$)",
            ]
            for pat in patterns:
                m = _re_inner.search(pat, q, flags=_re_inner.IGNORECASE)
                if not m:
                    continue
                target = m.group(1).strip()
                target = _re_inner.sub(r"^(?:the|a|an)\s+", "", target, flags=_re_inner.IGNORECASE)
                target = _re_inner.sub(
                    r"\b(?:that|which|who|are|is|were|was|can|do|does|did|appear|appears|appearing|shown|visible|present)\b.*$",
                    "",
                    target,
                    flags=_re_inner.IGNORECASE,
                )
                target = _re_inner.sub(r"\s+", " ", target).strip(" \t\r\n?.!,;:")
                if 1 < len(target) < 48:
                    return target
            return ""

        vr, fps, duration = _read_video(vpath)
        num_frames = len(vr)
        if num_frames <= 0:
            return uniform(vpath, budget)

        _pool_n = max(1, min(int(pool_n), num_frames))
        target_k = max(1, min(int(budget), num_frames, _pool_n))

        pool_idx = np.asarray(_pool_indices(num_frames, _pool_n), dtype=int)
        if pool_idx.size == 0:
            return uniform(vpath, budget)

        pool_frames = _get_frames_chunked(vr, pool_idx.tolist())
        pool_idx = pool_idx[: len(pool_frames)]
        if pool_idx.size == 0:
            return uniform(vpath, budget)

        target_obj = _extract_count_target(question)
        clip_text = f"a photo of {target_obj}" if target_obj else (question or "objects to count")

        scores, _ = _score_frames_clip(pool_frames, clip_text)
        scores = np.asarray(scores, dtype=np.float32)[: pool_idx.size]

        # Temporal bucket sampling: divide video into target_k buckets, pick best frame per bucket
        bucket_edges = np.linspace(0, num_frames, num=target_k + 1)
        selected_pos: List[int] = []
        selected_set: set = set()

        for bucket_i in range(target_k):
            lo = bucket_edges[bucket_i]
            hi = bucket_edges[bucket_i + 1]
            if bucket_i == target_k - 1:
                members = np.where((pool_idx >= lo) & (pool_idx <= hi - 1))[0]
            else:
                members = np.where((pool_idx >= lo) & (pool_idx < hi))[0]
            if members.size == 0:
                continue
            best_pos = int(members[np.argmax(scores[members])])
            if best_pos not in selected_set:
                selected_pos.append(best_pos)
                selected_set.add(best_pos)

        # Fill remaining slots with globally top-scored frames
        if len(selected_pos) < target_k:
            for pos in np.argsort(-scores):
                pos = int(pos)
                if pos in selected_set:
                    continue
                selected_pos.append(pos)
                selected_set.add(pos)
                if len(selected_pos) >= target_k:
                    break

        chosen_idx = np.sort(pool_idx[np.asarray(selected_pos[:target_k], dtype=int)])
        frames = _get_frames_chunked(vr, chosen_idx.tolist())
        return frames, duration
    except Exception as _e:
        print(f"[skill] clip_count_temporal_128 FALLBACK: {_e}", flush=True)
        return uniform(vpath, budget)


FRAME_STRATEGIES["clip_count_temporal_128"] = clip_count_temporal_128
SKILL_REGISTRY["clip_count_temporal"] = FallbackSkill(
    "clip_count_temporal", "clip_count_temporal_128", "direct",
    ["round4", "h1", "counting_aggregation"],
)


# ── clip_spatial_target_128 ──────────────────
# H2: spatial_layout_unclear — spatial_reasoning×medium=40% (n=10, 3 rounds unsolved).
# CLIP co-occurrence with directional target object extraction (NER fix over R3).
# Literature: Logic-in-Frames (arXiv:2503.13139, 2025), Out of Sight (EMNLP 2025, arXiv:2505.24257).
# Detection signal: spatial_query_score > 0.5.

def clip_spatial_target_128(
    vpath: str, question: str, pool_n: int = CLIP_POOL_N, budget: int = CLIP_BUDGET,
) -> Tuple[List[Image.Image], float]:
    """Spatial selector that explicitly includes the directional target object.

    Hypothesis: spatial reasoning improves when the directional target object AND reference
    objects are all scored jointly — frames showing all mentioned objects are preferred.
    R3 failure: siglip_spatial_wide only extracted reference objects, missed the target
    object ("backpack" in "is the backpack to my left?"). CLIP scoring with mean over all
    objects ensures target-containing frames are prioritized.
    Literature: Logic-in-Frames (arXiv:2503.13139, 2025), Out of Sight (EMNLP 2025).
    Detection signal: spatial_query_score > 0.5.
    """
    try:
        import re as _re_inner

        question = question or ""

        vr, fps, duration = _read_video(vpath)
        num_frames = len(vr)
        if num_frames <= 0:
            return uniform(vpath, budget)

        _pool_n = max(1, min(int(pool_n), num_frames))
        target_k = max(1, min(int(budget), num_frames, _pool_n))

        pool_idx = np.asarray(_pool_indices(num_frames, _pool_n), dtype=int)
        if pool_idx.size == 0:
            return uniform(vpath, budget)

        pool_frames = _get_frames_chunked(vr, pool_idx.tolist())
        pool_idx = pool_idx[: len(pool_frames)]
        if pool_idx.size == 0:
            return uniform(vpath, budget)

        # Extract reference objects (existing helper, now also extracts target via R4 fix)
        reference_objects: List[str] = []
        try:
            reference_objects = list(_extract_spatial_objects(question) or [])
        except Exception:
            reference_objects = []

        # Also inline-extract directional target as backup
        target_obj = ""
        target_m = _re_inner.search(
            r"is\s+(?:the\s+)?([a-zA-Z][a-zA-Z\s]{1,20}?)\s+to\s+(?:my\s+)?(?:left|right|back|front|behind)",
            question,
            _re_inner.IGNORECASE,
        )
        if target_m:
            target_obj = target_m.group(1).strip().rstrip(".,;:")

        # Combine target + reference objects (deduplicated)
        all_objects: List[str] = []
        seen: set = set()
        for obj in ([target_obj] if target_obj else []) + reference_objects:
            obj = str(obj).strip().rstrip(".,;:")
            obj_key = obj.lower()
            if not obj or obj_key in seen:
                continue
            seen.add(obj_key)
            all_objects.append(obj)

        if not all_objects:
            all_objects = [question]

        # Score frames: MEAN over per-object CLIP scores (avoids MIN clustering issue)
        per_object_scores = []
        for obj in all_objects:
            text = obj if obj == question else f"a photo of {obj}"
            scores, _ = _score_frames_clip(pool_frames, text)
            per_object_scores.append(np.asarray(scores, dtype=np.float32)[: pool_idx.size])

        mean_scores = np.mean(np.stack(per_object_scores, axis=0), axis=0)
        top_pos = np.argsort(-mean_scores)[:target_k]
        chosen_idx = np.sort(pool_idx[top_pos])

        frames = _get_frames_chunked(vr, chosen_idx.tolist())
        return frames, duration
    except Exception as _e:
        print(f"[skill] clip_spatial_target_128 FALLBACK: {_e}", flush=True)
        return uniform(vpath, budget)


FRAME_STRATEGIES["clip_spatial_target_128"] = clip_spatial_target_128
SKILL_REGISTRY["clip_spatial_target"] = FallbackSkill(
    "clip_spatial_target", "clip_spatial_target_128", "direct",
    ["round4", "h2", "spatial_layout_unclear"],
)


# ── clip_event_boundary_128 ──────────────────
# H3: routing_regression — temporal_localization×medium regressed −18pp from siglip_mmr.
# DINO event-boundary segmentation + CLIP per-segment query selection.
# Replaces savgol_boundary_count (pixel-L1 diffs) with DINO feature-space boundary detection.
# Literature: EFS (arXiv:2603.00983, 2026), WFS-SB (arXiv:2603.00512, 2026).
# Detection signal: temporal_medium_score > 0.5 (temporal ordering in medium/long videos).

def clip_event_boundary_128(
    vpath: str, question: str, pool_n: int = CLIP_POOL_N, budget: int = CLIP_BUDGET,
) -> Tuple[List[Image.Image], float]:
    """Temporal selector using DINO event boundaries + CLIP per-segment frame selection.

    Hypothesis: temporal localization in medium/long videos improves when:
    (1) DINO feature cosine distance peaks identify event-boundary frames,
    (2) video is segmented at boundaries,
    (3) best CLIP-query-matching frame from each segment is selected.
    This ensures all temporal events are represented and the most relevant frame
    per event is surfaced for sequence-ordering questions.
    Literature: EFS (arXiv:2603.00983, 2026), WFS-SB (arXiv:2603.00512, 2026).
    Detection signal: temporal_medium_score > 0.5 (medium/long temporal questions only).
    """
    try:
        import torch

        question = question or ""

        def _encode_dino_frames_local(frames: List[Image.Image], batch_size: int = 16) -> np.ndarray:
            state = _load_dino_state()
            model = state["model"]
            processor = state["processor"]
            device = state.get("device", "cpu")
            dtype = state.get("dtype", None)
            model.eval()
            chunks = []
            for start in range(0, len(frames), batch_size):
                batch_frames = frames[start:start + batch_size]
                inputs = processor(images=batch_frames, return_tensors="pt")
                if isinstance(inputs, dict):
                    moved = {}
                    for k, v in inputs.items():
                        if hasattr(v, "to"):
                            if k == "pixel_values" and dtype is not None and torch.is_floating_point(v):
                                moved[k] = v.to(device=device, dtype=dtype)
                            else:
                                moved[k] = v.to(device=device)
                        else:
                            moved[k] = v
                    inputs = moved
                elif hasattr(inputs, "to"):
                    try:
                        inputs = inputs.to(device)
                    except Exception:
                        pass
                with torch.inference_mode():
                    outputs = model(**inputs) if isinstance(inputs, dict) else model(inputs)
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    feats = outputs.pooler_output
                elif hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
                    feats = outputs.image_embeds
                elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                    feats = outputs.last_hidden_state[:, 0]
                elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                    feats = outputs[0]
                    if getattr(feats, "ndim", 0) == 3:
                        feats = feats[:, 0]
                else:
                    raise RuntimeError("Unsupported DINO output format")
                feats = feats.float()
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                chunks.append(feats.cpu().numpy())
            if not chunks:
                return np.zeros((0, 1), dtype=np.float32)
            return np.concatenate(chunks, axis=0)

        vr, fps, duration = _read_video(vpath)
        num_frames = len(vr)
        if num_frames <= 0:
            return uniform(vpath, budget)

        # Phase 1: DINO event boundary detection on ~1fps subsample
        detect_n = min(num_frames, max(64, min(128, int(round(duration)) if duration and duration > 0 else 64)))
        detect_idx = np.asarray(_pool_indices(num_frames, detect_n), dtype=int)
        if detect_idx.size < 2:
            return uniform(vpath, budget)

        detect_frames = _get_frames_chunked(vr, detect_idx.tolist())
        detect_idx = detect_idx[: len(detect_frames)]
        if detect_idx.size < 2:
            return uniform(vpath, budget)

        dino_feats = _encode_dino_frames_local(detect_frames)
        dino_feats = dino_feats[: detect_idx.size]
        if dino_feats.shape[0] < 2:
            return uniform(vpath, budget)

        # Inter-frame cosine distance: high value = event boundary
        dists = 1.0 - np.sum(dino_feats[:-1] * dino_feats[1:], axis=1)

        # Find local maxima above mean+0.25*std as event boundaries
        peak_candidates: List[int] = []
        for i in range(len(dists)):
            left = dists[i - 1] if i > 0 else -np.inf
            right = dists[i + 1] if i + 1 < len(dists) else -np.inf
            if dists[i] >= left and dists[i] >= right:
                peak_candidates.append(i)

        peak_candidates_arr = np.asarray(peak_candidates, dtype=int)
        if peak_candidates_arr.size > 0:
            thresh = float(dists.mean() + 0.25 * dists.std())
            strong_peaks = peak_candidates_arr[dists[peak_candidates_arr] >= thresh]
        else:
            strong_peaks = np.asarray([], dtype=int)

        max_boundaries = min(8, max(0, len(dists)))
        min_boundaries = min(2, max(0, len(dists)))

        chosen_peak_pos: List[int] = []
        if strong_peaks.size > 0:
            ordered_strong = strong_peaks[np.argsort(-dists[strong_peaks])]
            chosen_peak_pos.extend(int(x) for x in ordered_strong[:max_boundaries])
        if len(chosen_peak_pos) < min_boundaries:
            for p in np.argsort(-dists):
                p = int(p)
                if p not in chosen_peak_pos:
                    chosen_peak_pos.append(p)
                if len(chosen_peak_pos) >= min_boundaries:
                    break

        chosen_peak_pos = sorted(chosen_peak_pos[:max_boundaries])
        boundary_idx = [int(detect_idx[p + 1]) for p in chosen_peak_pos if 0 <= p + 1 < len(detect_idx)]

        # Build segment edges from boundaries
        segment_edges = [0]
        for idx in boundary_idx:
            if idx > segment_edges[-1]:
                segment_edges.append(idx)
        if segment_edges[-1] < num_frames:
            segment_edges.append(num_frames)

        # Phase 2: CLIP query scoring on full pool
        clip_pool_n = max(1, min(int(pool_n), num_frames))
        target_k = max(1, min(int(budget), num_frames, clip_pool_n))

        clip_idx = np.asarray(_pool_indices(num_frames, clip_pool_n), dtype=int)
        if clip_idx.size == 0:
            return uniform(vpath, budget)

        clip_frames = _get_frames_chunked(vr, clip_idx.tolist())
        clip_idx = clip_idx[: len(clip_frames)]
        if clip_idx.size == 0:
            return uniform(vpath, budget)

        clip_scores, clip_feats = _score_frames_clip(clip_frames, question or "video event")
        clip_scores = np.asarray(clip_scores, dtype=np.float32)[: clip_idx.size]
        clip_feats = np.asarray(clip_feats, dtype=np.float32)[: clip_idx.size]

        # Phase 3: Select best CLIP-scored frame from each segment
        selected_pos: List[int] = []
        selected_set: set = set()

        for seg_i in range(len(segment_edges) - 1):
            lo = segment_edges[seg_i]
            hi = segment_edges[seg_i + 1]
            if seg_i == len(segment_edges) - 2:
                members = np.where((clip_idx >= lo) & (clip_idx <= hi - 1))[0]
            else:
                members = np.where((clip_idx >= lo) & (clip_idx < hi))[0]
            if members.size == 0:
                continue
            best_pos = int(members[np.argmax(clip_scores[members])])
            if best_pos not in selected_set:
                selected_pos.append(best_pos)
                selected_set.add(best_pos)

        # Fill remaining budget with MMR-selected frames for diversity
        try:
            fill_order: List[int] = [int(x) for x in _mmr_select(clip_scores, clip_feats, k=len(clip_idx), alpha=0.5)]
        except Exception:
            fill_order = [int(x) for x in np.argsort(-clip_scores)]

        for pos in fill_order:
            if pos in selected_set:
                continue
            selected_pos.append(pos)
            selected_set.add(pos)
            if len(selected_pos) >= target_k:
                break

        if len(selected_pos) < target_k:
            for pos in np.argsort(-clip_scores):
                pos = int(pos)
                if pos in selected_set:
                    continue
                selected_pos.append(pos)
                selected_set.add(pos)
                if len(selected_pos) >= target_k:
                    break

        chosen_idx = np.sort(clip_idx[np.asarray(selected_pos[:target_k], dtype=int)])
        frames = _get_frames_chunked(vr, chosen_idx.tolist())
        return frames, duration
    except Exception as _e:
        print(f"[skill] clip_event_boundary_128 FALLBACK: {_e}", flush=True)
        return uniform(vpath, budget)


FRAME_STRATEGIES["clip_event_boundary_128"] = clip_event_boundary_128
SKILL_REGISTRY["clip_event_boundary"] = FallbackSkill(
    "clip_event_boundary", "clip_event_boundary_128", "direct",
    ["round4", "h3", "routing_regression"],
)


# ── signal_dispatcher_v4 ──────────────────────
# v4.0 — Round 4: Fixes dispatcher_v3 regression (−2.3%).
# Removes routing to savgol_boundary_count (−2.7%) and siglip_spatial_wide (−1.7%).
# Adds counting_rich_score gate for clip_count_temporal (H1).
# Adds temporal_medium_score gate for clip_event_boundary (H3, tighter: ~16% vs 31%).
# Adds spatial_query_score gate for clip_spatial_target (H2, with target NER fix).
# Uses clip_mmr_diverse (CLIP ViT-L/14) for duration rules (original MDP3 paper model).
#
# Rule 1: counting_rich_score > 0 → clip_count_temporal   (~10-15% routing)
# Rule 2: temporal_medium_score > 0.5 → clip_event_boundary (~16% routing)
# Rule 3: spatial_query_score > 0.5 → clip_spatial_target  (~10% routing)
# Rule 4: duration_s > 3600 → clip_mmr_diverse             (~15% routing)
# Rule 5: duration_s < 155  → clip_mmr_diverse             (~33% routing)
# ELSE: uniform_128_direct

_dur_skill_v4 = "clip_mmr_diverse" if "clip_mmr_diverse" in SKILL_REGISTRY else "siglip_mmr_diverse"

_dispatcher_v4 = DispatcherSkill(
    name="signal_dispatcher_v4",
    rules=[
        {
            "label": "counting_rich_long_dynamic",
            "signal": "counting_rich_score",
            "op": ">",
            "threshold": 0.0,
            "skill": "clip_count_temporal",
        },
        {
            "label": "temporal_ordering_medium_long",
            "signal": "temporal_medium_score",
            "op": ">",
            "threshold": 0.5,
            "skill": "clip_event_boundary",
        },
        {
            "label": "spatial_position_query",
            "signal": "spatial_query_score",
            "op": ">",
            "threshold": 0.5,
            "skill": "clip_spatial_target",
        },
        {
            "label": "very_long_video",
            "signal": "duration_s",
            "op": ">",
            "threshold": 3600.0,
            "skill": _dur_skill_v4,
        },
        {
            "label": "short_video",
            "signal": "duration_s",
            "op": "<",
            "threshold": 155.0,
            "skill": _dur_skill_v4,
        },
    ],
    default_skill=BASELINE,
)
DISPATCHER_REGISTRY["signal_dispatcher_v4"] = _dispatcher_v4
SKILL_REGISTRY["signal_dispatcher_v4"] = _dispatcher_v4


# ─────────────────────────────────────────────
# Round 5 — H1: clip_count_topk
#            H2: clip_temporal_clip
#            H3: gdino_bbox_prompt
# ─────────────────────────────────────────────

# ── H1: clip_count_topk ───────────────────────
# Hypothesis: counting needs dense object-rich frames (alpha=1.0 pure relevance, no temporal diversity)
# Literature: Free Video-LLM (arXiv:2410.10441, 2024); SMI concept (arXiv:2601.07459, 2026)
# Signal: counting_short_clean_score > 0 (short video + counting keywords + no temporal contamination)


def _extract_count_target(question: str) -> str:
    """Extract the counted object from a counting question."""
    patterns = [
        r"how\s+many\s+([a-zA-Z][a-zA-Z\s-]{0,40}?)(?:\s+(?:are|were|was|can|do|does|did|appear|appears|appearing|shown|visible|present|in|on|at|there)\b|[?.!,]|$)",
        r"count\s+(?:the\s+)?([a-zA-Z][a-zA-Z\s-]{0,40}?)(?:\s+(?:in|on|at|shown|visible|present)\b|[?.!,]|$)",
        r"number\s+of\s+([a-zA-Z][a-zA-Z\s-]{0,40}?)(?:\s+(?:in|on|at|shown|visible|present)\b|[?.!,]|$)",
    ]
    for pat in patterns:
        m = _re.search(pat, question, flags=_re.IGNORECASE)
        if not m:
            continue
        target = m.group(1).strip()
        target = _re.sub(r"^(?:the|a|an)\s+", "", target, flags=_re.IGNORECASE)
        target = _re.sub(
            r"\b(?:that|which|who|are|is|were|was|can|do|does|did|appear|appears|appearing|shown|visible|present)\b.*$",
            "", target, flags=_re.IGNORECASE,
        )
        target = _re.sub(r"\s+", " ", target).strip(" \t\r\n?.!,;:")
        if 1 < len(target) < 48:
            return target
    return ""


def clip_count_topk_128(
    vpath: str, question: str, pool_n: int = CLIP_POOL_N, budget: int = CLIP_BUDGET,
) -> Tuple[List[Image.Image], float]:
    """CLIP top-K pure relevance selector for counting in short/medium videos.

    Hypothesis: Counting 'how many X' needs frames where X is most visible,
    NOT temporally diverse frames. alpha=1.0 pure relevance avoids dilution.

    Literature: Free Video-LLM (arXiv:2410.10441, 2024); SMI-aligned high-alpha selection.
    Detection signal: counting_short_clean_score > 0
    """
    try:
        vr, fps, duration = _read_video(vpath)
        num_frames = len(vr)
        if num_frames <= 0:
            return uniform(vpath, budget)

        pool_n_ = max(1, min(int(pool_n), num_frames))
        target_k = max(1, min(int(budget), pool_n_, num_frames))

        pool_idx = _pool_indices(num_frames, pool_n_)
        if not pool_idx:
            return uniform(vpath, budget)

        pool_frames = _get_frames_chunked(vr, pool_idx)
        pool_idx = pool_idx[: len(pool_frames)]
        if len(pool_idx) == 0:
            return uniform(vpath, budget)

        count_target = _extract_count_target(question)
        clip_text = f"how many {count_target} visible" if count_target else question

        scores, _ = _score_frames_clip(pool_frames, clip_text)
        scores = np.asarray(scores, dtype=np.float32)[: len(pool_idx)]

        top_pos = np.argsort(-scores)[:target_k]
        chosen_idx = sorted(int(pool_idx[p]) for p in top_pos)
        if not chosen_idx:
            return uniform(vpath, budget)

        frames = _get_frames_chunked(vr, chosen_idx)
        return frames, float(duration)
    except Exception as _e:
        print(f"[skill] clip_count_topk_128 FALLBACK: {_e}", flush=True)
        return uniform(vpath, budget)


FRAME_STRATEGIES["clip_count_topk_128"] = clip_count_topk_128
SKILL_REGISTRY["clip_count_topk"] = FallbackSkill(
    "clip_count_topk", "clip_count_topk_128", "direct",
    ["round5", "h1", "counting_aggregation"],
)


# ── H2: clip_temporal_clip ────────────────────
# Hypothesis: temporal questions need LOCAL temporal context (adjacent frames around anchor)
#             NOT temporal spread. F2C: anchor + adjacent frames for "when did X happen?"
# Literature: F2C: From Frames to Clips (arXiv:2510.02262, Oct 2025; +8-10% on video benchmarks)
# Signal: temporal_loc_medium > 0 (45% of temporal_localization, 13% overall)


def clip_temporal_clip_128(
    vpath: str, question: str, budget: int = CLIP_BUDGET,
    anchor_radius: int = 3,
) -> Tuple[List[Image.Image], float]:
    """CLIP anchor-frame + temporal clip expansion (F2C approach).

    Selects top-M anchor frames by CLIP relevance, then expands each anchor
    to a short temporal clip (+-anchor_radius adjacent frames). Provides local
    temporal context for 'when did X happen?' questions.

    Literature: F2C / From Frames to Clips (arXiv:2510.02262, Oct 2025).
    Detection signal: temporal_loc_medium > 0
    """
    try:
        vr, fps, duration = _read_video(vpath)
        num_frames = len(vr)
        if num_frames <= 0:
            return uniform(vpath, budget)

        target_k = max(1, min(int(budget), num_frames))
        clip_span = 2 * anchor_radius + 1

        # Build pool capped at 600 frames for memory safety
        pool_size = max(1, min(600, num_frames))
        pool_idx_raw = _pool_indices(num_frames, pool_size)
        pool_idx = np.asarray(pool_idx_raw, dtype=int)
        if pool_idx.size == 0:
            return uniform(vpath, budget)

        pool_frames = _get_frames_chunked(vr, pool_idx.tolist())
        pool_idx = pool_idx[: len(pool_frames)]
        if pool_idx.size == 0:
            return uniform(vpath, budget)

        # Score by CLIP
        scores, _ = _score_frames_clip(pool_frames, question)
        scores = np.asarray(scores, dtype=np.float32)[: pool_idx.size]

        # Select anchors with minimum temporal separation (no overlapping clips)
        max_anchors = max(1, target_k // clip_span)
        ranked_pos = np.argsort(-scores).tolist()

        anchor_pos: List[int] = []
        for pos in ranked_pos:
            if all(abs(pos - prev) > (2 * anchor_radius) for prev in anchor_pos):
                anchor_pos.append(pos)
            if len(anchor_pos) >= max_anchors:
                break
        if not anchor_pos and ranked_pos:
            anchor_pos = [ranked_pos[0]]

        # Expand each anchor to temporal clip
        selected_set: set = set()
        selected_pos: List[int] = []
        for a_pos in sorted(anchor_pos):
            lo = max(0, a_pos - anchor_radius)
            hi = min(int(pool_idx.size) - 1, a_pos + anchor_radius)
            for pos in range(lo, hi + 1):
                if pos not in selected_set:
                    selected_pos.append(pos)
                    selected_set.add(pos)
                    if len(selected_pos) >= target_k:
                        break
            if len(selected_pos) >= target_k:
                break

        # Fill remaining budget with top-ranked frames not yet selected
        if len(selected_pos) < target_k:
            for pos in ranked_pos:
                if pos not in selected_set:
                    selected_pos.append(pos)
                    selected_set.add(pos)
                    if len(selected_pos) >= target_k:
                        break

        chosen_idx = sorted(int(pool_idx[p]) for p in selected_pos[:target_k])
        if not chosen_idx:
            return uniform(vpath, budget)

        frames = _get_frames_chunked(vr, chosen_idx)
        return frames, float(duration)
    except Exception as _e:
        print(f"[skill] clip_temporal_clip_128 FALLBACK: {_e}", flush=True)
        return uniform(vpath, budget)


FRAME_STRATEGIES["clip_temporal_clip_128"] = clip_temporal_clip_128
SKILL_REGISTRY["clip_temporal_clip"] = FallbackSkill(
    "clip_temporal_clip", "clip_temporal_clip_128", "direct",
    ["round5", "h2", "temporal_localization"],
)


# ── H3: gdino_bbox_prompt ─────────────────────
# Hypothesis: Inject GroundingDINO bbox coordinates as text prefix gives model explicit
#             spatial grounding without changing frame selection strategy.
# Literature: Draw-and-Understand (arXiv:2403.20271, ICLR 2025) - bbox text grounding concept
# Signal: spatial_query_score > 0.5 (24% overall, 53% of spatial_reasoning)


class GDinoBBoxPromptSkill:
    """Prompt-strategy spatial grounding: GroundingDINO bbox text injection.

    Runs GroundingDINO on 8 uniform frames, extracts spatial target bbox,
    prepends 'Spatial context: [target] found at [x1,y1,x2,y2] in frame [t/T]'
    to the question. Base frame strategy: clip_mmr_diverse.

    Literature: Draw-and-Understand (arXiv:2403.20271, ICLR 2025).
    Detection signal: spatial_query_score > 0.5
    """

    # Required by generate_trajectories.py (uses FRAME_STRATEGIES[skill.frame_key])
    frame_key  = "clip_mmr_diverse_128"
    prompt_key = "direct"

    def __init__(self, name: str = "gdino_bbox_prompt"):
        self.name = name
        self.tags = ["round5", "h3", "spatial_reasoning"]
        self._base_skill_name = "clip_mmr_diverse"

    def _build_prefix_lines(self, vpath: str, target: str, n_probe: int = 8) -> List[str]:
        """Run GroundingDINO on n_probe uniform frames, return bbox text lines."""
        import torch
        try:
            state = _load_gdino_state()
        except Exception as _gdino_e:
            print(f"[skill] gdino_bbox_prompt: GDino unavailable ({_gdino_e}), skipping bbox injection", flush=True)
            return []
        model = state["model"]
        processor = state["processor"]
        device = state["device"]

        vr, _, duration = _read_video(vpath)
        num_frames = len(vr)
        if num_frames <= 0:
            return []

        probe_idx = _pool_indices(num_frames, min(n_probe, num_frames))
        probe_frames = _get_frames_chunked(vr, probe_idx)
        if not probe_frames:
            return []

        total = len(probe_frames)
        prompt_text = target.lower().strip() + "."
        lines: List[str] = []

        for i, frame in enumerate(probe_frames, start=1):
            try:
                inputs = processor(images=frame, text=prompt_text, return_tensors="pt")
                inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
                with torch.inference_mode():
                    outputs = model(**inputs)
                results = processor.post_process_grounded_object_detection(
                    outputs, inputs["input_ids"],
                    box_threshold=0.35, text_threshold=0.25,
                    target_sizes=[frame.size[::-1]],
                )
                if not results:
                    continue
                boxes = results[0].get("boxes", None)
                scores = results[0].get("scores", None)
                if boxes is None or scores is None:
                    continue
                if hasattr(boxes, "cpu"):
                    boxes = boxes.cpu().numpy()
                if hasattr(scores, "cpu"):
                    scores = scores.cpu().numpy()
                boxes = np.asarray(boxes)
                scores_np = np.asarray(scores, dtype=np.float32)
                if boxes.size == 0 or scores_np.size == 0:
                    continue
                best = int(np.argmax(scores_np))
                x1, y1, x2, y2 = [int(round(float(v))) for v in boxes[best].tolist()]
                lines.append(
                    f"Spatial context: {target} found at [{x1},{y1},{x2},{y2}] in frame [{i}/{total}]"
                )
            except Exception:
                continue
        return lines

    def run(self, vpath: str, question: str) -> dict:
        try:
            question = question or ""
            objs = _extract_spatial_objects(question)
            target = objs[0] if objs else ""

            prefix_lines: List[str] = []
            if target:
                prefix_lines = self._build_prefix_lines(vpath, target)

            # Only prepend when detections found — avoid confusing "no detection" messages
            grounded_question = (
                "\n".join(prefix_lines) + "\n" + question if prefix_lines else question
            )

            base_skill = SKILL_REGISTRY.get(self._base_skill_name) or SKILL_REGISTRY[BASELINE]
            out = base_skill.run(vpath, grounded_question)
            if isinstance(out, dict):
                out.setdefault("meta", {}).update({
                    "skill": self.name,
                    "base_skill": self._base_skill_name,
                    "spatial_target": target,
                    "gdino_prefix_lines": prefix_lines,
                })
            return out
        except Exception as _e:
            print(f"[skill] gdino_bbox_prompt FALLBACK: {_e}", flush=True)
            fallback = SKILL_REGISTRY.get(BASELINE)
            if fallback is None:
                raise
            out = fallback.run(vpath, question)
            if isinstance(out, dict):
                out.setdefault("meta", {}).update({"skill": self.name, "fallback": BASELINE})
            return out


SKILL_REGISTRY["gdino_bbox_prompt"] = GDinoBBoxPromptSkill("gdino_bbox_prompt")




# ─────────────────────────────────────────────
# Round 6 — H2: clip_spatial_prompt
#            H3: clip_options_mmr
# ─────────────────────────────────────────────

# ── H2: clip_spatial_prompt ───────────────────
# H2: spatial_layout_unclear (5 rounds frame-selection failed → try prompt approach)
# Hypothesis: inject 1-line allocentric context (reference point + facing direction)
#             extracted from question text to help model orient spatially
# Literature: Allocentric Perceiver (arXiv:2602.05789, Feb 2026)
# Signal: spatial_query_score > 0 (53% of spatial_reasoning, 24% overall)
# Note: prompt prefix only (NOT CoT) — single factual line prepended to question


class ClipSpatialPromptSkill:
    """Prompt-only spatial context skill: reference + facing text prefix injection.

    Extracts reference_point and facing_direction from question text (regex),
    prepends a single-line spatial context to the question, then delegates to
    clip_mmr_diverse. Falls back to clip_mmr_diverse unchanged if extraction fails.

    Literature: Allocentric Perceiver (arXiv:2602.05789, Feb 2026).
    Detection signal: spatial_query_score > 0
    """

    # Required by generate_trajectories.py (uses FRAME_STRATEGIES[skill.frame_key])
    frame_key  = "clip_mmr_diverse_128"
    prompt_key = "direct"

    def __init__(self, name: str = "clip_spatial_prompt"):
        self.name = name
        self.tags = ["round6", "h2", "spatial_prompt"]
        self._base_skill_candidates = ["clip_mmr_diverse", "siglip_mmr_diverse", BASELINE]

    def _extract_reference_and_facing(self, question: str):
        q = question or ""
        reference_point = ""
        facing_point = ""

        ref_patterns = [
            r"(?:by|beside|near|next to|in front of|behind)\s+(?:the\s+)?([a-zA-Z][a-zA-Z\s]{1,24}?)(?=(?:\s+(?:while|and|facing|looking|toward|towards)\b|[?.!,]|$))",
            r"(?:standing|sitting|positioned|located)\s+(?:by|beside|near|next to|in front of|behind)\s+(?:the\s+)?([a-zA-Z][a-zA-Z\s]{1,24}?)(?=(?:\s+(?:while|and|facing|looking|toward|towards)\b|[?.!,]|$))",
        ]
        face_patterns = [
            r"(?:facing|looking at|toward|towards)\s+(?:the\s+)?([a-zA-Z][a-zA-Z\s]{1,24}?)(?=(?:\s+(?:while|and)\b|[?.!,]|$))",
        ]

        for pat in ref_patterns:
            m = _re.search(pat, q, flags=_re.IGNORECASE)
            if m:
                reference_point = m.group(1).strip().rstrip(".,;:")
                break

        for pat in face_patterns:
            m = _re.search(pat, q, flags=_re.IGNORECASE)
            if m:
                facing_point = m.group(1).strip().rstrip(".,;:")
                break

        if not reference_point:
            try:
                objs = list(_extract_spatial_objects(q) or [])
            except Exception:
                objs = []
            for obj in objs:
                obj = str(obj).strip().rstrip(".,;:")
                if obj and (not facing_point or obj.lower() != facing_point.lower()):
                    reference_point = obj
                    break

        return reference_point, facing_point

    def run(self, vpath: str, question: str) -> dict:
        try:
            question = question or ""
            base_skill = None
            base_name = None
            for n in self._base_skill_candidates:
                s = SKILL_REGISTRY.get(n)
                if s is not None and hasattr(s, "run"):
                    base_skill = s; base_name = n; break
            if base_skill is None:
                raise RuntimeError("No base skill available")

            ref, facing = self._extract_reference_and_facing(question)

            if not ref and not facing:
                out = base_skill.run(vpath, question)
            else:
                if ref and facing:
                    prefix = f"Spatial reference: by {ref}. Facing: {facing}."
                elif ref:
                    prefix = f"Spatial reference: by {ref}."
                else:
                    prefix = f"Spatial reference: facing {facing}."
                grounded_q = prefix + "\n" + question
                out = base_skill.run(vpath, grounded_q)

            if isinstance(out, dict):
                out.setdefault("meta", {}).update({
                    "skill": self.name, "base_skill": base_name,
                    "spatial_ref": ref, "spatial_facing": facing,
                })
            return out
        except Exception as _e:
            print(f"[skill] clip_spatial_prompt FALLBACK: {_e}", flush=True)
            fb = SKILL_REGISTRY.get(BASELINE)
            if fb is None: raise
            out = fb.run(vpath, question)
            if isinstance(out, dict):
                out.setdefault("meta", {}).update({"skill": self.name, "fallback": BASELINE})
            return out


SKILL_REGISTRY["clip_spatial_prompt"] = ClipSpatialPromptSkill("clip_spatial_prompt")


# ── H3: clip_options_mmr ─────────────────────��
# H3: option_aware_frame_selection (longform_reasoning × medium, n=47, +4pp stuck)
# Hypothesis: per-option CLIP scoring + max → finds frames depicting ANY answer option
#             more discriminative than single combined-query embedding
# Literature: A.I.R. (arXiv:2510.04428, Oct 2025) — adaptive query-frame scoring
# Signal: pure_reasoning_medium_score > 0 (41% longform × medium, 18% overall)


def clip_options_mmr_128(
    vpath: str, question: str,
    pool_n: int = CLIP_POOL_N, budget: int = CLIP_BUDGET,
) -> Tuple[List[Image.Image], float]:
    """Option-wise CLIP max scoring followed by MMR selection.

    Parses answer options (A./B./C./D./E.) from question, scores frames against
    each option separately, takes per-frame MAX across options, then applies MMR.
    Finds frames where at least one answer option is visually depicted.

    Literature: A.I.R. (arXiv:2510.04428, Oct 2025).
    Detection signal: pure_reasoning_medium_score > 0
    """
    try:
        import re as _re_opt

        q = question or ""

        def _parse_mcq(text: str):
            marker_re = _re_opt.compile(
                r"(?:^|\s)\(?([A-E])[)\.:\s]", flags=_re_opt.MULTILINE
            )
            matches = list(marker_re.finditer(text))
            if len(matches) < 2:
                return text.strip(), []
            stem = text[: matches[0].start()].strip()
            options = []
            for i, m in enumerate(matches):
                start = m.end()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                opt_text = _re_opt.sub(r"\s+", " ", text[start:end]).strip()
                if opt_text:
                    options.append(opt_text)
            return stem, options

        vr, fps, duration = _read_video(vpath)
        num_frames = len(vr)
        if num_frames <= 0:
            return uniform(vpath, budget)

        pool_n_ = max(1, min(int(pool_n), num_frames))
        target_k = max(1, min(int(budget), pool_n_, num_frames))

        pool_idx = np.asarray(_pool_indices(num_frames, pool_n_), dtype=int)
        if pool_idx.size == 0:
            return uniform(vpath, budget)

        pool_frames = _get_frames_chunked(vr, pool_idx.tolist())
        pool_idx = pool_idx[: len(pool_frames)]
        if pool_idx.size == 0:
            return uniform(vpath, budget)

        stem, options = _parse_mcq(q)
        probes = [f"{stem} {opt}".strip() for opt in options] if options else [q]

        max_scores = None
        feats_ref  = None
        for probe in probes:
            sc, ft = _score_frames_clip(pool_frames, probe)
            sc = np.asarray(sc, dtype=np.float32)[: pool_idx.size]
            ft = np.asarray(ft, dtype=np.float32)[: pool_idx.size]
            if max_scores is None:
                max_scores = sc; feats_ref = ft
            else:
                max_scores = np.maximum(max_scores, sc)

        if max_scores is None or feats_ref is None:
            return uniform(vpath, budget)

        try:
            selected_pos = _mmr_select(max_scores, feats_ref, target_k, alpha=0.8)
        except Exception:
            selected_pos = np.argsort(-max_scores)[:target_k].tolist()

        chosen_idx = np.sort(pool_idx[np.asarray(selected_pos[:target_k], dtype=int)])
        frames = _get_frames_chunked(vr, chosen_idx.tolist())
        return frames, float(duration)
    except Exception as _e:
        print(f"[skill] clip_options_mmr_128 FALLBACK: {_e}", flush=True)
        return uniform(vpath, budget)


FRAME_STRATEGIES["clip_options_mmr_128"] = clip_options_mmr_128
SKILL_REGISTRY["clip_options_mmr"] = FallbackSkill(
    "clip_options_mmr", "clip_options_mmr_128", "direct",
    ["round6", "h3", "option_max"],
)


# ── signal_dispatcher_v6 ──────────────────────
# v6.0 — Round 6: Fixes v5 regressions + adds video_summary routing
#
# Fix 1: temporal_loc_medium → clip_count_temporal  (v5 routed to clip_temporal_clip = WRONG)
# Fix 2: spatial_query_score > 1.0  (raised from > 0.5, reduces false positives on
#         obj_scene_recognition × medium −17pp and counting × medium −25pp regressions)
# New 3: video_summary_score → clip_count_temporal  (+14pp on video_summarization, 100% recall)
# New 4: pure_reasoning_medium_score → clip_options_mmr  (H3 option-aware, 41% longform × medium)
# Default: clip_mmr_diverse  (+2.7pp baseline gain, same as v5)
#
# Routing rates:
#   temporal_loc_medium > 0     : ~13% (38/300)
#   video_summary_score > 0     : ~7%  (22/300)
#   spatial_query_score > 1.0   : ~10% (requires ≥2 spatial keywords)
#   pure_reasoning_medium_score : ~18% (55/300)
#   DEFAULT clip_mmr_diverse    : remaining ~52%

_default_v6 = (
    "clip_mmr_diverse"
    if "clip_mmr_diverse" in SKILL_REGISTRY
    else ("siglip_mmr_diverse" if "siglip_mmr_diverse" in SKILL_REGISTRY else BASELINE)
)

_dispatcher_v6 = DispatcherSkill(
    name="signal_dispatcher_v6",
    rules=[
        {
            "label": "temporal_localization_medium_long",
            "signal": "temporal_loc_medium",
            "op": ">",
            "threshold": 0.0,
            "skill": "clip_count_temporal",
        },
        {
            "label": "video_summarization_broad_comprehension",
            "signal": "video_summary_score",
            "op": ">",
            "threshold": 0.0,
            "skill": "clip_count_temporal",
        },
        {
            "label": "pure_reasoning_medium_option_aware",
            "signal": "pure_reasoning_medium_score",
            "op": ">",
            "threshold": 0.0,
            "skill": "clip_options_mmr",
        },
        {
            "label": "strong_spatial_query",
            "signal": "spatial_query_score",
            "op": ">",
            "threshold": 1.0,
            "skill": "gdino_bbox_prompt",
        },
    ],
    default_skill=_default_v6,
)
DISPATCHER_REGISTRY["signal_dispatcher_v6"] = _dispatcher_v6
SKILL_REGISTRY["signal_dispatcher_v6"] = _dispatcher_v6


# ── Round 7 — H2: clip_text_dense ────────────────────────────────────
# H2: ocr_text_required (longform_reasoning × medium, 3 cells persistent)
# Hypothesis: critical textual anchors (branding, credits, overlays) absent from MMR-selected
#             frames. Dual CLIP probe: relevance + text-density → text-rich relevant frames.
# Literature: HiMu (arXiv:2603.18558, 2026); VidText (arXiv:2505.22810, 2025)
# Signal: pure_reasoning_medium_score > 0 (41% of longform×medium, 18% overall)

def clip_text_dense_128(
    vpath: str,
    question: str,
    pool_n: int = CLIP_POOL_N,
    budget: int = CLIP_BUDGET,
) -> Tuple[List[Image.Image], float]:
    """Dual-probe CLIP frame selector: relevance + text-density → text-rich relevant frames.

    Combines CLIP question-relevance score with CLIP text-density probe score.
    Uses MMR on question features for final diversity. Targets videos where critical
    textual anchors (logos, credits, branding overlays) are needed for reasoning.

    Literature: HiMu (arXiv:2603.18558, 2026); VidText (arXiv:2505.22810, 2025).
    Detection signal: pure_reasoning_medium_score > 0
    """
    TEXT_PROBE = "text written on screen caption subtitle logo label sign banner title"
    try:
        vr, fps, duration = _read_video(vpath)
        num_frames = len(vr)
        if num_frames <= 0:
            return uniform(vpath, budget)

        pool_n_ = max(1, min(int(pool_n), num_frames))
        target_k = max(1, min(int(budget), pool_n_, num_frames))

        pool_idx = _pool_indices(num_frames, pool_n_)
        if not pool_idx:
            return uniform(vpath, budget)

        pool_frames = _get_frames_chunked(vr, pool_idx)
        pool_idx = pool_idx[: len(pool_frames)]
        if len(pool_idx) == 0:
            return uniform(vpath, budget)

        # Score with question probe
        q_scores, q_feats = _score_frames_clip(pool_frames, question)
        q_scores = np.asarray(q_scores, dtype=np.float32)[: len(pool_idx)]

        # Score with text-density probe
        t_scores, _ = _score_frames_clip(pool_frames, TEXT_PROBE)
        t_scores = np.asarray(t_scores, dtype=np.float32)[: len(pool_idx)]

        # Normalize each probe to [0, 1]
        def _norm(arr: np.ndarray) -> np.ndarray:
            lo, hi = arr.min(), arr.max()
            if hi - lo < 1e-8:
                return np.zeros_like(arr)
            return (arr - lo) / (hi - lo)

        q_norm = _norm(q_scores)
        t_norm = _norm(t_scores)
        combined = 0.6 * q_norm + 0.4 * t_norm

        # MMR with question features for diversity
        q_feats_arr = np.asarray(q_feats, dtype=np.float32)[: len(pool_idx)]
        chosen_pos = _mmr_select(combined, q_feats_arr, target_k, alpha=0.7)

        chosen_idx = sorted(int(pool_idx[p]) for p in chosen_pos)
        if not chosen_idx:
            return uniform(vpath, budget)

        frames = _get_frames_chunked(vr, chosen_idx)
        return frames, float(duration)
    except Exception as _e:
        print(f"[skill] clip_text_dense_128 FALLBACK: {_e}", flush=True)
        return uniform(vpath, budget)


FRAME_STRATEGIES["clip_text_dense_128"] = clip_text_dense_128
SKILL_REGISTRY["clip_text_dense"] = FallbackSkill(
    "clip_text_dense", "clip_text_dense_128", "direct",
    ["round7", "h2", "ocr_text_required"],
)


# ── signal_dispatcher_v7 ──────────────────────────────────────────────
# v7.0 — Round 7: KEY CHANGE: clip_options_mmr REMOVED (regression −1.0pp)
# pure_reasoning_medium → clip_text_dense (addresses ocr_text_required anchor loss)
# All other rules preserved from v6; default = clip_mmr_diverse

_default_v7 = (
    "clip_mmr_diverse"
    if "clip_mmr_diverse" in SKILL_REGISTRY
    else ("siglip_mmr_diverse" if "siglip_mmr_diverse" in SKILL_REGISTRY else BASELINE)
)

_dispatcher_v7 = DispatcherSkill(
    name="signal_dispatcher_v7",
    rules=[
        {
            "label": "temporal_localization_medium_long",
            "signal": "temporal_loc_medium",
            "op": ">",
            "threshold": 0.0,
            "skill": "clip_count_temporal",
        },
        {
            "label": "video_summarization_broad_comprehension",
            "signal": "video_summary_score",
            "op": ">",
            "threshold": 0.0,
            "skill": "clip_count_temporal",
        },
        {
            "label": "pure_reasoning_medium_text_dense",
            "signal": "pure_reasoning_medium_score",
            "op": ">",
            "threshold": 0.0,
            "skill": "clip_text_dense",
        },
        {
            "label": "strong_spatial_query",
            "signal": "spatial_query_score",
            "op": ">",
            "threshold": 1.0,
            "skill": "gdino_bbox_prompt",
        },
    ],
    default_skill=_default_v7,
)
DISPATCHER_REGISTRY["signal_dispatcher_v7"] = _dispatcher_v7
SKILL_REGISTRY["signal_dispatcher_v7"] = _dispatcher_v7


# ═══════════════════════════════════════════════════════════════════
# ── Portfolio Fusion (Round-0, pre-registered V0, 2026-07-08) ──────
# Contribution line: self-evolving portfolio COMPOSITION — components come
# from earlier loop rounds (literature-derived); the composition below was
# selected by simulated-evolution Round-1 on dev matrices (MLVU-dev+VideoMME):
#   greedy max-coverage anchor+satellites; mechanism-diversity intuition
#   REFUTED by data (count_topk~spatial_cooccur rescue Jaccard=0.90);
#   quotas ∝ marginal coverage. Frozen before any held-out contact.
# Design: ONE shared 256-candidate pool, per-component RANKINGS over the
# same pool, quota merge with proximity dedup, single decode+extract,
# single VLM pass. Anti-dilution: anchor keeps 60% of budget.
# ═══════════════════════════════════════════════════════════════════

def _get_portfolio_quotas() -> List[Tuple[str, int]]:
    """Quotas via env PORTFOLIO_QUOTAS="anchor,topk,temporal,ocr" (default V0=76,32,12,8).
    Round-2 variants: V0b=96,20,8,4  V0c=104,14,6,4 (anti-dilution, from smoke decomposition)."""
    import os as _os_pf
    env = _os_pf.environ.get("PORTFOLIO_QUOTAS", "76,32,12,8")
    a, b, c, d = [int(x) for x in env.split(",")]
    return [("anchor_mmr", a), ("count_topk", b), ("count_temporal", c), ("ocr_text", d)]
PORTFOLIO_DEDUP_FRAC = 0.004   # ~1 candidate slot at pool=256


def portfolio_fusion_frames(
    vpath: str, question: str, pool_n: int = CLIP_POOL_N, budget: int = CLIP_BUDGET,
) -> Tuple[List[Image.Image], float]:
    """V0 portfolio fusion: merge anchor+satellite frame picks into one budget.
    Detection signal: none (applies to every sample — deployment is a single skill).
    Frozen config; see header comment for provenance."""
    try:
        question = question or ""
        vr, _, dur = _read_video(vpath)
        n_frames = len(vr)
        if n_frames <= 0:
            return uniform(vpath, budget)
        pool_idx = _pool_indices(n_frames, pool_n)
        if not pool_idx:
            return uniform(vpath, budget)
        candidates = _get_frames_chunked(vr, pool_idx)
        n = len(candidates)
        pool_idx = [int(x) for x in pool_idx[:n]]
        if n <= budget:
            return candidates, dur

        # ── component rankings over the SHARED candidate pool (local positions) ──
        sc_sig, feats_sig = _score_frames_and_feats(candidates, question[:300])
        sc_sig = np.asarray(sc_sig, dtype=np.float32)[:n]

        pf_quotas = _get_portfolio_quotas()
        anchor_quota = pf_quotas[0][1]
        anchor_set = _mmr_select(sc_sig, feats_sig, min(anchor_quota, n), alpha=0.5)

        tgt = _extract_count_target(question)
        clip_text = f"how many {tgt} visible" if tgt else question
        sc_clip, _ = _score_frames_clip(candidates, clip_text)
        sc_clip = np.asarray(sc_clip, dtype=np.float32)[:n]
        rank_topk = np.argsort(-sc_clip).tolist()

        n_buckets = 32
        per_bucket = [b[np.argsort(-sc_clip[b])].tolist()
                      for b in np.array_split(np.arange(n), n_buckets) if len(b)]
        rank_temporal: List[int] = []
        depth = 0
        while any(depth < len(b) for b in per_bucket):
            for b in per_bucket:
                if depth < len(b):
                    rank_temporal.append(int(b[depth]))
            depth += 1

        TEXT_PROBE = "text written on screen caption subtitle logo label sign banner title"
        sc_txt, _ = _score_frames_and_feats(candidates, TEXT_PROBE)
        sc_txt = np.asarray(sc_txt, dtype=np.float32)[:n]
        rank_ocr = np.argsort(-(0.6 * sc_sig + 0.4 * sc_txt)).tolist()

        rankings = {
            "anchor_mmr":     list(anchor_set) + np.argsort(-sc_sig).tolist(),
            "count_topk":     rank_topk,
            "count_temporal": rank_temporal,
            "ocr_text":       rank_ocr,
        }

        # ── quota merge with proximity dedup (original index space) ──
        radius = max(1, int(PORTFOLIO_DEDUP_FRAC * n_frames))
        kept_local: List[int] = []
        kept_orig: List[int] = []

        def _try_add(lp: int) -> bool:
            if lp < 0 or lp >= n or lp in kept_local:
                return False
            oi = pool_idx[lp]
            if any(abs(oi - ko) <= radius for ko in kept_orig):
                return False
            kept_local.append(lp)
            kept_orig.append(oi)
            return True

        cursors = {k: 0 for k in rankings}

        def _pull(name: str) -> bool:
            r = rankings[name]
            while cursors[name] < len(r):
                lp = int(r[cursors[name]])
                cursors[name] += 1
                if _try_add(lp):
                    return True
            return False

        for name, quota in pf_quotas:
            got = 0
            while got < quota and len(kept_local) < budget and _pull(name):
                got += 1

        while len(kept_local) < budget:
            progressed = False
            for name, _q in pf_quotas:
                if len(kept_local) >= budget:
                    break
                if _pull(name):
                    progressed = True
            if not progressed:
                break
        if len(kept_local) < budget:           # relax dedup: exact-unique fill
            for lp in rankings["anchor_mmr"]:
                lp = int(lp)
                if lp not in kept_local:
                    kept_local.append(lp)
                    kept_orig.append(pool_idx[lp])
                if len(kept_local) >= budget:
                    break

        final_idx = sorted(set(kept_orig))[:budget]
        if not final_idx:
            return uniform(vpath, budget)
        return _get_frames_chunked(vr, final_idx), dur
    except Exception as _e:
        print(f"[skill] portfolio_fusion FALLBACK: {_e}", flush=True)
        return uniform(vpath, budget)


FRAME_STRATEGIES["portfolio_fusion_128"] = portfolio_fusion_frames
SKILL_REGISTRY["portfolio_fusion"] = FallbackSkill(
    "portfolio_fusion", "portfolio_fusion_128", "direct",
    ["fusion", "portfolio", "round0_v0"],
)


# ── Ported from P610 backup (backup_260610/lmms-eval/skill_learning/skills.py) ──
# Adds skill_package_top10 members not present in this codebase:
# mmr_diversity_spatial, dino_episode_anchor, dino_context_expand,
# dino_wide_context, siglip_segment_query_anchor

DINO_SMALL_MODEL_NAME = "facebook/dinov2-small"
_DINO_SMALL_STATE: Optional[Dict[str, Any]] = None

TEMPORAL_LOC_KEYWORDS = (
    "before", "after", "then", "next", "following", "first", "last",
    "start", "began", "begin", "end", "ended", "happen", "happened",
    "happens", "during", "when", "while", "earlier", "later",
)

def _build_skill_output(
    skill: "FallbackSkill",
    frames: List[Image.Image],
    question: str,
    duration: float,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> dict:
    meta: Dict[str, Any] = {
        "skill": skill.name,
        "frame_key": skill.frame_key,
        "prompt_key": skill.prompt_key,
        "n_frames": len(frames),
        "question": question,
        "tags": list(skill.tags),
    }
    if extra_meta:
        meta.update(extra_meta)
    return {
        "frames": frames,
        "messages": PROMPT_STRATEGIES["direct"](frames, question, duration),
        "duration": duration,
        "meta": meta,
    }


def mmr_diversity_spatial_frames(vpath: str, question: str) -> Tuple[List[Image.Image], float]:
    vr, _, dur = _read_video(vpath)
    n_frames = len(vr)
    if n_frames <= 128:
        return uniform(vpath, 128)

    cand_idx = _pool_indices(n_frames, 256)
    cand_frames = _get_frames_chunked(vr, cand_idx)
    scores, feats = _score_frames_and_feats(cand_frames, question)

    # MMR: iteratively pick frame maximising λ*relevance − (1-λ)*max_sim_to_selected
    lambda_val: float = 0.5
    n_select: int = 128
    selected: List[int] = []
    remaining: List[int] = list(range(len(cand_idx)))

    while len(selected) < n_select and remaining:
        if not selected:
            best = int(np.argmax(scores[remaining]))
            best = remaining[best]
        else:
            sel_feats = feats[selected]          # (k, D)
            rem_feats = feats[remaining]          # (m, D)
            rel = scores[remaining]               # (m,)
            sim_to_sel = (rem_feats @ sel_feats.T).max(axis=1)  # (m,)
            mmr_scores = lambda_val * rel - (1.0 - lambda_val) * sim_to_sel
            best = remaining[int(np.argmax(mmr_scores))]
        selected.append(best)
        remaining.remove(best)

    selected_idx = sorted([cand_idx[i] for i in selected])
    return _get_frames_chunked(vr, selected_idx), dur


@dataclass
class MmrDiversitySpatialSkill(FallbackSkill):
    name: str = "mmr_diversity_spatial"
    frame_key: str = "mmr_diversity_spatial_frames"
    prompt_key: str = "direct"
    tags: List[str] = field(default_factory=lambda: ["mmr", "diversity", "spatial"])

    def run(self, vpath: str, question: str) -> dict:
        try:
            frames, duration = mmr_diversity_spatial_frames(vpath, question)
            return _build_skill_output(self, frames, question, duration)
        except Exception as e:
            return _fallback_to_baseline(self.name, vpath, question, e)


def _nearest_pool_slot(pool_idx: List[int], frame_idx: int) -> int:
    arr = np.asarray(pool_idx, dtype=np.int64)
    return int(np.argmin(np.abs(arr - int(frame_idx))))


def _seeded_mmr_select(
    scores: np.ndarray,
    feats: np.ndarray,
    n: int,
    lambda_: float = 0.5,
    seed_slots: Optional[List[int]] = None,
) -> List[int]:
    scores = np.asarray(scores, dtype=np.float32)
    feats = np.asarray(feats, dtype=np.float32)
    seed_slots = [] if seed_slots is None else list(seed_slots)
    selected = _unique_preserve_order([int(i) for i in seed_slots if 0 <= int(i) < len(scores)])
    remaining = [i for i in range(len(scores)) if i not in set(selected)]
    while len(selected) < n and remaining:
        if not selected:
            best = remaining[int(np.argmax(scores[np.array(remaining)]))]
        else:
            sel_feats = feats[np.array(selected)]
            rem_feats = feats[np.array(remaining)]
            rel = scores[np.array(remaining)]
            sim_to_sel = (rem_feats @ sel_feats.T).max(axis=1)
            mmr_scores = lambda_ * rel - (1.0 - lambda_) * sim_to_sel
            best = remaining[int(np.argmax(mmr_scores))]
        selected.append(best)
        remaining.remove(best)
    return selected[:n]


def _load_dino_small_state() -> Dict[str, Any]:
    global _DINO_SMALL_STATE
    if _DINO_SMALL_STATE is not None:
        return _DINO_SMALL_STATE
    import torch
    from transformers import AutoImageProcessor, AutoModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoImageProcessor.from_pretrained(
        DINO_SMALL_MODEL_NAME,
    )
    model = AutoModel.from_pretrained(
        DINO_SMALL_MODEL_NAME,
    ).to(device=device, dtype=dtype).eval()
    _DINO_SMALL_STATE = {"processor": processor, "model": model, "device": device, "dtype": dtype}
    return _DINO_SMALL_STATE


def _dino_small_feats(frames: List[Image.Image], batch_size: int = 32) -> np.ndarray:
    import torch
    state = _load_dino_small_state()
    processor, model = state["processor"], state["model"]
    device, dtype = state["device"], state["dtype"]
    feat_chunks: List[np.ndarray] = []
    for start in range(0, len(frames), batch_size):
        batch = frames[start : start + batch_size]
        inputs = processor(images=batch, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
        with torch.no_grad():
            out = model(pixel_values=pixel_values)
            feats = out.pooler_output if (hasattr(out, "pooler_output") and out.pooler_output is not None) else out.last_hidden_state[:, 0]
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        feat_chunks.append(feats.float().cpu().numpy())
    if not feat_chunks:
        return np.zeros((0, 1), dtype=np.float32)
    return np.concatenate(feat_chunks, axis=0)


def dino_episode_anchor_frames(vpath: str, question: str) -> Tuple[List[Image.Image], float]:
    vr, _, dur = _read_video(vpath)
    n_frames = len(vr)
    if n_frames <= 128:
        return uniform(vpath, 128)

    preview_idx = _pool_indices(n_frames, 64)
    preview_frames = _get_frames_chunked(vr, preview_idx)
    dino_feats = _dino_small_feats(preview_frames)

    if len(dino_feats) < 4:
        return uniform(vpath, 128)

    adj_change = 1.0 - np.sum(dino_feats[1:] * dino_feats[:-1], axis=1)
    n_clusters = min(10, len(preview_idx))
    n_boundaries = max(0, n_clusters - 1)

    if n_boundaries > 0:
        boundary_slots = (np.argsort(-adj_change)[:n_boundaries] + 1).tolist()
        boundary_slots = sorted(set(int(i) for i in boundary_slots if 0 < int(i) < len(preview_idx)))
    else:
        boundary_slots = []

    seg_starts = [0] + boundary_slots
    seg_ends = [b - 1 for b in boundary_slots] + [len(preview_idx) - 1]

    sig_scores, _ = _score_frames_and_feats(preview_frames, question)
    anchor_frame_idx: List[int] = []
    for start, end in zip(seg_starts, seg_ends):
        if end < start:
            continue
        best_local = start + int(np.argmax(sig_scores[start : end + 1]))
        anchor_frame_idx.append(preview_idx[best_local])

    cand_idx = _pool_indices(n_frames, 256)
    cand_frames = _get_frames_chunked(vr, cand_idx)
    cand_scores, cand_feats = _score_frames_and_feats(cand_frames, question)

    seed_slots = [_nearest_pool_slot(cand_idx, idx) for idx in anchor_frame_idx]
    selected_slots = _seeded_mmr_select(
        cand_scores, cand_feats, n=128, lambda_=0.5, seed_slots=seed_slots,
    )
    selected_idx = sorted(cand_idx[i] for i in selected_slots)
    return _get_frames_chunked(vr, selected_idx), dur


@dataclass
class DinoEpisodeAnchorSkill(FallbackSkill):
    name: str = "dino_episode_anchor"
    frame_key: str = "dino_episode_anchor_frames"
    prompt_key: str = "direct"
    tags: List[str] = field(default_factory=lambda: ["dino", "episode", "anchor", "mmr"])

    def run(self, vpath: str, question: str) -> dict:
        try:
            frames, duration = dino_episode_anchor_frames(vpath, question)
            return _build_skill_output(self, frames, question, duration)
        except Exception as e:
            return _fallback_to_baseline(self.name, vpath, question, e)


def dino_episode_anchor_long_frames(vpath: str, question: str) -> Tuple[List[Image.Image], float]:
    vr, _, dur = _read_video(vpath)
    if len(vr) <= 128:
        return uniform(vpath, 128)

    if dur <= 990:
        return uniform(vpath, 128)

    return dino_episode_anchor_frames(vpath, question)


def dino_context_expand_frames(vpath: str, question: str) -> Tuple[List[Image.Image], float]:
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    vr, _, dur = _read_video(vpath)
    n_frames = len(vr)

    if n_frames <= 128:
        return uniform(vpath, 128)

    if not _contains_any_keyword(question, TEMPORAL_LOC_KEYWORDS):
        if dur > 990:
            return dino_episode_anchor_long_frames(vpath, question)
        return uniform(vpath, 128)

    preview_idx    = _pool_indices(n_frames, 64)
    preview_frames = _get_frames_chunked(vr, preview_idx)
    if len(preview_frames) < 8:
        return dino_episode_anchor_long_frames(vpath, question) if dur > 990 else uniform(vpath, 128)

    dino_feats = _dino_small_feats(preview_frames)
    if len(dino_feats) < 4:
        return dino_episode_anchor_long_frames(vpath, question) if dur > 990 else uniform(vpath, 128)

    adj_change   = 1.0 - np.sum(dino_feats[1:] * dino_feats[:-1], axis=1)
    smooth_change = gaussian_filter1d(adj_change.astype(np.float32), sigma=1.0, mode="nearest")

    peaks, _ = find_peaks(
        smooth_change,
        distance=max(3, len(preview_idx) // 12),
        prominence=max(float(np.std(smooth_change)) * 0.25, 1e-4),
    )
    if len(peaks) == 0:
        boundary_slots = (np.argsort(-smooth_change)[: min(9, len(smooth_change))] + 1).tolist()
    else:
        ranked = peaks[np.argsort(-smooth_change[peaks])]
        boundary_slots = (ranked[:9] + 1).tolist()

    boundary_slots = sorted(set(int(b) for b in boundary_slots if 0 < int(b) < len(preview_idx)))
    seg_starts = [0] + boundary_slots
    seg_ends   = [b - 1 for b in boundary_slots] + [len(preview_idx) - 1]

    preview_scores, _ = _score_frames_and_feats(preview_frames, question)
    anchor_frame_idx: List[int] = []
    for start, end in zip(seg_starts, seg_ends):
        if end < start:
            continue
        local_best = start + int(np.argmax(preview_scores[start:end + 1]))
        anchor_frame_idx.append(preview_idx[local_best])

    cand_idx    = _pool_indices(n_frames, 256)
    cand_frames = _get_frames_chunked(vr, cand_idx)
    cand_scores, cand_feats = _score_frames_and_feats(cand_frames, question)

    # Expand each anchor with ±2 surrounding frames to capture event transitions.
    seed_slots: List[int] = []
    for anchor_idx in anchor_frame_idx:
        center = _nearest_pool_slot(cand_idx, anchor_idx)
        for offset in (-2, -1, 0, 1, 2):
            slot = center + offset
            if 0 <= slot < len(cand_idx):
                seed_slots.append(slot)

    seed_slots = _unique_preserve_order(seed_slots)
    if not seed_slots:
        return dino_episode_anchor_long_frames(vpath, question) if dur > 990 else uniform(vpath, 128)

    selected_slots = _seeded_mmr_select(
        cand_scores, cand_feats,
        n=min(128, len(cand_idx)),
        lambda_=0.45,
        seed_slots=seed_slots,
    )
    selected_idx = sorted(cand_idx[i] for i in selected_slots)
    return _get_frames_chunked(vr, selected_idx), dur


@dataclass
class DinoContextExpandSkill(FallbackSkill):
    name: str = "dino_context_expand"
    frame_key: str = "dino_context_expand_frames"
    prompt_key: str = "direct"
    tags: List[str] = field(default_factory=lambda: ["dino", "context", "temporal_localization", "round4"])

    def run(self, vpath: str, question: str) -> dict:
        try:
            frames, duration = dino_context_expand_frames(vpath, question)
            return _build_skill_output(
                self, frames, question, duration,
                extra_meta={"temporal_keyword": _contains_any_keyword(question, TEMPORAL_LOC_KEYWORDS)},
            )
        except Exception as e:
            return _fallback_to_baseline(self.name, vpath, question, e)


def dino_wide_context_frames(vpath: str, question: str) -> Tuple[List[Image.Image], float]:
    from scipy.ndimage import gaussian_filter1d

    vr, _, dur = _read_video(vpath)
    n_frames = len(vr)

    if n_frames <= 128:
        return uniform(vpath, 128)

    # For non-very-long videos, fall through to existing skill
    if dur <= 3600:
        return dino_context_expand_frames(vpath, question)

    pool_idx = _pool_indices(n_frames, 512)
    pool_frames = _get_frames_chunked(vr, pool_idx)
    if len(pool_frames) < 16:
        return dino_context_expand_frames(vpath, question)

    # DINO keypoints every 16th slot from the 512-pool (32 keypoints)
    key_slots = list(range(0, len(pool_idx), 16))
    key_frames = [pool_frames[i] for i in key_slots if i < len(pool_frames)]
    if len(key_frames) < 4:
        return dino_context_expand_frames(vpath, question)

    dino_feats = _dino_small_feats(key_frames)
    if len(dino_feats) < 4:
        return dino_context_expand_frames(vpath, question)

    # Smooth cosine-similarity change to detect episode boundaries
    adj_change = 1.0 - np.sum(dino_feats[1:] * dino_feats[:-1], axis=1)
    smooth_change = gaussian_filter1d(adj_change.astype(np.float32), sigma=1.0, mode="nearest")

    # Map from keypoint indices to pool_idx slots
    n_episodes = min(8, max(2, len(key_slots) // 4))
    n_boundaries = n_episodes - 1
    boundary_key_slots = sorted(
        (np.argsort(-smooth_change)[:n_boundaries] + 1).tolist()
    )
    boundary_key_slots = [i for i in boundary_key_slots if 0 < i < len(key_slots)]

    # Convert key_slot boundaries → pool_idx slots
    boundary_pool_slots = sorted(
        set(key_slots[i] for i in boundary_key_slots if i < len(key_slots))
    )

    seg_starts = [0] + boundary_pool_slots
    seg_ends = [b - 1 for b in boundary_pool_slots] + [len(pool_idx) - 1]

    # Score all pool frames with SigLIP
    pool_scores, pool_feats = _score_frames_and_feats(pool_frames, question)

    # For each episode, find best anchor frame and expand ±5 slots
    seed_slots: List[int] = []
    for start, end in zip(seg_starts, seg_ends):
        if end < start:
            continue
        local_best = start + int(np.argmax(pool_scores[start:end + 1]))
        for offset in range(-5, 6):
            slot = local_best + offset
            if start <= slot <= end:
                seed_slots.append(slot)

    seed_slots = _unique_preserve_order(seed_slots)
    if not seed_slots:
        return dino_context_expand_frames(vpath, question)

    selected_slots = _seeded_mmr_select(
        pool_scores, pool_feats,
        n=min(128, len(pool_idx)),
        lambda_=0.45,
        seed_slots=seed_slots,
    )
    selected_idx = sorted(pool_idx[i] for i in selected_slots)
    return _get_frames_chunked(vr, selected_idx), dur


@dataclass
class DinoWideContextSkill(FallbackSkill):
    name: str = "dino_wide_context"
    frame_key: str = "dino_wide_context_frames"
    prompt_key: str = "direct"
    tags: List[str] = field(default_factory=lambda: ["dino", "wide_context", "very_long", "round5"])

    def run(self, vpath: str, question: str) -> dict:
        try:
            frames, duration = dino_wide_context_frames(vpath, question)
            return _build_skill_output(
                self, frames, question, duration,
                extra_meta={"pool_size": 512, "context_window_slots": 5},
            )
        except Exception as e:
            try:
                out = SKILL_REGISTRY["dino_context_expand"].run(vpath, question)
                out.setdefault("meta", {}).update({
                    "skill": self.name,
                    "fallback_to": "dino_context_expand",
                    "fallback_reason": str(e),
                })
                return out
            except Exception:
                return _fallback_to_baseline(self.name, vpath, question, e)


def siglip_segment_query_anchor_frames(vpath: str, question: str) -> Tuple[List[Image.Image], float]:
    vr, _, dur = _read_video(vpath)
    n_frames = len(vr)

    if n_frames <= 128 or dur < 155 or dur > 990:
        return uniform(vpath, 128)

    pool_idx = _pool_indices(n_frames, 256)
    pool_frames = _get_frames_chunked(vr, pool_idx)
    if len(pool_frames) < 32:
        return uniform(vpath, 128)

    scores, feats = _score_frames_and_feats(pool_frames, question)

    # Split pool into 8 temporal blocks; pick top-4 frames per block as anchors
    block_edges = np.linspace(0, len(pool_idx), 9, dtype=int).tolist()
    anchor_slots: List[int] = []

    for b in range(8):
        start = block_edges[b]
        end = block_edges[b + 1]
        if end <= start:
            continue
        block_scores = scores[start:end]
        top_local = np.argsort(-block_scores)[: min(4, len(block_scores))].tolist()
        anchor_slots.extend(start + int(i) for i in top_local)

    anchor_slots = _unique_preserve_order(anchor_slots)
    if not anchor_slots:
        return uniform(vpath, 128)

    # Flat-relevance regime: higher diversity pressure (lambda_=0.40 vs default 0.50)
    selected_slots = _seeded_mmr_select(
        scores=scores,
        feats=feats,
        n=min(128, len(pool_idx)),
        lambda_=0.40,
        seed_slots=anchor_slots,
    )

    selected_idx = sorted(pool_idx[i] for i in selected_slots)
    return _get_frames_chunked(vr, selected_idx), dur


@dataclass
class SiglipSegmentQueryAnchorSkill(FallbackSkill):
    name: str = "siglip_segment_query_anchor"
    frame_key: str = "siglip_segment_query_anchor_frames"
    prompt_key: str = "direct"
    tags: List[str] = field(default_factory=lambda: ["siglip", "segment", "anchor", "flat_relevance", "round6"])

    def run(self, vpath: str, question: str) -> dict:
        try:
            frames, duration = siglip_segment_query_anchor_frames(vpath, question)
            return _build_skill_output(self, frames, question, duration)
        except Exception as e:
            return _fallback_to_baseline(self.name, vpath, question, e)


FRAME_STRATEGIES["mmr_diversity_spatial_frames"] = mmr_diversity_spatial_frames
SKILL_REGISTRY["mmr_diversity_spatial"] = MmrDiversitySpatialSkill()
FRAME_STRATEGIES["dino_episode_anchor_frames"] = dino_episode_anchor_frames
SKILL_REGISTRY["dino_episode_anchor"] = DinoEpisodeAnchorSkill()
FRAME_STRATEGIES["dino_context_expand_frames"] = dino_context_expand_frames
SKILL_REGISTRY["dino_context_expand"] = DinoContextExpandSkill()
FRAME_STRATEGIES["dino_wide_context_frames"] = dino_wide_context_frames
SKILL_REGISTRY["dino_wide_context"] = DinoWideContextSkill()
FRAME_STRATEGIES["siglip_segment_query_anchor_frames"] = siglip_segment_query_anchor_frames
SKILL_REGISTRY["siglip_segment_query_anchor"] = SiglipSegmentQueryAnchorSkill()

