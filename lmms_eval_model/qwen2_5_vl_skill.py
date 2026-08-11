"""
Qwen2.5-VL adapter with pluggable skill-based frame selection.

Usage (lmms_eval):
    --model qwen2_5_vl_skill
    --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,skill_name=aks_temporal_cover_128_direct,...

The adapter intercepts video path inputs, runs the configured skill to select
frames, then passes the selected frames as PIL images through the standard
lmms_eval pipeline. All other behaviour (processor, generate, decode, scoring)
is identical to the parent Qwen2_5_VL class.

Frames selected by the skill are passed as {"type": "video", "video": List[PIL.Image]}
which qwen_vl_utils.fetch_video supports natively. This uses video temporal encoding,
identical to the official baseline, giving comparable absolute scores.
"""
from __future__ import annotations

import base64
import io
import sys
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL

process_vision_info, _has_qwen_vl = optional_import("qwen_vl_utils", "process_vision_info")

# Path where skill_learning package lives
import os as _os
_PROJECT_DIR = _os.environ.get(
    "AUTO_SKILL_LMMS_DIR",
    "/Youtu_VITA/jiannhu/Auto-claude-video-skill-generation/lmms-eval",
)


@register_model("qwen2_5_vl_skill")
class Qwen2_5_VL_Skill(Qwen2_5_VL):
    """
    Qwen2.5-VL with pluggable video frame selection skills.

    Inherits all behaviour from Qwen2_5_VL but overrides generate_until to
    replace uniform frame sampling with skill-based frame selection.
    Skill-selected PIL frames are passed as {"type":"video","video":List[PIL.Image]}
    for video temporal encoding (same as official baseline).
    """

    def __init__(
        self,
        skill_name: str = "uniform_128_direct",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.skill_name = skill_name
        self._skill_registry = None
        eval_logger.info(f"[qwen2_5_vl_skill] skill_name={skill_name}")

    # ── Skill access ────────────────────────────────────────────────────────

    def _get_skill_registry(self):
        if self._skill_registry is None:
            if _PROJECT_DIR not in sys.path:
                sys.path.insert(0, _PROJECT_DIR)
            from skill_learning.skills import SKILL_REGISTRY  # noqa: PLC0415
            self._skill_registry = SKILL_REGISTRY
        return self._skill_registry

    def _skill_frames(self, vpath: str, question: str) -> Optional[List[Image.Image]]:
        """
        Run the configured skill on *vpath* and return selected PIL frames.
        Returns None if the skill fails (caller falls back to video-path mode).

        Priority:
          1. out["frames"] — direct PIL list, no encode/decode roundtrip (preferred)
          2. out["messages"] — decode base64 images for backward compat
        """
        try:
            registry = self._get_skill_registry()
            skill = registry[self.skill_name]
            out = skill.run(vpath, question)

            # 1. Direct PIL frames (fastest path — skills always populate this)
            raw_frames = out.get("frames")
            if raw_frames:
                imgs = [
                    f.convert("RGB") if hasattr(f, "convert") else f
                    for f in raw_frames
                ]
                if imgs:
                    return imgs

            # 2. Fallback: decode base64 images from messages (legacy skills)
            imgs: List[Image.Image] = []
            for msg in out.get("messages", []):
                for part in msg.get("content") or []:
                    if isinstance(part, dict) and part.get("type") == "image":
                        img = part["image"]
                        if isinstance(img, Image.Image):
                            imgs.append(img.convert("RGB"))
                        elif isinstance(img, str) and img.startswith("data:image"):
                            _, b64data = img.split(",", 1)
                            imgs.append(
                                Image.open(io.BytesIO(base64.b64decode(b64data))).convert("RGB")
                            )
            if imgs:
                return imgs
        except Exception as exc:
            eval_logger.warning(
                f"[qwen2_5_vl_skill] skill={self.skill_name} failed on {vpath}: {exc}"
            )
        return None

    # ── Override generate_until ──────────────────────────────────────────────

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visual_list = [
                doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id
            ]
            gen_kwargs = all_gen_kwargs[0]

            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected `gen_kwargs['until']` to be Union[str, list], got {type(until)}")
            until = [item for item in until if item != "\n\n"]

            contexts = list(contexts)
            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")

            batched_messages = []
            for i, context in enumerate(contexts):
                if "<image>" in context:
                    context = context.replace("<image>", "")

                message = [{"role": "system", "content": self.system_prompt}]
                if self.reasoning_prompt:
                    context = context.strip() + self.reasoning_prompt
                    contexts[i] = context

                processed_visuals = []
                if visual_list[i] is not None:
                    for visual in visual_list[i]:
                        if isinstance(visual, str) and visual.endswith(
                            (".mp4", ".avi", ".mov", ".MP4", ".mkv", ".webm")
                        ):
                            # ── Skill frame selection ──────────────────────
                            frames = self._skill_frames(visual, context)
                            if frames:
                                # Pass as video type with PIL list → video temporal encoding
                                # qwen_vl_utils.fetch_video supports List[PIL.Image] as video
                                processed_visuals.append(
                                    {
                                        "type": "video",
                                        "video": frames,
                                        "max_pixels": self.max_pixels,
                                        "min_pixels": self.min_pixels,
                                    }
                                )
                            else:
                                # Fallback: original video-path uniform sampling
                                processed_visuals.append(
                                    {
                                        "type": "video",
                                        "video": visual,
                                        "max_pixels": self.max_pixels,
                                        "min_pixels": self.min_pixels,
                                    }
                                )
                        elif isinstance(visual, Image.Image):
                            processed_visuals.append(
                                {
                                    "type": "image",
                                    "image": self._encode_image_data_url(visual),
                                    "max_pixels": self.max_pixels,
                                    "min_pixels": self.min_pixels,
                                }
                            )

                # All visuals before text (interleave_visuals not supported here)
                message.append(
                    {
                        "role": "user",
                        "content": processed_visuals + [{"type": "text", "text": context}],
                    }
                )
                batched_messages.append(message)

            texts = self.processor.apply_chat_template(
                batched_messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(batched_messages)

            # Skill selects frames before passing as {"type":"video","video":List[PIL.Image]}.
            # No resampling needed — skill already chose the frames.
            # For fallback (raw video path), apply max_num_frames subsampling.
            if video_inputs is not None and image_inputs is None:
                # Resample only for raw video-path fallback (torch.Tensor); skip for
                # skill-selected PIL lists (already the right number of frames).
                for vi, vid in enumerate(video_inputs):
                    if isinstance(vid, (list, tuple)):
                        total_frames = len(vid)
                    else:
                        total_frames = vid.shape[0]
                    if total_frames > self.max_num_frames:
                        indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
                        indices = np.unique(indices)
                        if isinstance(vid, (list, tuple)):
                            video_inputs[vi] = [vid[i] for i in indices]
                        else:
                            video_inputs[vi] = vid[indices]

            padding_side = "left" if self.batch_size > 1 else "right"
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                padding_side=padding_side,
                return_tensors="pt",
            )
            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            default_gen_kwargs = {
                "max_new_tokens": 32768,
                "temperature": 0.0,
                "top_p": None,
                "num_beams": 1,
            }
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None

            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, cont)
            ]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res
