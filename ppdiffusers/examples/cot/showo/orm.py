# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

import paddle

os.environ["TOKENIZERS_PARALLELISM"] = "true"
import math

import numpy as np
from omegaconf import OmegaConf
from paddlenlp.transformers import CodeGenTokenizer
from PIL import Image
from showo.selector import ImageSelector
from tqdm import tqdm

from paddlemix.models.showo import MAGVITv2, Showo, get_mask_chedule
from paddlemix.models.showo.prompting_utils import (
    UniversalPrompting,
    create_attention_mask_predict_next,
)
from paddlemix.models.showo.sampling import cosine_schedule, mask_by_random_topk

paddle.set_grad_enabled(mode=False)
import traceback

np.random.seed(1234)
paddle.seed(1234)


def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")


class ShowoOrm(Showo):
    _supports_gradient_checkpointing = True

    def __init__(
        self,
        w_clip_vit,
        vocab_size,
        llm_vocab_size,
        llm_model_path="",
        codebook_size=8192,
        num_vq_tokens=256,
        load_from_showo=True,
        **kwargs,
    ):
        super().__init__(
            w_clip_vit=w_clip_vit,
            vocab_size=vocab_size,
            llm_vocab_size=llm_vocab_size,
            llm_model_path=llm_model_path,
            codebook_size=codebook_size,
            num_vq_tokens=num_vq_tokens,
            load_from_showo=load_from_showo,
            **kwargs,
        )

    def t2i_generate(
        self,
        input_ids: paddle.Tensor = None,
        uncond_input_ids: paddle.Tensor = None,
        attention_mask=None,
        prompts_batch=None,
        outpath=None,
        process_path=None,
        sample_path=None,
        vq_model=None,
        selector=None,
        temperature=1.0,
        timesteps=18,
        guidance_scale=0,
        noise_schedule=cosine_schedule,
        config=None,
        **kwargs,
    ):
        """
        Generate 1:1 similar to the original MaskGit repo
        https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py#L79
        """
        input_ids_copy = input_ids.clone().detach()
        uncond_input_ids_copy = uncond_input_ids.clone().detach()
        attention_mask_copy = attention_mask.clone().detach()
        for _ in range(config.eval_num):
            image_paths = []
            for iteration_count in range(config.search_num):
                input_ids = input_ids_copy.clone().detach()
                uncond_input_ids = uncond_input_ids_copy.clone().detach()
                attention_mask = attention_mask_copy.clone().detach()
                mask_token_id = self.config.mask_token_id
                num_vq_tokens = config.model.showo.num_vq_tokens
                num_new_special_tokens = config.model.showo.num_new_special_tokens
                input_ids_minus_lm_vocab_size = input_ids[:, -(num_vq_tokens + 1) : -1].clone()
                input_ids_minus_lm_vocab_size = paddle.where(
                    condition=input_ids_minus_lm_vocab_size == mask_token_id,
                    x=mask_token_id,
                    y=input_ids_minus_lm_vocab_size - config.model.showo.llm_vocab_size - num_new_special_tokens,
                )
                if uncond_input_ids is not None:
                    uncond_prefix = uncond_input_ids[:, : config.dataset.preprocessing.max_seq_length + 1]
                for step in range(timesteps):
                    if uncond_input_ids is not None and guidance_scale > 0:
                        uncond_input_ids = paddle.concat(
                            x=[
                                uncond_prefix,
                                input_ids[:, config.dataset.preprocessing.max_seq_length + 1 :],
                            ],
                            axis=1,
                        )
                        model_input = paddle.concat(x=[input_ids, uncond_input_ids])
                        cond_logits, uncond_logits = self(model_input, attention_mask=attention_mask).chunk(chunks=2)
                        logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                        logits = logits[
                            :,
                            -(num_vq_tokens + 1) : -1,
                            config.model.showo.llm_vocab_size + num_new_special_tokens : -1,
                        ]
                    else:
                        logits = self(input_ids, attention_mask=attention_mask)
                        logits = logits[
                            :,
                            -(num_vq_tokens + 1) : -1,
                            config.model.showo.llm_vocab_size + num_new_special_tokens : -1,
                        ]
                    probs = paddle.nn.functional.softmax(logits, axis=-1)
                    sampled = probs.reshape(-1, logits.shape[-1])
                    sampled_ids = paddle.multinomial(x=sampled, num_samples=1)[:, 0].view(*tuple(logits.shape)[:-1])
                    unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
                    sampled_ids = paddle.where(
                        condition=unknown_map,
                        x=sampled_ids,
                        y=input_ids_minus_lm_vocab_size,
                    )
                    ratio = 1.0 * (step + 1) / timesteps
                    mask_ratio = noise_schedule(paddle.to_tensor(data=ratio))
                    selected_probs = paddle.take_along_axis(
                        arr=probs,
                        axis=-1,
                        indices=sampled_ids.astype(dtype="int64")[..., None],
                        broadcast=False,
                    )
                    selected_probs = selected_probs.squeeze(axis=-1)
                    selected_probs = paddle.where(
                        condition=unknown_map,
                        x=selected_probs,
                        y=paddle.finfo(dtype=selected_probs.dtype).max,
                    )
                    mask_len = (
                        paddle.to_tensor([(num_vq_tokens * mask_ratio).floor()]).unsqueeze(0).astype(dtype="int64")
                    )

                    mask_len = paddle.maximum(
                        paddle.to_tensor([1]), paddle.minimum(unknown_map.sum(axis=-1, keepdim=True) - 1, mask_len)
                    )

                    temperature = temperature * (1.0 - ratio)
                    masking = mask_by_random_topk(mask_len, selected_probs, temperature)
                    input_ids[:, -(num_vq_tokens + 1) : -1] = paddle.where(
                        condition=masking,
                        x=mask_token_id,
                        y=sampled_ids + config.model.showo.llm_vocab_size + num_new_special_tokens,
                    )
                    input_ids_minus_lm_vocab_size = paddle.where(condition=masking, x=mask_token_id, y=sampled_ids)
                    sampled_ids_copy = sampled_ids.clone()
                    sampled_ids_copy = paddle.clip(
                        x=sampled_ids_copy,
                        max=config.model.showo.codebook_size - 1,
                        min=0,
                    )
                    images = vq_model.decode_code(sampled_ids_copy)
                    images = paddle.clip(x=(images + 1.0) / 2.0, min=0.0, max=1.0)
                    images *= 255.0
                    images = images.transpose(perm=[0, 2, 3, 1]).cpu().numpy().astype(np.uint8)
                    image_filename = f"{iteration_count:05d}.png"
                    image_path = os.path.join(process_path, image_filename)
                    samples = [Image.fromarray(image) for image in images]
                    if step == timesteps - 1:
                        image_paths.append(image_path)
                        samples[0].save(image_path)

            highest_score_yes = float("-inf")
            for i, image_path in enumerate(image_paths):
                selected, score = selector.orm(prompts_batch, image_path)
                if score > highest_score_yes:
                    highest_score_yes = score
                    best_yes = i
            os.system("cp " + image_paths[best_yes] + " " + os.path.join(sample_path, f"{_:04d}.png"))
        return None


def main(opt):
    with open(opt.prompts_file) as fp:
        prompts = [line.strip() for line in fp if line.strip()]
    metadata_list = []
    try:
        with open(opt.metadata_file, "r", encoding="utf-8") as metadata_file:
            metadata_list = json.load(metadata_file)
    except json.JSONDecodeError:
        with open(opt.metadata_file, "r", encoding="utf-8") as metadata_file:
            for line in metadata_file:
                metadata_list.append(json.loads(line))
    paddle.distributed.init_parallel_env()
    rank = paddle.distributed.get_rank()

    if opt.model == "show-o":
        cli_conf = OmegaConf.create(
            {
                "batch_size": opt.batch_size,
                "validation_prompts_file": opt.validation_prompts_file,
                "guidance_scale": opt.guidance_scale,
                "generation_timesteps": opt.generation_timesteps,
                "mode": "t2i",
                "eval_num": opt.eval_num,
                "search_num": opt.search_num,
                "config": opt.config,
            }
        )
        yaml_conf = OmegaConf.load(cli_conf.config)
        config = OmegaConf.merge(yaml_conf, cli_conf)
        tokenizer = CodeGenTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side="left")
        uni_prompting = UniversalPrompting(
            tokenizer,
            max_text_len=config.dataset.preprocessing.max_seq_length,
            special_tokens=(
                "<|soi|>",
                "<|eoi|>",
                "<|sov|>",
                "<|eov|>",
                "<|t2i|>",
                "<|mmu|>",
                "<|t2v|>",
                "<|v2v|>",
                "<|lvg|>",
            ),
            ignore_id=-100,
            cond_dropout_prob=config.training.cond_dropout_prob,
        )
        model = ShowoOrm.from_pretrained(opt.dpo_model_path, dtype=opt.dtype)
        model.eval()
        mask_token_id = model.config.mask_token_id
        if config.get("validation_prompts_file", None) is not None:
            config.dataset.params.validation_prompts_file = config.validation_prompts_file
        config.training.batch_size = config.batch_size
        config.training.guidance_scale = config.guidance_scale
        config.training.generation_timesteps = config.generation_timesteps
    else:
        raise ValueError("model is not supported")
    vq_model = get_vq_model_class(config.model.vq_model.type)
    vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name)

    vq_model.eval()
    selector = ImageSelector(pretrained=opt.reward_model_path)

    global_n_samples = paddle.distributed.get_world_size()
    total_prompts = int(math.ceil(len(prompts) / global_n_samples) * global_n_samples)
    new_prompts = prompts + [prompts[0]] * (total_prompts - len(prompts))
    per_gpu_prompts = new_prompts[rank:total_prompts:global_n_samples]
    os.makedirs(opt.outdir, exist_ok=True)
    for index, prompt in tqdm(enumerate(per_gpu_prompts), total=len(per_gpu_prompts), desc=f"Rank {rank}"):
        global_index = index * global_n_samples + rank
        if global_index > len(prompts) - 1:
            break
        outpath = os.path.join(opt.outdir, f"{global_index:0>5}")
        os.makedirs(outpath, exist_ok=True)
        process_path = os.path.join(outpath, "process")
        os.makedirs(process_path, exist_ok=True)
        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        if global_index < len(metadata_list):
            metadata = metadata_list[global_index]
            if metadata["prompt"] != prompt:
                raise ValueError(
                    f"Mismatch detected at index {global_index}: Metadata prompt '{metadata['prompt']}' does not match the current prompt '{prompt}'. Aborting process."
                )
            with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
                json.dump(metadata, fp, indent=4)
        prompts_batch = [prompt]
        image_tokens = (
            paddle.ones(
                shape=(len(prompts_batch), config.model.showo.num_vq_tokens),
                dtype="int64",
            )
            * mask_token_id
        )
        input_ids, _ = uni_prompting((prompts_batch, image_tokens), "t2i_gen")
        if config.training.guidance_scale > 0:
            uncond_input_ids, _ = uni_prompting(([""], image_tokens), "t2i_gen")
            attention_mask = create_attention_mask_predict_next(
                paddle.concat(x=[input_ids, uncond_input_ids], axis=0),
                pad_id=int(uni_prompting.sptids_dict["<|pad|>"]),
                soi_id=int(uni_prompting.sptids_dict["<|soi|>"]),
                eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"]),
                rm_pad_in_image=True,
            )
        else:
            attention_mask = create_attention_mask_predict_next(
                input_ids,
                pad_id=int(uni_prompting.sptids_dict["<|pad|>"]),
                soi_id=int(uni_prompting.sptids_dict["<|soi|>"]),
                eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"]),
                rm_pad_in_image=True,
            )
            uncond_input_ids = None
        if config.get("mask_schedule", None) is not None:
            schedule = config.mask_schedule.schedule
            args = config.mask_schedule.get("params", {})
            mask_schedule = get_mask_chedule(schedule, **args)
        else:
            mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))
        model.t2i_generate(
            input_ids=input_ids.clone(),
            uncond_input_ids=uncond_input_ids.clone(),
            attention_mask=attention_mask.clone(),
            prompts_batch=prompts_batch,
            outpath=outpath,
            process_path=process_path,
            sample_path=sample_path,
            vq_model=vq_model,
            selector=selector,
            guidance_scale=config.training.guidance_scale,
            temperature=config.training.get("generation_temperature", 1.0),
            timesteps=config.training.generation_timesteps,
            noise_schedule=mask_schedule,
            noise_type=config.training.get("noise_type", "mask"),
            seq_len=config.model.showo.num_vq_tokens,
            uni_prompting=uni_prompting,
            config=config,
        )

    if rank == 0:
        print("Done.")


def run_orm(opt):
    try:
        main(opt)
    except Exception as e:
        rank = int(os.environ.get("RANK", -1))
        print(f"Error in rank {rank}: {e}")
        traceback.print_exc()
