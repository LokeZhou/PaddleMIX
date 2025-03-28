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

import argparse

import paddle

paddle.set_grad_enabled(mode=False)
from showo.orm import run_orm
from showo.parm import run_parm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="prompts.txt",
        help="Text file containing prompts, one per line",
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default="metadata.jsonl",
        help="Metadata for geneval",
    )
    parser.add_argument("--model", type=str, default="show-o", help="Huggingface model name")
    parser.add_argument("--outdir", type=str, help="dir to write results to", default="geneval/outputs")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="how many samples can be produced simultaneously",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="showo_prm.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
    )

    parser.add_argument(
        "--validation_prompts_file",
        type=str,
        default=None,
        help="Path to the validation prompts file",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.75,
        help="Guidance scale for generation",
    )
    parser.add_argument(
        "--generation_timesteps",
        type=int,
        default=18,
        help="Number of timesteps for generation",
    )
    parser.add_argument("--eval_num", type=int, default=4, help="for geneval benchmark")
    parser.add_argument("--search_num", type=int, default=20, help="search number")
    parser.add_argument("--reward_model", type=str, default="", help="Mode of reward model")
    parser.add_argument("--dpo_model", type=str, default="")
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_args()
    outputname_list = [i for i in [opt.model, opt.reward_model, opt.dpo_model] if i]
    opt.outdir = "results/" + "_".join(outputname_list)
    print(f"Result will be saved under: {opt.outdir}")
    if opt.dpo_model == "dpo":
        opt.dpo_model_path = "ckpts/dpo"
        print("Running Initial DPO...")
    elif opt.dpo_model == "dpo_iter":
        opt.dpo_model_path = "ckpts/dpo_iter"
        print("Running Iterative DPO...")
    elif opt.dpo_model == "dpo_iter_parm_gudie":
        opt.dpo_model_path = "ckpts/dpo_iter_parm_gudie"
        print("Running Iterative DPO with PARM Guidance...")
    elif opt.dpo_model == "":
        opt.dpo_model_path = "showlab/show-o-512x512"
        print("Running without DPO...")
    else:
        raise ValueError(f"DPO model: {opt.dpo_model} is not supported yet...")
    if opt.reward_model == "orm_zs":
        opt.reward_model_path = "lmms-lab/llava-onevision-qwen2-7b-ov"
        print("Running Zero-shot ORM...")
        run_orm(opt)
    elif opt.reward_model == "orm_ft":
        opt.reward_model_path = "ckpts/orm_ft"
        print("Running Fine-tuned ORM...")
        run_orm(opt)
    elif opt.reward_model == "parm":
        opt.reward_model_path = "ckpts/parm"
        print("Running PARM...")
        run_parm(opt)
    # elif opt.reward_model == "":
    #     opt.reward_model_path = ""
    #     print("Running without Reward Model...")
    #     run_showo(opt)
    else:
        raise ValueError(f"Reward model: {opt.reward_model} is not supported yet...")
