import os
import json
import argparse
import time
import re

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


from RepoEnv_v7 import ComputerEnvSetupInductionEnvV7_5

# ------------------- 配置 -------------------

parser = argparse.ArgumentParser()
parser.add_argument("--policy_dir", type=str, default="/raid/hpc/qika/symbol/models/Qwen3-4B-Instruct-2507")
parser.add_argument("--n_gpus", type=int, default=2)
parser.add_argument("--num_test_data", type=int, default=111)
parser.add_argument("--save_file", type=str, default="output/251213-4.json")
parser.add_argument("--max_steps", type=int, default=120)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
policy_dir = args.policy_dir
visible_gpus = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip() != ""]
n_gpus = min(args.n_gpus, len(visible_gpus)) if len(visible_gpus) > 0 else args.n_gpus


# ------------------- LLM 初始化 -------------------
policy_model = LLM(
    model=policy_dir,
    tensor_parallel_size=n_gpus,
    max_model_len=8192*6,
    trust_remote_code=True,
    gpu_memory_utilization=0.8,
)
policy_tokenizer = AutoTokenizer.from_pretrained(policy_dir, trust_remote_code=True)
sampling_params = SamplingParams(
    max_tokens=4096*2,
    logprobs=1,
    temperature=0.6,
    stop=["</action>", "</finish>"],
)



# ------------------- 工具函数 -------------------
def extract_action(text: str) -> str:
    """从 <action> 标签中提取动作。"""
    m = re.search(r"<action>(.*?)</action>", text, re.IGNORECASE | re.DOTALL)
    # m = re.search(r"<action>(.*?)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def generate_prompt(env, history, rules):
    """生成 LLM 的输入 prompt"""

    grid_text = env.return_obs()
    history_text = "\n\n".join(history)

    prompt = f"""
You are an intelligent computer-using agent.

# Environment Overview
You are interacting with a simulated Python project setup environment.
This environment mimics real-world difficulties of configuring a repo:
- Partial information (no full dependency graph)
- Object-level runtime failures (module/symbol/kwarg), not explicit version instructions
- Non-monotonic side-effects: installing one package may upgrade/downgrade other packages
- Hidden rules that may only trigger in specific submodules or late-stage scripts

# Repo Hierarchy & Debugging
The repo is hierarchical: it contains multiple runnable scripts under subdirectories.
You can debug incrementally by running sub-scripts (to locate which subsystem fails),
but the final goal is to make the entire project pass.

Use:
- `repo tree` (or `repo ls`) to list available scripts in the repo.
- `python <script_path>` to run a specific sub-script and "fix it step by step".
- `python run.py` to run the whole project (a sequence of entrypoints). This is the only command that ends the episode with success.

# Goal
Your ultimate goal is to make:
`python run.py`
execute successfully.

# Action Space (ONE command per step)
- Install Python:
  - `pip install python==3.10`

- Install packages:
  - `pip install pkgX`
  - `pip install pkgX==1.2`  (note: if you output x.y.z, it will be interpreted as x.y)
  - `pip install pkgX>=1.1,<2.0`

- Uninstall packages:
  - `pip uninstall pkgX`

- Inspect environment:
  - `pip list`

- Inspect repo structure:
  - `repo tree` / `repo ls`

- Execute scripts:
  - `python run.py`
  - `python core/smoke.py`  (example; use `repo tree` to discover actual paths)

Other commands (e.g., `--upgrade`) are not supported.

# How to Interpret Errors (Important)
Errors are meant as clues without directly stating version ranges:
- `ModuleNotFoundError: No module named 'pkgX'` usually means pkgX is missing.
- `ImportError: cannot import name 'S' from 'pkgX.mod'` often means pkgX version does not export that symbol.
- `TypeError: ... got an unexpected keyword argument 'kw'` indicates signature/API mismatch.
  If the message says "during project entry", adjust the provider package used by the project.
  If it says "while importing 'caller_pkg'", it indicates a caller->provider incompatibility.

Because installations can trigger side effects, a later fix may break an earlier sub-script.
Use sub-scripts to localize failures, but always re-run `python run.py` to confirm global consistency.

# Dependency Hints 
To help you finish the task, the hidden dependency among packages is listed as follows:
{rules}

# History Action and Feedback:
{history_text}

# Current Environment Feedback:
{grid_text}

Now think step by step and choose the next action.
Output exactly ONE action inside <action></action>, e.g. <action>pip install pkg0==2.1</action>.
"""
    return prompt.strip()

# ------------------- 主逻辑 -------------------
def infer():
    with open(f"test_data/repo/test_repo_lite_251215.json", 'r') as file:
        test_data = json.load(file)
    args.num_test_data = len(test_data)

    # 初始化结果列表
    if os.path.exists(args.save_file):
        with open(args.save_file, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = []
    origin_num = len(results)
    for env_idx in range(origin_num, args.num_test_data):
        print(f"\n===== [Env {env_idx+1}/{args.num_test_data}] =====")
        d = test_data[env_idx]
        env = ComputerEnvSetupInductionEnvV7_5(d)
        # env.reset()
        history = []
        feedback = ""
        traj = {"env_id": env_idx, "custom_logic": d, "initial_state": env.return_obs(), \
                    "num_steps": 120, "steps": [], "token_num_total": 0, "success": False}
        done = False
        token_num_total = 0
        token_num_step = 0
        for step in range(args.max_steps):
            user_prompt = generate_prompt(env, history, d["rules_nl_deps_only"])
            chat_inputs = policy_tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,

            )

            outputs = policy_model.generate(chat_inputs, sampling_params)
            token_num_step = len(outputs[0].outputs[0].token_ids)
            token_num_total += token_num_step
            action_text = outputs[0].outputs[0].text.strip()
            # print(action_text)
            print("-"*20)
            action_str = extract_action(action_text+"</action>")

            # ---------- 尝试解析动作 ----------
            try:
                action = action_str
            except Exception:
                print(f"[WARN] Invalid action output: {action_text}")
                traj["steps"].append(
                    {"step": step, "raw_output": action_text, "token_num": token_num_step, "action": None, "error": "invalid_action"}
                )
                continue

            if step==0:
                history.append("=== Step 1 ===\n>>> Command: " + action_str)
            else:
                history.append("Feedback:\n" + env.return_obs() + f"\n\n=== Step {step+1} ===\n>>> Command: " + action_str)
            # ---------- 环境交互 ----------
            obs, reward, done, _ = env.step(action)

            # feedback = getattr(env, "feedback", "")  # 如果 step() 设置了反馈

            traj["steps"].append(
                {
                    "step": step,
                    "action": action,
                    "raw_output": action_text,
                    "token_num": token_num_step,
                    "obs": env.return_obs(),
                    "feedback": obs,
                }
            )

            print(f"Step {step}:\n>>> Command: {action}")
            print(env.return_obs() + "\n")
            # print(obs)

            if done:
                print("✅ Mission complete!")
                traj["success"] = True
                traj["num_steps"] = step
                break

        traj["token_num_total"] = token_num_total
        results.append(traj)

        # 保存
        os.makedirs(os.path.dirname(args.save_file), exist_ok=True)
        with open(args.save_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\n✅ Done! Results saved to {args.save_file}")


if __name__ == "__main__":
    infer()
