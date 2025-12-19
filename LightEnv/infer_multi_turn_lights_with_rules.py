import os
import json
import argparse
import time
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from TextEnv_v2 import LightBulbEnv

# ------------------- 配置 -------------------

parser = argparse.ArgumentParser()
parser.add_argument("--policy_dir", type=str, default="/raid/hpc/qika/symbol/models/Qwen3-4B-Instruct-2507")
parser.add_argument("--n_gpus", type=int, default=2)
parser.add_argument("--num_test_data", type=int, default=111)
parser.add_argument("--save_file", type=str, default="output/251121-4.json")
parser.add_argument("--max_steps", type=int, default=200)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
policy_dir = args.policy_dir
visible_gpus = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip() != ""]
n_gpus = min(args.n_gpus, len(visible_gpus)) if len(visible_gpus) > 0 else args.n_gpus

# ------------------- LLM 初始化 -------------------
policy_model = LLM(
    model=policy_dir,
    tensor_parallel_size=n_gpus,
    max_model_len=8192*4,
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

def generate_prompt(env, history, feedback, rules):
    """生成 LLM 的输入 prompt"""
    # desc = env.describe()
    # grid_text = env.render_text()
    # goal_hint = env.goal_hint
    grid_text = env.return_obs()
    history_text = "\n".join(history)

    prompt = f"""
You are an intelligent agent.

### Goal:
Your mission is to light on all the bulbs.
However, the accessibility of the bulbs is based on the current condition of other bulbs.
The dependency rule to control each bulb is as follows:
{rules}

### Action Space:
The action space is based on the index of bulbs. For example, you would like to light on / off the first bulb, you should \
output <action>0</action> to toggle the state of the bulb. 

### History Action and Feedback:
{history_text}

### Current State:
{grid_text}

Now think step by step and choose the next action to act in the environment.
Output ONLY one action in the format: <action>n</action>
"""
    return prompt.strip()

# ------------------- 主逻辑 -------------------
def infer():
    with open(f"test_data/turnonlights/test_turnonlights_lite_251030.json", 'r') as file:
        test_data = json.load(file)
    args.num_test_data = len(test_data)
    results = []
    for env_idx in range(args.num_test_data):
        print(f"\n===== [Env {env_idx+1}/{args.num_test_data}] =====")
        d = test_data[env_idx]
        env = LightBulbEnv(custom_logic=d["custom_logic"], num_bulbs=d["level"])
        # env.reset()
        history = []
        feedback = ""
        traj = {"env_id": env_idx, "level": d["level"], "custom_logic": d["custom_logic"], "initial_state": env.return_obs(), \
                    "num_steps": 0, "steps": [], "token_num_total": 0, "success": False}
        done = False
        token_num_total = 0
        for step in range(args.max_steps):
            user_prompt = generate_prompt(env, history, feedback, d["custom_logic"])

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
                action = int(action_str)
                assert action in [i for i in range(env.num_bulbs)]

            except Exception:
                print(f"[WARN] Invalid action output: {action_text}")
                traj["steps"].append(
                    {"step": step, "raw_output": action_text, "token_num": token_num_step, "action": None, "error": "invalid_action"}
                )
                continue

            # ---------- 环境交互 ----------
            obs, feedback, done, _ = env.step(action)
            env_state = obs
            # feedback = getattr(env, "feedback", "")  # 如果 step() 设置了反馈
            history.append(f"Action: {action}, Feedback: {feedback}, State: {obs}")

            traj["steps"].append(
                {
                    "step": step,
                    "action": action,
                    "raw_output": action_text,
                    "token_num": token_num_step,
                    "grid": env_state,
                    "feedback": feedback,
                }
            )

            print(f"Step {step}: Action={action}")
            print(feedback)
            print(env_state)
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
