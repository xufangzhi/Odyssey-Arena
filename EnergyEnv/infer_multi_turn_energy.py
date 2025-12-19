import os
import json
import argparse
import time
import re
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from EnergyEnv_v5 import DynamicEnergyGrid

# ------------------- 配置 -------------------

parser = argparse.ArgumentParser()
parser.add_argument("--policy_dir", type=str, default="/raid/hpc/qika/symbol/models/Qwen3-4B-Instruct-2507")
parser.add_argument("--n_gpus", type=int, default=2)
parser.add_argument("--num_test_data", type=int, default=111)
parser.add_argument("--save_file", type=str, default="output/251211-10.json")
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
    gpu_memory_utilization=0.85,
)
policy_tokenizer = AutoTokenizer.from_pretrained(policy_dir, trust_remote_code=True)
sampling_params = SamplingParams(
    max_tokens=4096*5,
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


def generate_prompt(env, history, target_stability, target_carbon):
    """生成 LLM 的输入 prompt"""
    # desc = env.describe()
    # grid_text = env.render_text()
    # goal_hint = env.goal_hint
    grid_text = env.return_obs()
    history_text = "\n".join(history[-40:])

    prompt = f"""
You are an intelligent energy system operator managing a Dynamic Energy Grid.
Your goal is to achieve a safe, stable, and low-carbon electricity supply across a long planning horizon.
Each day, you adjust the composition of generation resources within strict physical and economic limits.
To perform well, you must learn and exploit hidden temporal patterns from the history.

# ENVIRONMENT OVERVIEW

This environment simulates a long-horizon national power grid with four generation types:
Thermal — highly reliable, carbon-intensive, lowest cost.
Wind — highly variable, driven by seasonal cycles.
Solar — variable, driven by seasonal cycles.
Battery (Storage) — A storage buffer that can charge or discharge based on the capacity. Its carbon footprint is determined by the source of energy used for charging.

Each day t, the system evolves according to underlying temporal dynamics.
The agent must design the next day's rated generation scheme while anticipating these dynamics.

## Demand & Budget

The allocation scheme must strictly satisfy both demand and budget constraints.
current_demand (MW) — electricity required today.
current_budget — tomorrow's maximum allowable total generation cost.

## Generation Cost Model (Unit Prices)

Each generation type has a fixed unit cost per MW of rated output:
Thermal: cheapest (e.g., 3.0 cost/unit)
Wind: moderate cost (e.g., 5.0 cost/unit)
Solar: more expensive (e.g., 6.0 cost/unit)
Battery: operational cost (Charge/Discharge), very low (e.g., 0.1 cost/unit)

## Grid Stability

To maintain a stable grid, the agent must avoid large day-to-day changes in the rated outputs.
Sudden increases or decreases (ramping) reduce stability, which affects overall performance.
A good strategy adjusts gradually, anticipating future needs rather than reacting abruptly.
violating the daily budget or failing to meet the demand would largely damage system stability.

## Carbon Intensity

Thermal generation emits carbon.
To maintain a clean and sustainable city, the agent should limit the proportion of thermal power while still meeting demand and respecting budget constraints.
This creates a non-trivial trade-off between cost, stability, and carbon impact.

## Season & Efficiency

Actual generation is not equal to rated generation.
It depends on a time-varying efficiency term:

actual_output = rated_output × efficiency(t)

Efficiency changes periodically over time. Solar and Wind share different periods. Agent is required to derive the hidden temporal rules from the history observation.
Because actual output fluctuates around rated output, the agent must leave safety margins and learn the temporal structure.

# Objective
The agent needs to simulate across a long planning horizon (120 Turns).
The task is successful only if the final metric **Stability > {target_stability:.3f}, Carbon < {target_carbon:.3f} **.
Notably, violation of daily cost or demand constraints for 3 consecutive steps would lead to termination.

# Action Space
Each day, the agent must decide the rated generation for the next day within the capacity limit:
thermal (MW), Rated Power Command, [0,600], Must be non-negative.
wind (MW), Rated Power Command, [0,350], Must be non-negative.
solar (MW), Rated Power Command, [0,250], Must be non-negative.
battery (MW), Net Flow Command, battery capacity=80, Bidirectional: Negative = Charge (Consumption), Positive = Discharge (Supply).

**Action Format Example 1**:
<action>{{"thermal": 400.0, "wind": 10.0, "solar": 35.0, "battery": -15.0}}</action>
Interpretation: The agent sets the Rated Power for Thermal, Wind, and Solar to 400 MW, 10 MW, and 35 MW, respectively. Additionally, the agent commands the battery to consume 15 MW from the grid for charging. This 15 MW consumption will be drawn from the total supply available from the three generation units.

**Action Format Example 2**:
<action>{{"thermal": 350.0, "wind": 25.0, "solar": 15.0, "battery": 10.0}}</action>
Interpretation: The agent sets the Rated Power for Thermal, Wind, and Solar to 350 MW, 25 MW, and 15 MW, respectively. Additionally, the agent commands the battery to supply 10 MW of power to the grid (discharging). This 10 MW is added to the total supply from the three generation units.

# History Action and Feedback:
{history_text}

# Current State:
{grid_text}

**Important Note:** 
- Set Rated Capacity above Actual Demand to save room for the efficiency gap (Rated vs. Actual output) and forecast uncertainty.
- Keep daily cost within the budget and meet the daily demand, violation of either cost and supply for three consecutive steps would lead to immediate, irreversible grid collapse.
- Stability and Carbon are long-term average metric. After 120-turn, stability must be > {target_stability:.3f}, Carbon must be < {target_carbon:.3f}.

Now think step by step and choose the next action to act in the environment.
You are encouraged to act actively to derive the environment dynamics.
Output the action within the tag of <action></action>.
"""
    return prompt.strip()

# ------------------- 主逻辑 -------------------
def infer():
    with open(f"test_data/energy/test_energy_lite_251207.json", 'r') as file:
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
        env = DynamicEnergyGrid(config=d)
        # env.reset()
        history = []
        feedback = ""
        traj = {"env_id": env_idx, "custom_logic": d, "initial_state": env.return_obs(), \
                    "num_steps": 0, "steps": [], "token_num_total": 0, "success": False}
        done = False
        token_num_total = 0
        token_num_step = 0
        for step in range(args.max_steps):
            user_prompt = generate_prompt(env, history, d["target_stability"], d["target_carbon"])

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
                if action_str:
                    action = json.loads(action_str)

            except Exception:
                print(f"[WARN] Invalid action output: {action_text}")
                traj["steps"].append(
                    {"step": step, "raw_output": action_text, "token_num": token_num_step, "action": None, "error": "invalid_action"}
                )
                continue

            history.append(env.return_obs() + "\nAction:" + action_str)
            # ---------- 环境交互 ----------
            obs, reward, done, _ = env.step(action)

            # feedback = getattr(env, "feedback", "")  # 如果 step() 设置了反馈

            traj["steps"].append(
                {
                    "step": step,
                    "action": action,
                    "raw_output": action_text,
                    "token_num": token_num_step,
                    "stability": env.stability_avg,
                    "carbon": env.share_thermal,
                    "obs": env.return_obs(),
                    "feedback": obs,
                    "reward": reward
                }
            )

            print(f"Step {step}: Action={action}")
            print(env.return_obs())
            # print(obs)

            # if env.budget_violation or env.demand_violation or env.carbon_violation:
            #     print("❌ Mission failed!")
            #     traj["success"] = False
            #     traj["num_steps"] = step
            #     break

            if done and reward:
                print("✅ Mission complete!")
                traj["success"] = True
                traj["num_steps"] = step
                break
            elif done:
                print("❌ Mission failed!")
                traj["success"] = False
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
