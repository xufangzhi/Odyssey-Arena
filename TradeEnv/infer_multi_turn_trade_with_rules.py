import os
import json
import argparse
import time
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from TradeEnv_v2 import TradeArenaEnv_Deterministic

# ------------------- 配置 -------------------

parser = argparse.ArgumentParser()
parser.add_argument("--policy_dir", type=str, default="/raid/hpc/qika/symbol/models/Qwen3-4B-Instruct-2507")
parser.add_argument("--n_gpus", type=int, default=2)
parser.add_argument("--num_test_data", type=int, default=30)
parser.add_argument("--save_file", type=str, default="output/251203-2.json")
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
    max_model_len=8192*8,
    trust_remote_code=True,
    gpu_memory_utilization=0.8,
)
policy_tokenizer = AutoTokenizer.from_pretrained(policy_dir, trust_remote_code=True)
sampling_params = SamplingParams(
    max_tokens=4096*4,
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

def generate_stock_rules(data):
    """
    rule description
    """
    stocks = data["stocks"]
    variables = data["variables"]
    dependency_matrix = data["dependency_matrix"]

    rule_text_lines = []
    for stock_idx, stock_name in enumerate(stocks):
        coefficients = dependency_matrix[stock_idx]

        # 构建公式
        formula_parts = []
        for var_idx, coef in enumerate(coefficients):
            # 保留小数点后3位
            coef_rounded = round(coef, 3)
            var_name = variables[var_idx]

            if coef_rounded >= 0:
                if len(formula_parts) == 0:
                    formula_parts.append(f"{coef_rounded}Δ{var_name}")
                else:
                    formula_parts.append(f"+ {coef_rounded}Δ{var_name}")
            else:
                formula_parts.append(f"- {abs(coef_rounded)}Δ{var_name}")

        formula_str = " ".join(formula_parts)

        # 生成文字描述
        rule_text = f"The price of Stock {stock_name} is affected by：{formula_str}."

        rule_text_lines.append(rule_text)

    return "\n".join(rule_text_lines)


def generate_prompt(env, history, rules):
    """生成 LLM 的输入 prompt"""
    # desc = env.describe()
    # grid_text = env.render_text()
    # goal_hint = env.goal_hint
    # grid_text = env.return_obs()
    history_text = "\n\n".join(history[-50:])

    prompt = f"""You are an intelligent trading agent.

### Goal:
Your mission is to maximize your total portfolio value by buying and selling stocks.
The market prices are influenced by underlying variables F, and each day's news provides hints about future price changes.
You need to learn the hidden dynamics of the simulated market and make decisions accordingly.
Please note that the underlying meaning of variables may differ from the real stock.

### Rules
The stock price change is affected by the change of variables. The detailed rules are as follows:
{rules}

### Action Space:
You can take actions in the form of buying or selling multiple stocks each day.
You can combine buy and sell in one action.
The environment will first execute all sell actions, then all buy actions.
You cannot spend more cash than you have or sell stocks you don't own.

**Action Format Examples:**
- To buy 10 shares of S0 and 20 shares of S2, and sell 10 shares of S1:
<action>{{"buy": {{"S0": 10, "S2": 20}}, "sell": {{"S1": 10}}}}</action>

- To only buy:
<action>{{"buy": {{"S0": 5}}, "sell": {{}}}}</action>

- To do nothing:
<action>{{"buy": {{}}, "sell": {{}}}}</action>

**Important:** 
- Stock symbols and numbers should NOT have quotes
- Use valid JSON format inside <action></action> tags
- If you cannot afford a purchase or don't own enough shares to sell, that part of the action will be ignored

### History Actions and Feedback:
{history_text}

### Current State:
{env.get_render_obs()}

Think carefully step by step and decide your next action.
You are encouraged to act proactively, using the news to predict future price changes,
and to improve your strategy over time.

Provide your action in the format: <action>...</action>
"""
    return prompt.strip()

# ------------------- 主逻辑 -------------------
def infer():

    # 初始化结果列表
    if os.path.exists(args.save_file):
        with open(args.save_file, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = []
    origin_num = len(results)

    for env_idx in range(origin_num, args.num_test_data):
        print(f"\n===== [Env {env_idx+1}/{args.num_test_data}] =====")
        with open(f"test_data/trade/test_trade_config_{env_idx+1}.json", 'r') as file:
            config = json.load(file)

        env = TradeArenaEnv_Deterministic(cfg=config)
        # env.reset()
        history = []
        feedback = ""
        traj = {"env_id": env_idx, "config": config, "num_steps": 0, "steps": [], "token_num_total": 0, "final_state": ""}
        done = False
        token_num_total = 0
        for step in range(args.max_steps):
            user_prompt = generate_prompt(env, history, generate_stock_rules(config))
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
                if not action_str:
                    action = {}
                else:
                    action = json.loads(action_str)

            except Exception:
                print(f"[WARN] Invalid action output: {action_text}")
                traj["steps"].append(
                    {"step": step, "raw_output": action_text, "token_num": token_num_step, "action": None, "error": "invalid_action"}
                )
                action = {}

            # ---------- 环境交互 ----------
            obs, reward, done, info = env.step(action)
            history.append(env.get_render_obs()+"\nAction:"+action_str)

            traj["steps"].append(
                {
                    "step": step,
                    "action": action,
                    "raw_output": action_text,
                    "token_num": token_num_step,
                    "feedback": obs,
                }
            )

            print(f"Step {step+1}: Action={action}")
            env.render()

        traj["num_steps"] = step
        traj["token_num_total"] = token_num_total
        traj["final_state"] = env.get_render_obs()
        results.append(traj)

        # 保存
        os.makedirs(os.path.dirname(args.save_file), exist_ok=True)
        with open(args.save_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\n✅ Done! Results saved to {args.save_file}")


if __name__ == "__main__":
    infer()
