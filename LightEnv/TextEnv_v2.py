import json
import random
import re
import json
import collections
import numpy as np
from collections import deque

class LightBulbEnv:
    def __init__(self, num_bulbs=5, seed=None, max_attempts=100, min_steps=5, expose_logic=False, custom_logic=None):
        self.num_bulbs = num_bulbs
        self.rng = random.Random(seed)
        self.seed_value = seed
        self.max_attempts = max_attempts
        self.min_steps = min_steps
        self.expose_logic = expose_logic
        self.custom_logic = custom_logic  # 👈 新增参数
        self.reset()

    # ---------------------------
    def reset(self):
        self.bulbs = {f"B{i}": False for i in range(self.num_bulbs)}
        self.steps = 0

        if self.custom_logic is not None:
            # 👇 如果用户手动传入逻辑，直接使用
            self.logic_expr = self.custom_logic
        else:
            # 否则执行随机生成逻辑
            for attempt in range(self.max_attempts):
                self._generate_dependencies()
                if self._validate_min_steps():
                    break
                else:
                    self.rng.seed(self.seed_value + attempt + 1)

        return self._get_obs()

    # ---------------------------
    def _generate_dependencies(self):
        """
        随机生成每个灯泡的依赖逻辑（布尔表达式）
        保证形成 DAG（无环），且初始全 False 状态下不可直接点亮
        """
        self.logic_expr = {}
        bulbs = list(self.bulbs.keys())
        n = len(bulbs)

        for i in range(n):
            # 可以依赖前面的灯泡，保证 DAG
            if i == 0:
                # 第一个灯泡无依赖，直接可切换
                self.logic_expr[bulbs[i]] = "True"
                continue

            dep_count = self.rng.randint(1, min(8, i))
            dep_indices = self.rng.sample(range(i), dep_count)
            terms = []
            for idx in dep_indices:
                name = bulbs[idx]
                if self.rng.random() < 0.5:
                    terms.append(f"not {name}")
                else:
                    terms.append(name)
            # 随机用 and/or 组合
            expr = terms[0]
            for term in terms[1:]:
                op = self.rng.choice(["and", "or"])
                expr = f"({expr} {op} {term})"
            self.logic_expr[bulbs[i]] = expr
        self._shuffle_bulbs()

    def _shuffle_bulbs(self):
        """
        随机打乱灯泡顺序，但保持逻辑结构正确（使用占位符两步替换以避免替换冲突）
        """
        bulbs = list(self.logic_expr.keys())  # 原名字列表，保证包含所有变量
        shuffled = bulbs.copy()
        self.rng.shuffle(shuffled)

        # rename_map: old_name -> new_name
        rename_map = {old: new for old, new in zip(bulbs, shuffled)}

        # 第一步：把每个原名替换为唯一占位符，避免任何冲突或部分匹配
        placeholder_map = {}
        for i, old in enumerate(bulbs):
            placeholder = f"__VAR_{i}__"
            placeholder_map[old] = placeholder

        # 用占位符替换原表达式中的变量（使用单词边界）
        intermediate_logic = {}
        for old, expr in self.logic_expr.items():
            new_expr = expr
            for old_name, placeholder in placeholder_map.items():
                # 使用 \b 确保只匹配完整变量名
                new_expr = re.sub(rf"\b{re.escape(old_name)}\b", placeholder, new_expr)
            intermediate_logic[old] = new_expr

        # 第二步：把占位符替换为目标新名字
        final_logic = {}
        for old, inter_expr in intermediate_logic.items():
            final_expr = inter_expr
            # 找到原 old 对应的新名字 target_name
            target_name = rename_map[old]
            # 将 intermediate 表达式中的每个占位符替换成对应的 rename_map 值
            for orig_name, placeholder in placeholder_map.items():
                final_name = rename_map[orig_name]
                final_expr = final_expr.replace(placeholder, final_name)
            # 最终把表达式放到新的键（即 target_name）下
            final_logic[target_name] = final_expr

        # 更新 self.logic_expr 和 self.bulbs（保持全 False 初始值或现有值映射）
        # 这里保持原来 bulbs 的布尔值映射（如果需要保留现有状态）
        old_bulb_states = self.bulbs.copy()
        # 先置空再填充，确保键与 final_logic 一致
        self.logic_expr = final_logic
        self.bulbs = {new_name: old_bulb_states[old_name] for old_name, new_name in rename_map.items()}

    # ---------------------------
    def _validate_min_steps(self):
        """
        验证从全 False 状态存在操作序列能点亮所有灯泡，
        且最少操作步数 >= self.min_steps
        """
        bulbs_list = list(self.bulbs.keys())
        visited = set()

        def dfs(state, path_len):
            key = tuple(state.values())
            if key in visited:
                return None
            visited.add(key)

            if all(state.values()):
                return path_len
            min_len = None
            for bulb in bulbs_list:
                # 模拟 toggle
                can_toggle = self._eval_logic(bulb, state)
                new_state = state.copy()
                if can_toggle:
                    new_state[bulb] = not new_state[bulb]
                    result = dfs(new_state, path_len + 1)
                    if result is not None:
                        if min_len is None or result < min_len:
                            min_len = result
            return min_len

        min_path = dfs({k: False for k in bulbs_list}, 0)
        if min_path is None:
            return False
        return min_path >= self.min_steps

    # ---------------------------
    def _eval_logic(self, bulb, state=None):
        """
        计算某个灯泡依赖逻辑是否满足
        """
        if state is None:
            state = self.bulbs
        expr = self.logic_expr[bulb]
        local_vars = state.copy()
        try:
            return bool(eval(expr, {"__builtins__": {}}, local_vars))
        except Exception:
            return False

    # ---------------------------
    def step(self, action):
        """
        action: int in [0, num_bulbs-1], 对应灯泡索引
        """
        bulb_name = f"B{action}"
        self.steps += 1

        if self._eval_logic(bulb_name):
            # toggle 成功
            self.bulbs[bulb_name] = not self.bulbs[bulb_name]
            hint = f"Toggled {bulb_name} to {self.bulbs[bulb_name]}"
        else:
            hint = f"{bulb_name} remains inactive... remaining bulbs should be in specific mode."

        done = all(self.bulbs.values())
        return self._get_obs(), hint, done, {}

    # ---------------------------
    def _get_obs(self):
        """
        返回一维灯泡状态列表
        """
        return [self.bulbs[f"B{i}"] for i in range(self.num_bulbs)]

    # ---------------------------
    def render(self):
        state = ["💡" if self.bulbs[f"B{i}"] else "○" for i in range(self.num_bulbs)]
        print(" ".join(state))

        if self.expose_logic:
            print("Logic expressions (Only hint to human test, not exposed to agent test):")
            for k, v in self.logic_expr.items():
                print(f"{k}: {v}")
            print()

    def return_obs(self):
        state = ["💡" if self.bulbs[f"B{i}"] else "○" for i in range(self.num_bulbs)]
        return " ".join(state)


# ---------------------------
# 简单使用示例
if __name__ == "__main__":
    # 手动指定逻辑
    custom_logic = {
        "B0": "True",
        "B1": "B0",
        "B2": "B1 and not B0",
        "B3": "B2 or B1",
        "B4": "not B3",
        "B5": "B4 and B2",
        "B6": "B5 or not B1",
        "B7": "B6 and B4",
    }
    count = collections.defaultdict(int)
    for i in range(7, 8):
        for j in range(50):
            num_bulbs = i
            seed = random.randint(0, 9999)
            env = LightBulbEnv(num_bulbs=num_bulbs, custom_logic=None, seed=seed, min_steps=10, expose_logic=False)
            obs = env.reset()
            env.render()
            for k, v in env.logic_expr.items():
                print(f"{k}: {v}")
            if input("Your choice is: ")=="1":
                try:
                    with open(f"test_data/turnonlights/test_turnonlights_251029.json", 'r') as file:
                        test_data = json.load(file)
                except:
                    test_data = []
                count[num_bulbs] += 1
                data_dict = {}
                data_dict['level'] = num_bulbs
                data_dict['custom_logic'] = env.logic_expr
                test_data.append(data_dict)
                with open(f"test_data/turnonlights/test_turnonlights_251029.json", 'a') as file:
                    json.dump(test_data, file, indent=4)
            print("-"*20)
            print(count)
            print("-"*20)




    # done = False
    # idx = 0
    # while not done:
    #     print("=" * 10, f"Step {idx + 1}", "=" * 10)
    #     action = int(input(f"Your action is (choose from 0-{num_bulbs-1}): "))
    #     obs, hint, done, _ = env.step(action)
    #     print(hint)
    #     idx += 1
    #     env.render()
