import json
import numpy as np
import EnergyGenerator  # 假设这是你之前的生成器文件


class DynamicEnergyGrid:
    """
    Dynamic Energy Grid Environment v11 (Battery Fixed)
    - Battery: Bidirectional (Charge/Discharge) with SoC tracking
    - Battery Cost: Based on operation (wear & tear), not generation price
    - Logic Fixes: Budget violation calculation, Observation consistency
    """

    def __init__(self, config):
        self.cfg = config
        self.horizon = config["horizon"]

        self.target_stability = config['target_stability']
        self.target_carbon = config['target_carbon']

        self.world = config["world"]
        self.demand_series = config["demand"]
        self.budget_series = config["budget"]

        self.capacity = config["capacity"]
        self.initial_rated_cfg = config["initial_rated"]
        self.initial_stability = config["initial_stability"]

        self.prices = config["prices"]
        self.penalty = config["penalty"]

        self.supply_total = 0
        self.budget_violation = False
        self.demand_violation = False
        self.stability_violation = False
        self.stability_sta = []
        self.stability_avg = 1.0

        self.share_thermal = 0.0

        self.violation_days_cont = 0
        # [NEW] 电池物理参数配置
        # 电池运维损耗成本 (远低于发电成本，鼓励使用)
        self.battery_op_cost = 0.1
        self.battery_cur = 0.0
        self.reset()

    # ------------------------------------------
    def reset(self):
        self.t = 0

        # 重置额定功率设定
        self.thermal_rated = self.initial_rated_cfg["thermal"]
        self.wind_rated = self.initial_rated_cfg["wind"]
        self.solar_rated = self.initial_rated_cfg["solar"]

        # [NEW] 电池状态重置
        self.battery_rated = 0.0  # 意图功率
        self.soc = 0.5  # 初始电量 50%
        self.prev_soc = 0.5

        self.prev_rated = dict(self.initial_rated_cfg)
        self.prev_rated["battery"] = 0.0

        self.stability = self.initial_stability
        self.stability_avg = 1.0
        self.stability_sta = []
        # 当期实际功率
        self.thermal_actual = 0
        self.wind_actual = 0
        self.solar_actual = 0
        self.battery_actual = 0  # 正=放电，负=充电
        self.battery_cur = 0.0

        self.supply_total = 0
        self.budget_violation = False
        self.demand_violation = False
        self.stability_violation = False
        # 累计指标
        self.cum_unmet = 0
        self.cum_carbon = 0
        self.cum_budget_violation = 0
        self.cum_ramp = 0

        self.done = False
        return self._get_obs()

    # ------------------------------------------
    def step(self, action):
        if self.done:
            raise RuntimeError("Episode finished. Call reset() first.")

        t = self.t

        # -------------------- 1. 解析 Action --------------------
        # 发电资源 (只能为正)
        self.thermal_rated = self._clamp(action.get("thermal", 0.0), 0, self.capacity["thermal"])
        self.wind_rated = self._clamp(action.get("wind", 0.0), 0, self.capacity["wind"])
        self.solar_rated = self._clamp(action.get("solar", 0.0), 0, self.capacity["solar"])

        # [NEW] 电池资源 (双向: 负=充, 正=放)
        # 限制意图在 [-额定, +额定] 之间
        raw_bat = action.get("battery", 0.0)
        self.battery_rated = self._clamp(raw_bat, -self.capacity["battery"], self.capacity["battery"])

        # -------------------- 2. 发电效率计算 --------------------
        eff_th = self.world["eff_thermal"][t]
        eff_w = self.world["eff_wind"][t]
        eff_s = self.world["eff_solar"][t]

        # 基础发电量
        self.thermal_actual = self.thermal_rated * eff_th
        self.wind_actual = self.wind_rated * eff_w
        self.solar_actual = self.solar_rated * eff_s

        # -------------------- 3. [NEW] 电池物理模拟 (核心修改) --------------------
        # current_energy_mwh = self.soc * self.battery_capacity_mwh
        # empty_space_mwh = self.battery_capacity_mwh - current_energy_mwh
        #
        # actual_bat_flow = 0.0

        if self.battery_rated >= 0:  # 意图：放电
            # 限制：不能超过剩余电量
            actual_bat_flow = min(self.battery_rated, self.battery_cur)
            # # 更新 SoC (假设步长为1小时)
            # self.soc -= actual_bat_flow / self.battery_capacity_mwh

        elif self.battery_rated < 0:  # 意图：充电
            # 限制：不能超过剩余空间
            power_to_charge = abs(self.battery_rated)
            actual_charged = min(power_to_charge, self.capacity["battery"] - self.battery_cur)
            actual_bat_flow = - actual_charged  # 负流向
            # # 更新 SoC
            # self.soc += actual_charged / self.battery_capacity_mwh

        self.battery_actual = actual_bat_flow
        self.battery_cur -= actual_bat_flow
        # self.prev_soc = self.soc  # 记录用于 next state

        # -------------------- 4. 供需平衡计算 --------------------
        # 总供给 = 发电 + 电池流向
        # 注意：如果 battery_actual 为负（充电），它会减少对 Grid 的供给（电被电池吃掉了）
        supply = (
                self.thermal_actual +
                self.wind_actual +
                self.solar_actual +
                self.battery_actual
        )
        # 电能供给判断
        supply = max(1e-6, supply)

        demand = self.demand_series[t]

        if demand > 1e-6:
            unmet = max(0, 1 - supply / demand)
        else:
            unmet = 0
        self.demand_violation = (supply < demand)

        self.cum_unmet += unmet



        # -------------------- 5. [FIX] 成本与预算 --------------------
        # 只有发电需要支付昂贵的燃料/维护费
        # 电池只支付低廉的损耗费，且不管是充还是放都产生损耗
        cost_today = (
                self.thermal_rated * self.prices["thermal"] +
                self.wind_rated * self.prices["wind"] +
                self.solar_rated * self.prices["solar"] +
                abs(self.battery_actual) * self.battery_op_cost
        )

        budget_today = self.budget_series[t]

        # [FIX] 预算违约修正：只有当 Cost > Budget 时才是正数
        self.budget_violation = (cost_today > budget_today)

        budget_over = max(0, cost_today - budget_today)
        budget_unmet = budget_over / budget_today
        self.cum_budget_violation += budget_over

        # -------------------- 6. Ramp (爬坡) 计算 --------------------
        ramp = (
                abs(self.thermal_rated - self.prev_rated["thermal"]) +
                abs(self.wind_rated - self.prev_rated["wind"]) +
                abs(self.solar_rated - self.prev_rated["solar"]) +
                abs(self.battery_rated - self.prev_rated["battery"])
        )
        self.cum_ramp += ramp

        # 更新历史
        self.prev_rated = {
            "thermal": self.thermal_rated,
            "wind": self.wind_rated,
            "solar": self.solar_rated,
            "battery": self.battery_rated,
        }

        # -------------------- 7. 碳排放 --------------------
        # 分母是发电总量（不含电池放电，因为那是存储的绿电或火电）
        # 或者简化处理：分母为 Grid 总供给
        self.supply_total += supply
        self.cum_carbon += self.thermal_actual
        self.share_thermal = self.cum_carbon / self.supply_total

        # -------------------- 8. 稳定性与 Reward --------------------
        max_ramp = sum(self.capacity.values())
        normalized_ramp = min(1.0, ramp / max_ramp)

        a, b, c = 2, 2, 0.5
        # stability = 1 - a * unmet - b * budget_unmet - c * normalized_ramp
        stability = 1 - c * normalized_ramp
        if self.budget_violation:
            stability -= 0.5
        if self.demand_violation:
            stability -= 0.5

        self.stability = self._clamp(stability, 0, 1)

        self.stability_violation = (self.stability <= self.target_stability)

        self.stability_sta.append(self.stability)
        self.stability_avg = np.mean(self.stability_sta)
        # 步长奖励 (Dense Reward) - 可选
        # step_reward = (
        #         - self.penalty["unmet"] * unmet
        #         - self.penalty["budget"] * (budget_violation / (budget_today + 1))
        #         - self.penalty["ramp"] * normalized_ramp
        # )

        # -------------------- 9. 时间推进 --------------------
        self.t += 1
        done = (self.t >= self.horizon)
        self.done = done

        if self.budget_violation or self.demand_violation:
            self.violation_days_cont += 1
            if self.violation_days_cont==3:
                done = True
                self.done = done
        else:
            self.violation_days_cont = 0

        # -------------------- 10. Obs & Info --------------------
        obs = self._get_obs()

        if self.done and self.stability_avg>self.target_stability and self.share_thermal<self.target_carbon and self.violation_days_cont<3:
            reward = 1.0
        else:
            reward = 0.0

        info = {
            "cost_today": cost_today,
            "budget_today": budget_today,
            "budget_violation": self.budget_violation,
            "demand_violation": self.demand_violation,
            "stability_violation": self.stability_violation,
            "demand_unmet": unmet,
            "budget_unmet": budget_unmet,
            "violation_days_cont": self.violation_days_cont,
            # "soc": self.soc,
            "stability": self.stability,
            "battery_flow": self.battery_actual,
            # "step_reward": step_reward
        }

        return obs, reward, done, info

    # ------------------------------------------
    def _clamp(self, x, lo, hi):
        return max(lo, min(hi, x))

    # ------------------------------------------
    def _trend_sentence(self, today, tomorrow, typ):
        delta = tomorrow - today
        x = abs(delta)
        if x < 0.01:
            phrase = "stable"
        elif x < 0.03:
            phrase = "slightly increase" if delta > 0 else "slightly decrease"
        elif x < 0.06:
            phrase = "moderately increase" if delta > 0 else "moderately decrease"
        else:
            phrase = "sharply increase" if delta > 0 else "sharply decrease"

        return f"{typ}: {phrase}"

    # ------------------------------------------
    def _get_obs(self):
        h = self.horizon
        t = min(self.t, h - 1)
        t_yesterday = max(0, self.t - 1)

        # 基础数据
        demand_today = self.demand_series[t]
        budget_today = self.budget_series[t]

        # 昨天的数据 (用于计算 change 或 history)
        demand_prev = self.demand_series[t_yesterday]

        # 气象预报
        # w_today = self.world["weather_wind_raw"][t]
        # s_today = self.world["weather_solar_raw"][t]
        # w_prev = self.world["weather_wind_raw"][t_yesterday]
        # s_prev = self.world["weather_solar_raw"][t_yesterday]
        #
        # nl_forecast = (
        #         self._trend_sentence(w_prev, w_today, "Wind") + ", " +
        #         self._trend_sentence(s_prev, s_today, "Solar")
        # )

        # [FIX] 确保 t=0 时不返回 None，而是返回 0 向量
        if self.t == 0:
            actual_dict = {k: 0.0 for k in ["thermal", "wind", "solar", "battery", "supply", "demand_met"]}
            efficiency_dict = {k: 0.0 for k in ["thermal", "wind", "solar"]}
            prev_stability = 1.0  # 初始视为稳定
            nl_forecast_val = "First day, no history."
        else:
            supply_prev = (
                    self.thermal_actual + self.wind_actual +
                    self.solar_actual + self.battery_actual
            )
            actual_dict = {
                "thermal": self.thermal_actual,
                "wind": self.wind_actual,
                "solar": self.solar_actual,
                "battery": self.battery_actual,
                "supply": supply_prev,
                "demand_met": self._clamp(supply_prev / max(1e-6, demand_prev), 0, 1)
            }
            efficiency_dict = {
                "thermal": self.world["eff_thermal"][t_yesterday],
                "wind": self.world["eff_wind"][t_yesterday],
                "solar": self.world["eff_solar"][t_yesterday],
            }

            # nl_forecast_val = nl_forecast

        obs = {
            "day": t,
            "rated_prev": self.prev_rated,  # 上一时刻的决策
            "actual_prev": actual_dict,  # 上一时刻的效果
            "battery_cur": self.battery_cur,
            # "efficiency_prev": efficiency_dict,
            "stability": self.stability_avg,
            "carbon": self.share_thermal,
            "demand_today": demand_today,
            "budget_today": budget_today,
            "demand_violation": self.demand_violation,
            "stability_violation": self.stability_violation,
            "violation_days_cont": self.violation_days_cont,
            # "nl_forecast": nl_forecast_val,
        }

        return obs

    def return_obs(self):
        h = self.horizon
        t = min(self.t, h - 1)  # 当前想要规划的时间 t
        t_prev = max(0, self.t - 1)  # 刚刚过去的时间 t-1

        output = f"\n=== Day {t} ===\n"

        # 1. 状态显示
        output += f"Status: Stability={self.stability_avg:.3f} | Carbon={self.share_thermal:.3f} | Battery={self.battery_cur}\n"

        # 2. 上一步的结果
        if self.t > 0:
            bat_act = self.battery_actual
            bat_str = f"Discharge {bat_act:.1f}" if bat_act >= 0 else f"Charge {abs(bat_act):.1f}"

            output += "▶ Last Step Summary:\n"
            output += f"   Actual Gen: Thermal={self.thermal_actual:.1f}, Wind={self.wind_actual:.1f}, Solar={self.solar_actual:.1f}\n"
            output += f"   Battery:    {bat_str}\n"

            supply = self.thermal_actual + self.wind_actual + self.solar_actual + self.battery_actual
            demand = self.demand_series[t_prev]
            output += f"   Grid:       Supply {supply:.1f} / Demand {demand:.1f}\n"

            # 3. 费用 (Commented out in the original code, so not included in the final string)
            cost = (
                    self.thermal_rated * self.prices["thermal"] +
                    self.wind_rated * self.prices["wind"] +
                    self.solar_rated * self.prices["solar"] +
                    abs(self.battery_actual) * self.battery_op_cost
            )
            budget = self.budget_series[t_prev]
            vio = max(0, cost - budget)
            output += f"   Finance:    Cost {cost:.1f} / Budget {budget:.1f} (Vio: {vio:.1f})\n"

            if self.demand_violation or self.budget_violation:
                output += f"   {'Demand Violated ' if self.demand_violation else ''}" \
                          f"{'Budget Violated' if self.budget_violation else ''}\n"
            else:
                output += f"   Demand Satisfied, Budget Satisfied.\n"


        output += "▶ Forecast for Next Day:\n"
        # obs = self._get_obs()
        output += f"   Demand: {self.demand_series[t]:.2f}\n"
        output += f"   Budget: {self.budget_series[t]:.2f}\n"
        # output += f"   Weather Forecast: {obs['nl_forecast']}"

        return output


    # ------------------------------------------
    def render(self):
        h = self.horizon
        t = min(self.t, h - 1)  # 当前想要规划的时间 t
        t_prev = max(0, self.t - 1)  # 刚刚过去的时间 t-1

        print(f"\n=== Day {t} ===")

        # 1. 状态显示
        print(f"Status: Stability={self.stability_avg:.3f} | Carbon={self.share_thermal:.3f} | Battery={self.battery_cur}")

        # 2. 上一步的结果
        if self.t > 0:
            bat_act = self.battery_actual
            bat_str = f"Discharge {bat_act:.1f}" if bat_act >= 0 else f"Charge {abs(bat_act):.1f}"

            print("▶ Last Step Summary:")
            print(f"   Actual Gen: Thermal={self.thermal_actual:.1f}, Wind={self.wind_actual:.1f}, Solar={self.solar_actual:.1f}")
            print(f"   Battery:    {bat_str}")

            supply = self.thermal_actual + self.wind_actual + self.solar_actual + self.battery_actual
            demand = self.demand_series[t_prev]
            print(f"   Grid:       Supply {supply:.1f} / Demand {demand:.1f}")

            # 3. 费用
            cost = (
                    self.thermal_rated * self.prices["thermal"] +
                    self.wind_rated * self.prices["wind"] +
                    self.solar_rated * self.prices["solar"] +
                    abs(self.battery_actual) * self.battery_op_cost
            )
            budget = self.budget_series[t_prev]
            vio = max(0, cost - budget)
            print(f"   Finance:    Cost {cost:.1f} / Budget {budget:.1f} (Vio: {vio:.1f})")

            violation_list = []
            if self.demand_violation:
                violation_list.append("Demand")
            if self.budget_violation:
                violation_list.append("Budget")
            if violation_list:
                print(f"    Violation: {' ,'.join(violation_list)}\n")
            else:
                print(f"    Violation: None\n")

        # 4. 今天的预测
        print("▶ Forecast for Next Day:")
        obs = self._get_obs()
        print(f"   Demand: {self.demand_series[t]:.2f}")
        print(f"   Budget: {self.budget_series[t]:.2f}")
        print(f"   Weather Forecast: {obs['nl_forecast']}")


if __name__ == "__main__":
    # 测试代码
    # 1. 生成配置
    config = EnergyGenerator.generate_energy_grid_config_v11(days=20, seed=42)

    # 2. 修改价格以测试电池逻辑 (让火电极贵)
    config["prices"]["thermal"] = 100.0
    config["prices"]["battery"] = 999.0  # 这个配置值现在应该被代码里的 op_cost 覆盖/忽略

    env = DynamicEnergyGrid(config)
    obs = env.reset()
    env.render()

    # 3. 手动测试序列
    # Day 0: 需求低，充电 (Charge)
    print("\n>>> ACTION: Charging Battery...")
    act0 = {"thermal": 50, "wind": 0, "solar": 0, "battery": -20}  # 充 20
    obs, r, done, info = env.step(act0)
    env.render()

    # Day 1: 需求高，放电 (Discharge)
    print("\n>>> ACTION: Discharging Battery...")
    act1 = {"thermal": 0, "wind": 0, "solar": 0, "battery": 20}  # 放 20
    obs, r, done, info = env.step(act1)
    env.render()