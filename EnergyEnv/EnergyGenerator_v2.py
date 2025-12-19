import json
import numpy as np
import math


# =====================================================================
# 生成更真实的随机 base pattern（用在每周期内部）
# =====================================================================
def generate_random_base_pattern(T, eff_min, eff_max, seed=None):
    rng = np.random.default_rng(seed)

    # 1. Piecewise-linear segments：天气连续段落
    pattern = []
    t = 0
    while t < T:
        seg_len = rng.integers(2, 6)  # 每段 2~5 天
        seg_len = min(seg_len, T - t)
        value = rng.uniform(eff_min + 0.05, eff_max - 0.05)
        pattern.extend([value] * seg_len)
        t += seg_len
    pattern = np.array(pattern)

    # 2. Random walk：趋势变化
    drift = np.cumsum(rng.normal(0, 0.02, size=T))
    pattern = pattern + drift

    # 3. Spikes (5% chance)：模拟极端天气
    for i in range(T):
        if rng.random() < 0.05:
            pattern[i] += rng.normal(0.1, 0.05)

    # 4. Final clip
    pattern = np.clip(pattern, eff_min, eff_max)
    return pattern


# =====================================================================
# 周期重复 + 周期噪声 + 日噪声（结构与之前一致）
# =====================================================================
def generate_periodic_efficiency(
    length=120,
    min_period=15,
    max_period=30,
    cycle_noise=0.01,
    daily_noise=0.01,
    eff_min=0.6,
    eff_max=1.1,
    seed=None,
):
    rng = np.random.default_rng(seed)

    # 1. 随机周期
    T = rng.integers(min_period, max_period + 1)

    # 2. 基准周期模式（更随机版）
    base = generate_random_base_pattern(T, eff_min, eff_max, seed=seed)

    # 3. 平铺周期
    cycles = int(np.ceil(length / T))
    pattern = np.tile(base, cycles)[:length]

    # 4. 不同周期加入偏移
    final_curve = pattern.copy()
    for c in range(cycles):
        offset = rng.normal(0, cycle_noise)
        start = c * T
        end = min((c + 1) * T, length)
        final_curve[start:end] += offset

    # 5. 每日噪声
    final_curve += rng.normal(0, daily_noise, size=length)

    # 6. Clip
    final_curve = np.clip(final_curve, eff_min, eff_max)

    return final_curve, T


# =====================================================================
# ★ generate_world_profile_v12（最终更随机版）
# =====================================================================
def generate_world_profile_v12(days=120, seed=42):
    rng = np.random.default_rng(seed)

    # ---- Wind ----
    eff_wind, wind_T = generate_periodic_efficiency(
        length=days,
        min_period=15,
        max_period=25,
        eff_min=0.6,
        eff_max=1.05,
        seed=seed + 101,
    )

    # ---- Solar ----
    eff_solar, solar_T = generate_periodic_efficiency(
        length=days,
        min_period=15,
        max_period=25,
        eff_min=0.65,
        eff_max=1.1,
        seed=seed + 202,
    )

    # ---- Thermal：保持稳定 ----
    eff_thermal = np.clip(
        1.0 + rng.normal(0, 0.01, size=days),
        0.95, 1.05
    )

    # ---- 返回结构保持 v12 完全一致 ----
    return {
        "days": days,
        "eff_wind": eff_wind.tolist(),
        "eff_solar": eff_solar.tolist(),
        "eff_thermal": eff_thermal.tolist(),
        "wind_period": int(wind_T),
        "solar_period": int(solar_T),
        "seed": seed,
    }




# =====================================================================
# 下面保持你原来的 demand / budget / capacity 等函数不变
# =====================================================================
def generate_demand_v12(days=120, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.uniform(320, 480)
    amp = rng.uniform(0.25, 0.35)
    noise = 0.04
    phase_demand = rng.uniform(0, 2 * math.pi)
    demand = np.zeros(days)
    for t in range(days):
        season = math.sin(2 * math.pi * (t % 30) / 30 + phase_demand)
        demand[t] = base * (1 + amp * season) * (1 + rng.normal(0, noise))
    return demand.tolist()


def generate_budget_v12(demand, multiplier=4.2):
    return [multiplier * d for d in demand]


def generate_target_v12(seed=0):
    rng = np.random.default_rng(seed)
    # stability, carbon
    return rng.uniform(0.950, 0.980), rng.uniform(0.660, 0.720)


def generate_initial_rated_v12(capacity, demand_day1, rng):
    p_th = rng.uniform(0.55, 0.75)
    p_w = rng.uniform(0.20, 0.40)
    p_s = rng.uniform(0.15, 0.35)
    p_b = 0.0
    raw = np.array([p_th, p_w, p_s, p_b])
    raw = raw / raw.sum()
    target_total = demand_day1 * rng.uniform(0.95, 1.05)
    thermal_r0 = min(raw[0] * target_total, capacity["thermal"])
    wind_r0 = min(raw[1] * target_total, capacity["wind"])
    solar_r0 = min(raw[2] * target_total, capacity["solar"])
    battery_r0 = min(raw[3] * target_total, capacity["battery"])
    return {
        "thermal": thermal_r0,
        "wind": wind_r0,
        "solar": solar_r0,
        "battery": battery_r0,
    }


# =====================================================================
# 顶层配置生成函数
# =====================================================================
def generate_energy_grid_config_v12(days=120, seed=0):
    rng = np.random.default_rng(seed)

    world = generate_world_profile_v12(days, seed)
    demand = generate_demand_v12(days, seed)
    budget = generate_budget_v12(demand, multiplier=4.2)
    target = generate_target_v12(seed)

    capacity = {
        "thermal": 600.0,
        "wind": 350.0,
        "solar": 250.0,
        "battery": 80.0,
    }

    initial_rated = generate_initial_rated_v12(capacity, demand_day1=demand[0], rng=rng)

    prices = {
        "thermal": 3.0,
        "wind": 5.0,
        "solar": 6.0,
        "battery": 0.1,
    }

    penalty = {
        "unmet": 3.0,
        "carbon": 1.0,
        "budget": 2.0,
        "ramp": 0.0005,
        "stability": 1.0,
    }

    config = {
        "horizon": days,
        "target_stability": target[0],
        "target_carbon": target[1],
        "world": world,
        "demand": demand,
        "budget": budget,
        "capacity": capacity,
        "initial_rated": initial_rated,
        "initial_stability": 1.0,
        "prices": prices,
        "penalty": penalty,
        "seed": seed,
    }

    return config



# =====================================================================
# JSON序列化助手 & main
# =====================================================================
def default_serializer(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')


if __name__ == "__main__":
    test_config = []
    for idx in range(30):
        cfg = generate_energy_grid_config_v12(days=120, seed=42 + idx)
        test_config.append(cfg)

    print(f"Generated {len(test_config)} configs.")

    with open("test_data/energy/test_energy_lite_251207.json", "w") as f:
        json.dump(test_config, f, indent=4, default=default_serializer)
