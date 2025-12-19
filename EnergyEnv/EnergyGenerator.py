import json
import numpy as np
import math


# =====================================================================
# 1. World Generation (Optimized for Smoothness in [0.6, 1.2])
# =====================================================================
def generate_world_profile_v11(days=120, seed=0):
    rng = np.random.default_rng(seed)

    # -------------------------
    # 1. Season Setup
    # -------------------------
    # 将中心点移至 0.9 (0.6和1.2的中点)，确保上下空间对称
    center_wind = 0.85
    center_solar = 0.85
    center_thermal = 0.95  # 火电保持稳定

    # 减小振幅，留出空间给随机噪声
    # 原来是 (0.15, 0.35)，现在改为 (0.08, 0.14)
    # 这样基准波动范围大约在 [0.76, 1.04]，非常安全
    amp_wind = rng.uniform(0.08, 0.14)
    amp_solar = rng.uniform(0.08, 0.14)

    # 火电振幅微调
    amp_thermal = 0.03

    # 相位设置 (保持不变)
    phase_wind = rng.uniform(0, 2 * math.pi)
    delta_phase_solar = rng.uniform(-0.3 * math.pi, 0.3 * math.pi)
    phase_solar = phase_wind + math.pi + delta_phase_solar

    # 初始化数组
    season_wind = np.zeros(days)
    season_solar = np.zeros(days)
    season_thermal = np.zeros(days)
    weather_wind_raw = np.zeros(days)
    weather_solar_raw = np.zeros(days)
    weather_thermal_raw = np.zeros(days)
    eff_wind = np.zeros(days)
    eff_solar = np.zeros(days)
    eff_thermal = np.zeros(days)

    # -------------------------
    # 2. Extreme Events (Softened)
    # -------------------------
    n_storm_events = 3
    n_cloudy_events = 3

    all_days = np.arange(days)
    storm_starts = rng.choice(all_days, size=n_storm_events, replace=False)
    cloudy_starts = rng.choice(all_days, size=n_cloudy_events, replace=False)

    storm_days, cloudy_days = set(), set()
    for d in storm_starts:
        for k in range(rng.integers(2, 4)):  # 持续2-3天
            if 0 <= d + k < days:
                storm_days.add(d + k)
    for d in cloudy_starts:
        for k in range(rng.integers(2, 4)):
            if 0 <= d + k < days:
                cloudy_days.add(d + k)

    # -------------------------
    # 3. Weather Noise (AR1 - Damped)
    # -------------------------
    trend = 0.0
    trend_decay = 0.85
    # 大幅降低噪声尺度，从 0.12 降至 0.045
    # 这样 random walk 不会轻易跑出 ±0.15 的范围
    noise_scale = 0.045
    thermal_noise_scale = 0.015

    for t in range(days):
        # A. Season Component (Sinusoidal)
        # 30天周期
        season_wind[t] = center_wind + amp_wind * math.sin(2 * math.pi * (t % 30) / 30 + phase_wind)
        season_solar[t] = center_solar + amp_solar * math.sin(2 * math.pi * (t % 30) / 30 + phase_solar)
        season_thermal[t] = center_thermal + amp_thermal * math.sin(2 * math.pi * (t % 30) / 30)

        # B. AR(1) Trend Component
        noise = rng.normal(0, noise_scale)
        trend = trend_decay * trend + (1 - trend_decay) * noise

        # 限制 trend 的绝对值，作为第二道防线，防止极个别离群点
        # 如果 trend 超过 ±0.2，进行软压缩
        if abs(trend) > 0.2:
            trend = 0.2 * (trend / abs(trend))

        weather_factor = 1.0 + trend

        weather_wind_raw[t] = weather_factor
        weather_solar_raw[t] = weather_factor
        weather_thermal_raw[t] = 1.0 + rng.normal(0, thermal_noise_scale)

        # C. Combine & Daily Jitter
        # 减小每日的微小抖动 (0.03 -> 0.015) 以增加平滑感
        daily_jitter = 1 + rng.normal(0, 0.015)

        ew = season_wind[t] * weather_factor * daily_jitter
        es = season_solar[t] * weather_factor * daily_jitter
        et = season_thermal[t] * weather_thermal_raw[t]

        # D. Apply Extreme Events (Softened Logic)
        if t in storm_days:
            # 风暴：原来 *1.15 (易破1.2)，改为 *1.10
            # 配合 Center 0.9，大约达到 0.99~1.1 左右，很安全
            ew *= 1.10
            # 风暴时太阳能小幅下降
            es *= 0.80

        if t in cloudy_days:
            # 阴天：原来 *0.5 (易破0.6)，改为 *0.72
            # 0.9 * 0.72 ≈ 0.648，完美落在 0.6 之上，不会被截断
            es *= 0.72
            # 阴天风能微增
            ew *= 1.05

        # E. Final Clip (Safety Net)
        # 由于参数经过精心控制，这里极少会触发硬截断，从而保持曲线丝滑
        eff_wind[t] = np.clip(ew, 0.6, 1.2)
        eff_solar[t] = np.clip(es, 0.6, 1.2)
        eff_thermal[t] = np.clip(et, 0.95, 1.05)

    return {
        "days": days,
        "eff_wind": eff_wind.tolist(),
        "eff_solar": eff_solar.tolist(),
        "eff_thermal": eff_thermal.tolist(),
        "season_wind": season_wind.tolist(),
        "season_solar": season_solar.tolist(),
        "season_thermal": season_thermal.tolist(),
        "weather_wind_raw": weather_wind_raw.tolist(),
        "weather_solar_raw": weather_solar_raw.tolist(),
        "weather_thermal_raw": weather_thermal_raw.tolist(),
        "storm_days": sorted(list(storm_days)),
        "cloudy_days": sorted(list(cloudy_days)),
        "phase_wind": phase_wind,
        "phase_solar": phase_solar,
        "amp_wind": amp_wind,
        "amp_solar": amp_solar,
        "seed": seed,
    }


# 下面保留原有的其余函数，保持不变
def generate_demand_v11(days=120, seed=0):
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


def generate_budget_v11(demand, multiplier=4.2):
    return [multiplier * d for d in demand]


def generate_initial_rated_v11(capacity, demand_day1, rng):
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


def generate_energy_grid_config_v11(days=120, seed=0):
    rng = np.random.default_rng(seed)
    world = generate_world_profile_v11(days, seed)
    demand = generate_demand_v11(days, seed)
    budget = generate_budget_v11(demand, multiplier=4.2)
    capacity = {
        "thermal": 600.0,
        "wind": 350.0,
        "solar": 250.0,
        "battery": 80.0,
    }
    initial_rated = generate_initial_rated_v11(capacity, demand_day1=demand[0], rng=rng)
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
    # 生成30个样本
    for idx in range(0, 30):
        config = generate_energy_grid_config_v11(days=120, seed=42 + idx)
        test_config.append(config)

    print(f"Generated {len(test_config)} configs.")
    # 路径请根据实际情况调整
    with open(f"test_data/energy/test_energy_lite_smooth.json", "w") as file:
        json.dump(test_config, file, indent=4, default=default_serializer)