import numpy as np
import json
import random

class TradeArenaEnv_Deterministic:
    """
    Odyssey Arena - AI Trading Environment (Deterministic version)
    ---------------------------------------------------------------
    - Controlled by external config file (no random state inside)
    - Agent observes current prices + next day's news
    - News influences next day's prices (delayed effect)
    - Actions can include multiple buy/sell ops, respecting available cash
    """

    def __init__(self, cfg):
        # === Load config ===
        # with open(config_path, "r") as f:
        #     cfg = json.load(f)

        self.num_days = cfg["num_days"]
        self.stocks = cfg["stocks"]
        self.variables = cfg["variables"]

        # shape: [num_stocks, num_variables]
        self.dependency_matrix = np.array(cfg["dependency_matrix"])
        self.initial_prices = np.array(cfg["initial_prices"])
        self.initial_variables = np.array(cfg["initial_variables"])
        self.timeline = cfg["timeline"]  # day_i: {variable_changes: [...], news_text: "..."}

        # noise config
        self.price_noise_scale = cfg.get("price_noise_scale", 0.0)

        # trading config
        self.initial_cash = cfg.get("initial_cash", 10000.0)

        # === Internal state ===
        self.reset()

    # -------------------------------------------------------
    def reset(self):
        """Reset to day 0"""
        self.t = 0
        self.cash = self.initial_cash
        self.positions = np.zeros(len(self.stocks))
        self.prices = self.initial_prices.copy()
        self.variables_state = self.initial_variables.copy()

        # pre-compute day0 news_next_day (agent can see day1 news)
        self.next_day_news = self.timeline.get("day_1", None)
        return self._get_observation()

    # -------------------------------------------------------
    def _get_observation(self):
        """Return current market observation."""
        obs = {
            "day": self.t,
            "prices": {s: float(p) for s, p in zip(self.stocks, self.prices)},
            "cash": float(self.cash),
            "positions": {s: int(pos) for s, pos in zip(self.stocks, self.positions)},
            "total_value": float(self.cash + np.sum(self.positions * self.prices)),
            "news_next_day": self.next_day_news["variable_changes"] if self.next_day_news else None,
            "news_next_day_text": self.next_day_news["news_text"] if self.next_day_news else None
        }
        return obs

    # -------------------------------------------------------
    def step(self, action):
        """
        action = {
            "buy": {"STOCK_A": 10, "STOCK_B": 5},
            "sell": {"STOCK_C": 2}
        }
        """
        assert isinstance(action, dict)

        # 1️⃣ execute sells first
        for stock, qty in action.get("sell", {}).items():
            if stock in self.stocks:
                idx = self.stocks.index(stock)
                try:
                    qty = int(qty)
                except:
                    qty = 0.0
                qty = min(qty, self.positions[idx])
                revenue = self.prices[idx] * qty
                self.positions[idx] -= qty
                self.cash += revenue

        # 2️⃣ then buys (subject to available cash)
        for stock, qty in action.get("buy", {}).items():
            if stock in self.stocks:
                idx = self.stocks.index(stock)
                qty = int(qty)
                cost = self.prices[idx] * qty
                if cost <= self.cash:
                    self.positions[idx] += qty
                    self.cash -= cost

        # 3️⃣ advance one day
        self.t += 1
        done = self.t >= self.num_days

        # 4️⃣ update variable states & prices based on today's news (day_t)
        if not done:
            news_today = self.timeline.get(f"day_{self.t}", None)
            if news_today:
                deltas = np.array(news_today["variable_changes"])
                self.variables_state += deltas
                self._update_prices_from_variables(deltas)

        # 5️⃣ prepare next day's news for observation
        self.next_day_news = self.timeline.get(f"day_{self.t + 1}", None) if not done else None

        # 6️⃣ reward: total portfolio value change
        reward = self._compute_reward()
        obs = self._get_observation()
        return obs, reward, done, {}

    # -------------------------------------------------------
    def _update_prices_from_variables(self, delta_vars):
        """Update prices deterministically based on variable changes."""
        delta_price = self.dependency_matrix @ delta_vars
        noise = np.zeros_like(delta_price) if self.price_noise_scale == 0 else np.random.normal(
            0, self.price_noise_scale, len(self.stocks)
        )
        self.prices += delta_price + noise
        self.prices = np.clip(self.prices, 0.1, None)  # avoid negative prices

    # -------------------------------------------------------
    def _compute_reward(self):
        """Reward = total portfolio value change since previous day."""
        total_value = self.cash + np.sum(self.positions * self.prices)
        return round(float(total_value),2)

    # -------------------------------------------------------
    def render(self):
        lines = []
        lines.append(f"\n=== Day {self.t} ===")
        for s, p in zip(self.stocks, self.prices):
            lines.append(f"{s}: {p:.2f} (holding {int(self.positions[self.stocks.index(s)])})")
        lines.append(f"Cash: {self.cash:.2f}")
        lines.append(f"Total Value: {self.cash + np.sum(self.positions * self.prices):.2f}")
        if self.next_day_news:
            lines.append(f"Next day news: {self.next_day_news['news_text']}")
        else:
            lines.append("No more news.")

        output = "\n".join(lines)
        print(output)

    def get_render_obs(self):
        lines = []
        lines.append(f"\n=== Day {self.t} ===")
        for s, p in zip(self.stocks, self.prices):
            lines.append(f"{s}: {p:.2f} (holding {int(self.positions[self.stocks.index(s)])})")
        lines.append(f"Cash: {self.cash:.2f}")
        lines.append(f"Total Value: {self.cash + np.sum(self.positions * self.prices):.2f}")
        if self.next_day_news:
            lines.append(f"Next day news: {self.next_day_news['news_text']}")
        else:
            lines.append("No more news.")

        output = "\n".join(lines)
        return output



if __name__ == "__main__":
    with open("trade_env_config.json") as f:
        example_cfg = json.load(f)
    env = TradeArenaEnv_Deterministic(cfg=example_cfg)
    obs = env.reset()
    env.render()

    # print("Day 0 prices:", obs["prices"])
    # print("Day 0 variables:", obs["variables"])
    # print("Day 0 news:", obs["news"])
    print("="*20)

    # for _ in range(2):
    #     obs, reward, done, info = env.step({"buy":[{"stock":"S0","amount":2}]})
    #     env.render()
    for day in range(50):
        action = {
            "buy": {"S1": 10},
            "sell": {"S2": 1}
        }
        obs, reward, done, info = env.step(action)
        env.render()
        print("Reward:", reward)
        if done:
            break