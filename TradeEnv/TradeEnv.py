import numpy as np
import random
from typing import Dict, List, Optional, Any


class TradeArenaEnv:
    """
    Robust TradeArenaEnv:
    - Custom stock-variable dependencies
    - Multi-stock sell/buy per day (cash/position limits)
    - Sell executed first, then buy
    - News reflects variable changes and magnitude
    """

    def __init__(
            self,
            stocks: Optional[List[str]] = None,
            variables: Optional[List[str]] = None,
            stock_var_map: Optional[Dict[str, List[str]]] = None,
            episode_length: int = 120,
            seed: Optional[int] = None,
            init_cash: float = 1000.0,
            max_position_per_stock: float = 100.0,
            noise_scale: float = 0.01,
            news_prob: float = 0.5,
    ):
        self.stocks = stocks if stocks is not None else [f"S{i}" for i in range(5)]
        self.num_stocks = len(self.stocks)

        self.variables = variables if variables is not None else [
            "interest_rate", "inflation", "sentiment", "oil_price", "policy_risk",
            "gdp_growth", "unemployment", "earnings_surprise", "currency_index",
            "commodity_index", "tech_index", "consumer_confidence", "bond_yield",
            "credit_spread", "volatility_index"
        ]
        self.num_variables = len(self.variables)

        self.stock_var_map = stock_var_map  # custom dependencies
        self.episode_length = episode_length
        self.init_cash = init_cash
        self.max_position_per_stock = max_position_per_stock
        self.noise_scale = noise_scale
        self.news_prob = news_prob

        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed if seed is not None else None)
        self.seed_val = seed

        self.prices = np.zeros(self.num_stocks)
        self.positions = np.zeros(self.num_stocks)
        self.cash = 0.0
        self.t = 0
        self.done = False
        self.history = []

        self.alpha = np.zeros((self.num_stocks, self.num_variables))
        self.variable_values = np.zeros(self.num_variables)
        self.variable_changes = np.zeros(self.num_variables)

        self.reset(seed)

    def seed(self, seed: Optional[int] = None):
        self.seed_val = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed if seed is not None else None)

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self.seed(seed)
        self.t = 0
        self.done = False
        self.cash = self.init_cash
        self.positions = np.zeros(self.num_stocks)
        self.prices = self.np_rng.uniform(10, 20, size=self.num_stocks)

        # 初始化 alpha
        self.alpha = np.zeros((self.num_stocks, self.num_variables))
        for i, stock in enumerate(self.stocks):
            if self.stock_var_map and stock in self.stock_var_map:
                vars_for_stock = self.stock_var_map[stock]
                for var in vars_for_stock:
                    if var in self.variables:
                        j = self.variables.index(var)
                        self.alpha[i, j] = self.np_rng.uniform(-0.5, 0.5)
            else:
                self.alpha[i, :] = self.np_rng.uniform(-0.5, 0.5, self.num_variables)

        # 初始化变量
        self.variable_values = self.np_rng.uniform(-1.0, 1.0, size=self.num_variables)
        self.variable_changes = np.zeros(self.num_variables)  # 保证 reset 后安全调用新闻

        self.history = []
        return self._get_observation()

    def _update_variables(self):
        self.variable_changes = self.np_rng.normal(scale=0.02, size=self.num_variables)
        self.variable_values += self.variable_changes

    def _generate_news(self) -> List[Dict[str, Any]]:
        news_today = []
        for i, delta in enumerate(self.variable_changes):
            threshold = 0.01
            if abs(delta) > threshold and self.rng.random() < self.news_prob:
                mag = "slightly" if abs(delta) < 0.05 else "moderately" if abs(delta) < 0.15 else "sharply"
                direction = "increased" if delta > 0 else "decreased"
                token = f"{self.variables[i]} {direction} {mag}"
                news_today.append({"token": token, "effect": {self.variables[i]: delta}})
        return news_today

    def _update_prices(self, news: List[Dict[str, Any]]):
        delta = self.alpha.dot(self.variable_values)
        for n in news:
            for var, effect in n["effect"].items():
                if var in self.variables:
                    var_idx = self.variables.index(var)
                    for stock_idx in range(self.num_stocks):
                        delta[stock_idx] += self.alpha[stock_idx, var_idx] * effect
        delta += self.np_rng.normal(scale=self.noise_scale, size=self.num_stocks)
        self.prices = np.maximum(0.01, self.prices + delta)

    def _get_observation(self) -> Dict[str, Any]:
        return {
            "day": self.t,
            "prices": {self.stocks[i]: float(self.prices[i]) for i in range(self.num_stocks)},
            "variables": {self.variables[i]: float(self.variable_values[i]) for i in range(self.num_variables)},
            "news": self._generate_news(),
            "portfolio": {self.stocks[i]: float(self.positions[i]) for i in range(self.num_stocks)},
            "cash": float(self.cash),
            "remaining_days": self.episode_length - self.t
        }

    def step(self, action: Dict[str, Any]) -> (Dict[str, Any], float, bool, Dict[str, Any]):
        if self.done:
            raise RuntimeError("Episode finished. Call reset()")

        self._update_variables()
        news_today = self._generate_news()

        reward = 0.0
        info = {"events": []}

        sell_orders = action.get("sell", [])
        buy_orders = action.get("buy", [])

        # ✅ 先卖
        for order in sell_orders:
            stock = order["stock"]
            amount = float(order["amount"])
            if stock not in self.stocks:
                continue
            idx = self.stocks.index(stock)
            sell_amt = min(self.positions[idx], amount)
            self.positions[idx] -= sell_amt
            self.cash += sell_amt * self.prices[idx]
            info["events"].append(f"Sold {sell_amt} {stock} at {self.prices[idx]:.2f}")

        # ✅ 再买
        for order in buy_orders:
            stock = order["stock"]
            amount = float(order["amount"])
            if stock not in self.stocks:
                continue
            idx = self.stocks.index(stock)
            price = self.prices[idx]
            cost = price * amount
            if self.cash >= cost:
                self.positions[idx] += amount
                self.cash -= cost
                info["events"].append(f"Bought {amount} {stock} at {price:.2f}")
            else:
                max_afford = self.cash // price
                if max_afford > 0:
                    self.positions[idx] += max_afford
                    self.cash -= max_afford * price
                    info["events"].append(f"Bought {max_afford} {stock} at {price:.2f} (partial due to cash)")
                else:
                    info["events"].append(f"Failed buy {stock}, insufficient cash")

        self._update_prices(news_today)

        total_value = self.cash + np.sum(self.positions * self.prices)
        prev_value = self.history[-1]["total_value"] if self.history else self.init_cash
        reward = total_value - prev_value

        self.history.append({
            "day": self.t,
            "action": action,
            "prices": self.prices.copy(),
            "positions": self.positions.copy(),
            "cash": self.cash,
            "total_value": total_value,
            "news": news_today
        })

        self.t += 1
        if self.t >= self.episode_length:
            self.done = True

        obs = self._get_observation()
        return obs, reward, self.done, info

    def render(self):
        obs = self._get_observation()
        print(f"Day {obs['day']}:")
        print("Prices:", obs["prices"])
        print("Variables:", obs["variables"])
        print("Cash:", obs["cash"])
        print("Positions:", obs["portfolio"])
        print("News:")
        for n in obs["news"]:
            print("  ", n["token"], "| effect:", n["effect"])
        print("-" * 60)


# Example usage
if __name__ == "__main__":
    stock_var_map = {"S0": ["interest_rate", "inflation"], "S1": ["oil_price", "sentiment"]}
    env = TradeArenaEnv(stocks=["S0", "S1", "S2"], episode_length=5, seed=42, stock_var_map=stock_var_map)
    obs = env.reset()
    print("Day 0 prices:", obs["prices"])
    print("Day 0 variables:", obs["variables"])
    print("Day 0 news:", obs["news"])
    print("="*20)
    for day in range(5):
        action = {
            "sell": [{"stock": "S0", "amount": 3}],
            "buy": [{"stock": "S2", "amount": 5}]
        }
        obs, reward, done, info = env.step(action)
        env.render()
        print("Reward:", reward)
        if done:
            break
