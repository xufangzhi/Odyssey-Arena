import numpy as np
import json
import random

class TradeTimelineGenerator:
    def __init__(self,
                 num_days=5,
                 stocks=None,
                 variables=None,
                 dependency_matrix=None,
                 initial_prices=None,
                 initial_variables=None,
                 price_noise_scale=0.0,
                 seed=None):
        self.num_days = num_days
        self.stocks = stocks if stocks else ["AAPL", "GOOG", "TSLA"]
        self.variables = variables if variables is not None else [
            "interest_rate", "inflation", "sentiment", "oil_price", "policy_risk",
            "gdp_growth", "unemployment", "earnings_surprise", "currency_index",
            "commodity_index", "tech_index", "consumer_confidence", "bond_yield",
            "credit_spread", "volatility_index"
        ]
        self.num_stocks = len(self.stocks)
        self.num_vars = len(self.variables)

        # dependency matrix [num_stocks x num_vars]
        if dependency_matrix is None:
            self.dependency_matrix = np.random.uniform(-1.8, 1.8, size=(self.num_stocks, self.num_vars))
        else:
            self.dependency_matrix = np.array(dependency_matrix)

        self.initial_prices = np.array(initial_prices) if initial_prices is not None else np.random.uniform(10, 100, self.num_stocks)
        self.initial_variables = np.array(initial_variables) if initial_variables is not None else np.zeros(self.num_vars)
        self.price_noise_scale = price_noise_scale

        self.rng = np.random.default_rng(seed)

    def generate_timeline(self):
        timeline = {}
        current_vars = self.initial_variables.copy()

        for day in range(1, self.num_days + 1):
            # 随机生成变量变化 delta
            delta_vars = self.rng.normal(0, 0.1, size=self.num_vars)
            current_vars += delta_vars

            # 自动生成简易 news 文本
            news_text_list = []
            for var_name, delta in zip(self.variables, delta_vars):
                if delta > 0.05:
                    news_text_list.append(f"{var_name} increased significantly (+{delta:.2f})")
                elif delta > 0.01:
                    news_text_list.append(f"{var_name} rose slightly (+{delta:.2f})")
                elif delta < -0.05:
                    news_text_list.append(f"{var_name} decreased significantly ({delta:.2f})")
                elif delta < -0.01:
                    news_text_list.append(f"{var_name} dropped slightly ({delta:.2f})")
                else:
                    news_text_list.append(f"{var_name} stable ({delta:.2f})")

            timeline[f"day_{day}"] = {
                "variable_changes": [float(round(d,4)) for d in delta_vars],
                "news_text": " | ".join(news_text_list)
            }

        return timeline

    def generate_config(self, initial_cash=10000.0):
        timeline = self.generate_timeline()
        config = {
            "num_days": self.num_days,
            "stocks": self.stocks,
            "variables": self.variables,
            "dependency_matrix": self.dependency_matrix.tolist(),
            "initial_prices": self.initial_prices.tolist(),
            "initial_variables": self.initial_variables.tolist(),
            "initial_cash": initial_cash,
            "price_noise_scale": self.price_noise_scale,
            "timeline": timeline
        }
        return config

    def save_config(self, path="config.json", initial_cash=10000.0):
        cfg = self.generate_config(initial_cash)
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[INFO] Config saved to {path}")
        return cfg

# =========================
# Example usage
# =========================
if __name__ == "__main__":
    # for i in range(30):
    total_num = 0
    for num_stock in [5,6,7,8,9]:
        for num_var in [4,5,6]:
            for i in range(2):
                generator = TradeTimelineGenerator(
                    num_days=500,
                    stocks=[f"S{n}" for n in range(num_stock)],
                    # variables=["interest_rate", "inflation", "sentiment", "oil_price", "policy_risk", "gdp_growth"],
                    variables=[f"F{n}" for n in range(num_var)],
                    seed=42+i
                )
                config = generator.save_config(r"test_data/trade/test_trade_config_"+f"{total_num+1}.json", initial_cash=50000)
                total_num += 1

# variables = ["interest_rate", "inflation", "sentiment", "oil_price", "policy_risk",
#             "gdp_growth", "unemployment", "earnings_surprise", "currency_index",
#             "commodity_index", "tech_index", "consumer_confidence", "bond_yield",
#             "credit_spread", "volatility_index"]