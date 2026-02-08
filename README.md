<h1 align="center">
🏁 OdysseyArena: Benchmarking Large Language Models For Long-Horizon, Active and Inductive Interactions
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2602.05843"><b>[📜 Paper]</b></a> •
  <a href="https://github.com/xufangzhi/Odyssey-Arena/"><b>[🐱 GitHub]</b></a>
  <a href="https://yayayacc.github.io/Odyssey-Home/"><b>[🏆 Leaderboard]</b></a>
  <a href="https://huggingface.co/spaces/beatccjiang/OdysseyArena/"><b>[🌍 Space]</b></a>
  
</p>


<!-- <p align="center"> Repo for "MUR: Momentum Uncertainty Guided Reasoning For Large Language Models</a>"</p>
<a href="https://arxiv.org/abs/2507.14958" target="_blank"> -->

## 🔥 News

- [2026/02] 🔥🔥🔥 OdysseyArena is released !!!

## 🌍 Environments (What Should Agents Do)

- **TurnOnLights (LightEnv)**: toggle bulbs to turn all bulbs on, under hidden dependency rules between bulbs.
- **AI Trading (TradeEnv)**: trade multiple stocks over time to maximize final portfolio value under market dynamics and constraints.
- **Energy Grid (EnergyEnv)**: schedule generation/storage each day to meet energy demand and daily budget while maintaining grid stability and reducing carbon over a long horizon.
- **Computer-using / Repo Setup (RepoEnv)**: act like a developer to fix a broken Python repo by running terminal commands (e.g., pip install/uninstall, run scripts) until `python run.py` succeeds.

### 📊 Benchmark Stats

We provide two versions of datasets for each environment:
- **Lite**: 30 samples per environment (Recommended)
- **Pro**: 200 samples per environment

| Env | # Samples (Lite) | # Samples (Pro) | Max Turns |
|---|:---:|:---:|:---:|
| TurnOnLights (`LightEnv`) | 30 | 200 | 200 |
| AI Trading (`TradeEnv`) | 30 | 200 | 120 |
| Energy Grid (`EnergyEnv`) | 30 | 200 | 120 |
| Computer-using / Repo Setup (`RepoEnv`) | 30 | 200 | 120 |
| **Odyssey-Arena (Total)** | **120** | **800** | **120-200** |

## Leaderboard of OdysseyArena-Lite

<p>
You can also refer to the 🏆<a href="https://yayayacc.github.io/Odyssey-Home/"><b>Leaderboard</b></a> for more details.
</p>

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="2">Turn On Lights</th>
<th colspan="2">AI Trading</th>
<th colspan="2">Energy Dispatch</th>
<th colspan="2">Repo System</th>
</tr>

<tr>
<th>Avg@4</th><th>Pass@4</th>
<th>Avg@4</th><th>Pass@4</th>
<th>Avg@4</th><th>Pass@4</th>
<th>Avg@4</th><th>Pass@4</th>
</tr>
</thead>

<tbody>

<tr>
<td>Human</td>
<td>81.67</td><td>100.00</td>
<td>+92.55%</td><td>+197.23%</td>
<td>25.00</td><td>60.00</td>
<td>77.50</td><td>100.00</td>
</tr>

<tr style="background-color:#E6F4F1;">
<td>Gemini 3 Pro Preview</td>
<td><b>44.17</b></td><td><b>76.67</b></td>
<td><b>+67.71%</b></td><td><b>+76.94%</b></td>
<td><b>30.00</b></td><td>36.67</td>
<td><b>65.83</b></td><td>80.00</td>
</tr>

<tr style="background-color:#E6F4F1;">
<td>GPT-5</td>
<td>28.33</td><td>40.00</td>
<td>+17.32%</td><td>+20.47%</td>
<td>23.33</td><td><b>40.00</b></td>
<td>62.50</td><td><b>83.33</b></td>
</tr>

<tr style="background-color:#E6F4F1;">
<td>Gemini 2.5 Pro</td>
<td>29.17</td><td>50.00</td>
<td>+33.02%</td><td>+40.12%</td>
<td>10.83</td><td>26.67</td>
<td>50.00</td><td>66.67</td>
</tr>

<tr>
<td>gpt-oss-120b (high)</td>
<td>27.50</td><td>40.00</td>
<td>+23.27%</td><td>+27.47%</td>
<td>0.00</td><td>0.00</td>
<td>18.33</td><td>33.33</td>
</tr>

<tr>
<td>DeepSeek-V3.2</td>
<td>18.33</td><td>36.67</td>
<td>+8.62%</td><td>+12.88%</td>
<td>0.00</td><td>0.00</td>
<td>48.33</td><td>76.67</td>
</tr>

<tr style="background-color:#E6F4F1;">
<td>Grok 4 Fast</td>
<td>14.17</td><td>40.00</td>
<td>+5.70%</td><td>+11.52%</td>
<td>0.00</td><td>0.00</td>
<td>38.33</td><td>60.00</td>
</tr>

<tr>
<td>Qwen3-235B-A22B-Instruct</td>
<td>15.00</td><td>43.33</td>
<td>+11.26%</td><td>+17.67%</td>
<td>0.00</td><td>0.00</td>
<td>15.83</td><td>36.67</td>
</tr>

<tr>
<td>Qwen3-30B-A3B-Instruct</td>
<td>11.67</td><td>26.67</td>
<td>+4.76%</td><td>+8.94%</td>
<td>0.00</td><td>0.00</td>
<td>26.67</td><td>50.00</td>
</tr>

<tr>
<td>gpt-oss-120b (medium)</td>
<td>16.67</td><td>40.00</td>
<td>+3.21%</td><td>+7.09%</td>
<td>0.00</td><td>0.00</td>
<td>2.50</td><td>6.67</td>
</tr>

<tr>
<td>GLM-4-32B-0414</td>
<td>14.17</td><td>33.33</td>
<td>+3.14%</td><td>+7.24%</td>
<td>0.00</td><td>0.00</td>
<td>9.17</td><td>30.00</td>
</tr>

<tr>
<td>gpt-oss-120b (low)</td>
<td>7.50</td><td>13.33</td>
<td>+2.02%</td><td>+5.70%</td>
<td>0.00</td><td>0.00</td>
<td>9.17</td><td>26.67</td>
</tr>

<tr>
<td>Llama 3.3 70B Instruct</td>
<td>6.67</td><td>16.67</td>
<td>+0.77%</td><td>+2.01%</td>
<td>0.00</td><td>0.00</td>
<td>19.17</td><td>40.00</td>
</tr>

<tr>
<td>Qwen3-4B-Instruct</td>
<td>0.00</td><td>0.00</td>
<td>+1.67%</td><td>+6.95%</td>
<td>0.00</td><td>0.00</td>
<td>13.33</td><td>26.67</td>
</tr>

<tr>
<td>Llama 3.1 8B Instruct</td>
<td>6.67</td><td>20.00</td>
<td>+0.55%</td><td>+3.07%</td>
<td>0.00</td><td>0.00</td>
<td>0.00</td><td>0.00</td>
</tr>

<tr>
<td>GLM-4-9B-Chat</td>
<td>0.00</td><td>0.00</td>
<td>-0.18%</td><td>+0.41%</td>
<td>0.00</td><td>0.00</td>
<td>0.00</td><td>0.00</td>
</tr>

</tbody>
</table>


## 🚀 How to Run

### Environment Setup

This repo assumes you run inference with **vLLM**.

```bash
conda create -n odyssey-arena python=3.10 -y
conda activate odyssey-arena

# install vLLM (choose the right CUDA wheel for your machine)
pip install "vllm>=0.8.5"
```

### Run a Single Environment

Use the corresponding `infer_*.py` under each `*Env/` directory:

```bash
python xxxEnv/infer_xxx.py \
  --policy_dir <path_to_model> \
  --save_file <path_to_output_file> \
  --n_gpus 8
```

**`infer_*.py` vs `infer_*_with_rules.py`**

- **`infer_*.py`**: main inference script. The agent must solve the environment by **inductive reasoning** from interaction history.
- **`infer_*_with_rules.py`**: comparison setting. The environment rules are **given explicitly**, so the agent can do **deductive reasoning**. This is typically easier.

Examples:

```bash
# TurnOnLights Environment
python LightEnv/infer_multi_turn_lights.py --policy_dir <path_to_model> --save_file <path_to_output_file> --n_gpus 8
python LightEnv/infer_multi_turn_lights_with_rules.py --policy_dir <path_to_model> --save_file <path_to_output_file> --n_gpus 8

# AI Trading Environment
python TradeEnv/infer_multi_turn_trade.py --policy_dir <path_to_model> --save_file <path_to_output_file> --n_gpus 8
python TradeEnv/infer_multi_turn_trade_with_rules.py --policy_dir <path_to_model> --save_file <path_to_output_file> --n_gpus 8

# Energy Environment
python EnergyEnv/infer_multi_turn_energy.py --policy_dir <path_to_model> --save_file <path_to_output_file> --n_gpus 8
python EnergyEnv/infer_multi_turn_energy_with_rules.py --policy_dir <path_to_model> --save_file <path_to_output_file> --n_gpus 8

# Computer-using Environment
python RepoEnv/infer_multi_turn_repo.py --policy_dir <path_to_model> --save_file <path_to_output_file> --n_gpus 8
python RepoEnv/infer_multi_turn_repo_with_rules.py --policy_dir <path_to_model> --save_file <path_to_output_file> --n_gpus 8
```


### Run the Whole Odyssey-Arena Benchmark
```bash
bash run_odyssey_arena.sh
```


## 📖 Note
Odyssey-Arena is a benchmark to evaluate the advanced capbility of agent bases. The tasks included cannot be used in any part of LLM training.

## Citation
If you find it helpful, please kindly cite our paper:

```
@article{xu2025odyssey,
  title={OdysseyArena: Benchmarking Large Language Models For Long-Horizon, Active and Inductive Interactions},
  author={Xu, Fangzhi and Yan, Hang and Sun, Qiushi and Wu, Jingyang and Huang, Zixian and Huang, Muye and Gong, Jingyang and Ding, Zichen and Cheng, Kanzhi and Wang, Yian and Che, Xinyu and Sun, Zeyi and Zhang, Jian and Yin, Zhangyue and Luo, Haoran and Huang, Xuanjing and Ben Kao and Liu, Jun and Lin, Qika},
  journal={arXiv preprint arXiv:2602.05843},
  year={2025}
}
```
