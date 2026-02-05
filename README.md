<h1 align="center">
🏁 OdysseyArena: Benchmarking Large Language Models For Long-Horizon, Active and Inductive Interactions
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2507.14958"><b>[📜 Paper]</b></a> •
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
  title={Odyssey-Arena: xxx},
  author={Xu, Fangzhi},
  journal={arXiv preprint arXiv:2601.xxxxx},
  year={2025}
}
```
