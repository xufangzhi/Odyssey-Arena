# Odyssey-Arena
Extremely Long-Horizon Agentic Tasks Requiring Active Acting and Inductive Reasoning


## How to Run

### 🧰 Environment Setup

This repo assumes you run inference with **vLLM**.

```bash
conda create -n odyssey-arena python=3.10 -y
conda activate odyssey-arena

# install vLLM (choose the right CUDA wheel for your machine)
pip install "vllm>=0.8.5"
```

### 🚀 Run a Single Environment

Use the corresponding `infer_*.py` under each `*Env/` directory:

```bash
python xxxEnv/infer_xxx.py \
  --policy_dir <path_to_model> \
  --save_file <path_to_output_file> \
  --n_gpus 8
```

### 🧪 `infer_*.py` vs `infer_*_with_rules.py`

- **`infer_*.py`**: main inference script. The agent must solve the environment by **inductive reasoning** from interaction history.
- **`infer_*_with_rules.py`**: comparison setting. The environment rules are **given explicitly**, so the agent can do **deductive reasoning**. This is typically easier.

Examples:

```bash
python EnergyEnv/infer_multi_turn_energy.py --policy_dir <path_to_model> --save_file <path_to_output_file> --n_gpus 8
python EnergyEnv/infer_multi_turn_energy_with_rules.py --policy_dir <path_to_model> --save_file <path_to_output_file> --n_gpus 8

python LightEnv/infer_multi_turn_lights.py --policy_dir <path_to_model> --save_file <path_to_output_file> --n_gpus 8
python LightEnv/infer_multi_turn_lights_with_rules.py --policy_dir <path_to_model> --save_file <path_to_output_file> --n_gpus 8

python TradeEnv/infer_multi_turn_trade.py --policy_dir <path_to_model> --save_file <path_to_output_file> --n_gpus 8
python TradeEnv/infer_multi_turn_trade_with_rules.py --policy_dir <path_to_model> --save_file <path_to_output_file> --n_gpus 8

python RepoEnv/infer_multi_turn_repo.py --policy_dir <path_to_model> --save_file <path_to_output_file> --n_gpus 8
python RepoEnv/infer_multi_turn_repo_with_rules.py --policy_dir <path_to_model> --save_file <path_to_output_file> --n_gpus 8
```


### 🏁 Run the Whole Odyssey-Arena Benchmark
```bash
bash run_odyssey_arena.sh
```


## 📖 Note
Odyssey-Arena is a benchmark to evaluate the advanced capbility of agent bases. The tasks included cannot be used in any part of LLM training.

## Citation
If you find it helpful, please kindly cite our paper as well as the inference-time decoding algorithm $\phi$-Decoding:

```
@article{xu2025odyssey,
  title={Odyssey-Arena: xxx},
  author={Xu, Fangzhi},
  journal={arXiv preprint arXiv:2601.xxxxx},
  year={2025}
}
```