# Odyssey-Arena
Extremely Long-Horizon Agentic Tasks Requiring Active Acting and Inductive Reasoning


## How to Run

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
  --policy_dir /path/to/your/model \
  --save_file output/run.json \
  --n_gpus 8
```

Examples:

```bash
python EnergyEnv/infer_multi_turn_energy.py --policy_dir /path/to/model --save_file output/energy.json --n_gpus 8
python EnergyEnv/infer_multi_turn_energy_with_rules.py --policy_dir /path/to/model --save_file output/energy_rules.json --n_gpus 8

python LightEnv/infer_multi_turn_lights.py --policy_dir /path/to/model --save_file output/lights.json --n_gpus 8
python LightEnv/infer_multi_turn_lights_with_rules.py --policy_dir /path/to/model --save_file output/lights_rules.json --n_gpus 8

python TradeEnv/infer_multi_turn_trade.py --policy_dir /path/to/model --save_file output/trade.json --n_gpus 8
python TradeEnv/infer_multi_turn_trade_with_rules.py --policy_dir /path/to/model --save_file output/trade_rules.json --n_gpus 8

python RepoEnv/infer_multi_turn_repo.py --policy_dir /path/to/model --save_file output/repo.json --n_gpus 8
python RepoEnv/infer_multi_turn_repo_with_rules.py --policy_dir /path/to/model --save_file output/repo_rules.json --n_gpus 8
```


### Run the Whole Odyssey-Arena Benchmark
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