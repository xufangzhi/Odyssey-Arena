#!/usr/bin/env bash

PATH_TO_MODEL="<path_to_model>"
N_GPUS=8

mkdir -p output


# TurnOnLights Environment
python LightEnv/infer_multi_turn_lights.py --policy_dir "${PATH_TO_MODEL}" --save_file output/251219-1.json --n_gpus "${N_GPUS}"
python LightEnv/infer_multi_turn_lights_with_rules.py --policy_dir "${PATH_TO_MODEL}" --save_file output/251219-2.json --n_gpus "${N_GPUS}"

# AI Trading Environment
python TradeEnv/infer_multi_turn_trade.py --policy_dir "${PATH_TO_MODEL}" --save_file output/251219-3.json --n_gpus "${N_GPUS}"
python TradeEnv/infer_multi_turn_trade_with_rules.py --policy_dir "${PATH_TO_MODEL}" --save_file output/251219-4.json --n_gpus "${N_GPUS}"

# Energy Environment
python EnergyEnv/infer_multi_turn_energy.py --policy_dir "${PATH_TO_MODEL}" --save_file output/251219-5.json --n_gpus "${N_GPUS}"
python EnergyEnv/infer_multi_turn_energy_with_rules.py --policy_dir "${PATH_TO_MODEL}" --save_file output/251219-6.json --n_gpus "${N_GPUS}"

# Computer-using Environment
python RepoEnv/infer_multi_turn_repo.py --policy_dir "${PATH_TO_MODEL}" --save_file output/251219-7.json --n_gpus "${N_GPUS}"
python RepoEnv/infer_multi_turn_repo_with_rules.py --policy_dir "${PATH_TO_MODEL}" --save_file output/251219-8.json --n_gpus "${N_GPUS}"

