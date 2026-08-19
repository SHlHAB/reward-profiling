# Stabilizing Policy Gradient Methods via Reward Profiling

This repository provides the minimal implementation accompanying the paper:

[**Stabilizing Policy Gradient Methods via Reward Profiling**  ](https://arxiv.org/abs/2511.16629) (AAAI'26)

## Installation
```bash
git clone https://github.com/SHlHAB/reward-profiling.git
cd reward-profiling
pip install -r requirements.txt
```

## Usage

### Basic Example
```bash
# Run PPO with three-points profiling on CartPole
python profiling_rl.py --algo ppo --env CartPole-v1 --variant three-points

# PPO on HalfCheetah with lookback profiling
python profiling_rl.py --algo ppo --env HalfCheetahBulletEnv-v0 --variant lookback \
  --total-timesteps 500000 --episodes-budget 50 --seed 42

# TRPO on Hopper with mixup profiling and WandB logging
python profiling_rl.py --algo trpo --env HopperBulletEnv-v0 --variant mixup \
  --total-timesteps 1000000 --episodes-budget 100 \
  --wandb-project reward-profiling --seed 0

# TD3 on Ant with vanilla baseline
python profiling_rl.py --algo td3 --env AntBulletEnv-v0 --variant vanilla \
  --total-timesteps 1000000 --eval-episodes 10
```

### Available Options
- `--algo`: Algorithm choice (`ppo`, `trpo`, `ddpg`, `td3`)
- `--env`: Gymnasium environment ID (e.g., `CartPole-v1`, `HalfCheetahBulletEnv-v0`)
- `--variant`: Profiling variant (`vanilla`, `lookback`, `mixup`, `three-points`)
- `--total-timesteps`: Total training timesteps (default: 100000)
- `--episodes-budget`: Number of profiling episodes (default: 10)
- `--eval-episodes`: Episodes per evaluation (default: 5)
- `--alpha`, `--beta`: Beta distribution parameters for λ sampling (default: 2.0, 2.0)
- `--clip-min`, `--clip-max`: Clipping bounds for λ (default: 0.75, 0.90)
- `--seed`: Random seed (default: 0)
- `--log-dir`: Directory for logs and models (default: `logs`)
- `--wandb-project`: WandB project name (optional)

### Using the Helper Script
```bash
# Run with default settings
./run.sh

# Or customize the environment
ENV_ID=HopperBulletEnv-v0 ./run.sh
```

## License
MIT License.

## Citation

If you use this work, please cite:

Ahmed, S., Bergou, E. H., Wang, Y., & Dutta, A. (2026). **Stabilizing Policy Gradient Methods via Reward Profiling.** *Proceedings of the AAAI Conference on Artificial Intelligence, 40*(24), 19560–19568. https://doi.org/10.1609/aaai.v40i24.39035

```bibtex
@inproceedings{ahmed2026stabilizing,
  title={Stabilizing Policy Gradient Methods via Reward Profiling},
  author={Ahmed, Shihab and Bergou, El Houcine and Wang, Yue and Dutta, Aritra},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={24},
  pages={19560--19568},
  year={2026},
  doi={10.1609/aaai.v40i24.39035}
}
