# profiling_rl.py

import os
import argparse
import numpy as np
import torch
import gymnasium as gym
import pybullet_envs_gymnasium  # registers Bullet envs
import wandb

from stable_baselines3 import PPO, DDPG, TD3
from sb3_contrib import TRPO


def evaluate_policy(model, env, n_eval_episodes=5):
    """Run deterministic rollouts and return mean/std of cumulative rewards."""
    returns = []
    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        total_r = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            total_r += r
            done = term or trunc
        returns.append(total_r)
    return np.mean(returns), np.std(returns)


def sample_lambda(alpha, beta, clip_min, clip_max):
    """Sample λ ∼ Beta(α,β) then clip into [clip_min, clip_max]."""
    lam = np.random.beta(alpha, beta)
    return float(np.clip(lam, clip_min, clip_max))


def general_profiling(
    algo_name: str,
    env_id: str,
    variant: str,
    total_timesteps: int,
    episodes_budget: int,
    eval_episodes: int,
    alpha: float,
    beta: float,
    clip_min: float,
    clip_max: float,
    seed: int,
    log_dir: str,
    wandb_project: str = None,
):
    # reproducibility
    os.makedirs(log_dir, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # optionally init WandB
    run = None
    if wandb_project:
        run = wandb.init(
            project=wandb_project,
            name=f"{env_id.split('-')[0]}-{algo_name}-{variant}-seed{seed}",
            config=dict(
                algo=algo_name,
                env=env_id,
                variant=variant,
                total_timesteps=total_timesteps,
                episodes_budget=episodes_budget,
                eval_episodes=eval_episodes,
                alpha=alpha,
                beta=beta,
                clip_min=clip_min,
                clip_max=clip_max,
                seed=seed,
            ),
            dir=log_dir,
            reinit=True,
        )

    # create environment and model
    env = gym.make(env_id)
    algo = algo_name.lower()
    if algo == "ppo":
        model = PPO("MlpPolicy", env, seed=seed, verbose=0)
    elif algo == "trpo":
        model = TRPO("MlpPolicy", env, seed=seed, verbose=0)
    elif algo == "ddpg":
        model = DDPG("MlpPolicy", env, seed=seed, verbose=0)
    elif algo == "td3":
        model = TD3("MlpPolicy", env, seed=seed, verbose=0)
    else:
        raise ValueError(f"Unsupported algorithm: {algo_name}")

    # bookkeeping
    t_per_iter = total_timesteps // episodes_budget
    history = {
        "episode": [],
        "r_old": [],
        "r_new": [],
        "r_mix": [],
        "r_sel": [],
        "policy_sel": [],
        "lambda": [],
    }

    # main profiling loop
    for ep in range(1, episodes_budget + 1):
        # backup old parameters
        old_state = model.policy.state_dict()
        r_old, _ = evaluate_policy(model, env, eval_episodes)

        # one chunk of learning
        model.learn(total_timesteps=t_per_iter, reset_num_timesteps=False)
        new_state = model.policy.state_dict()
        r_new, _ = evaluate_policy(model, env, eval_episodes)

        # mix
        lam = sample_lambda(alpha, beta, clip_min, clip_max)
        mixed_state = {
            k: lam * new_state[k] + (1 - lam) * old_state[k]
            for k in old_state
        }
        model.policy.load_state_dict(mixed_state)
        r_mix, _ = evaluate_policy(model, env, eval_episodes)

        # select
        if variant.lower() == "lookback":
            r_sel, choice = (r_old, "old") if r_old >= r_new else (r_new, "new")
            state_sel = old_state if choice == "old" else new_state
        elif variant.lower() == "mixup":
            r_sel, choice = (r_old, "old") if r_old >= r_mix else (r_mix, "mix")
            state_sel = old_state if choice == "old" else mixed_state
        else:  # three-points
            vals = [r_old, r_new, r_mix]
            idx = int(np.argmax(vals))
            choice = ["old", "new", "mix"][idx]
            r_sel = vals[idx]
            state_sel = [old_state, new_state, mixed_state][idx]

        # restore selected
        model.policy.load_state_dict(state_sel)

        # record
        history["episode"].append(ep)
        history["r_old"].append(r_old)
        history["r_new"].append(r_new)
        history["r_mix"].append(r_mix)
        history["r_sel"].append(r_sel)
        history["policy_sel"].append(choice)
        history["lambda"].append(lam)

        # log to WandB
        if run:
            wandb.log(
                {
                    "episode": ep,
                    "reward/old": r_old,
                    "reward/new": r_new,
                    "reward/mix": r_mix,
                    "reward/selected": r_sel,
                    "lambda": lam,
                }
            )

    # save final model
    save_path = os.path.join(log_dir, f"{algo_name}_{env_id}_{variant}_seed{seed}")
    model.save(save_path)

    if run:
        run.finish()
    env.close()
    return history


def parse_args():
    p = argparse.ArgumentParser(description="Run reward profiling experiments")
    p.add_argument("--algo", required=True, choices=["ppo", "trpo", "ddpg", "td3"])
    p.add_argument("--env", required=True)
    p.add_argument("--variant", required=True, choices=["vanilla", "lookback", "mixup", "three-points"])
    p.add_argument("--total-timesteps", type=int, default=100_000)
    p.add_argument("--episodes-budget", type=int, default=10)
    p.add_argument("--eval-episodes",   type=int, default=5)
    p.add_argument("--alpha",   type=float, default=2.0)
    p.add_argument("--beta",    type=float, default=2.0)
    p.add_argument("--clip-min", type=float, default=0.75)
    p.add_argument("--clip-max", type=float, default=0.90)
    p.add_argument("--seed",    type=int, default=0)
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--wandb-project", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    hist = general_profiling(
        algo_name=args.algo,
        env_id=args.env,
        variant=args.variant,
        total_timesteps=args.total_timesteps,
        episodes_budget=args.episodes_budget,
        eval_episodes=args.eval_episodes,
        alpha=args.alpha,
        beta=args.beta,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
        seed=args.seed,
        log_dir=args.log_dir,
        wandb_project=args.wandb_project,
    )
    # optionally: save history as CSV
    out_csv = os.path.join(args.log_dir, f"history_{args.algo}_{args.env}_{args.variant}_seed{args.seed}.csv")
    import pandas as pd
    pd.DataFrame(hist).to_csv(out_csv, index=False)
    print(f"Done; history saved to {out_csv}")


if __name__ == "__main__":
    main()
