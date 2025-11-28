#!/bin/bash
# Helper script to run reward profiling experiments

# Default settings (can be overridden by environment variables)
ALGO=${ALGO:-ppo}
ENV_ID=${ENV_ID:-CartPole-v1}
VARIANT=${VARIANT:-three-points}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-100000}
EPISODES_BUDGET=${EPISODES_BUDGET:-10}
SEED=${SEED:-0}

echo "Running reward profiling experiment:"
echo "  Algorithm: $ALGO"
echo "  Environment: $ENV_ID"
echo "  Variant: $VARIANT"
echo "  Timesteps: $TOTAL_TIMESTEPS"
echo "  Episodes: $EPISODES_BUDGET"
echo "  Seed: $SEED"
echo ""

python profiling_rl.py \
  --algo $ALGO \
  --env $ENV_ID \
  --variant $VARIANT \
  --total-timesteps $TOTAL_TIMESTEPS \
  --episodes-budget $EPISODES_BUDGET \
  --seed $SEED \
  "$@"
