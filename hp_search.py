import argparse
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from src.agent import DQNAgent
from src.evaluate import evaluate_agent, render_episode
from src.network import QNetwork
from src.replay_buffer import ReplayBuffer
from src.train import train

STATE_DIM = 2
ACTION_DIM = 3
EVAL_EPISODES = 20


class Hyperparameters(BaseModel):
    name: str = Field(min_length=1)
    seed: int = Field(default=42)
    gamma: float = Field(gt=0.0, le=1.0)
    learning_rate: float = Field(gt=0.0)
    batch_size: int = Field(ge=1)
    buffer_capacity: int = Field(ge=1)
    epsilon_start: float = Field(ge=0.0, le=1.0)
    epsilon_end: float = Field(ge=0.0, le=1.0)
    epsilon_decay: float = Field(gt=0.0, le=1.0)
    target_update_freq: int = Field(ge=1)
    episodes: int = Field(default=300, ge=1)


class HPSearchConfig(BaseModel):
    results_root: str = "results/hp"
    base: Hyperparameters | None = None
    experiments: list[Hyperparameters]


class EvalResults(BaseModel):
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    successes: int
    episodes: int


def build_agent(hp: Hyperparameters) -> DQNAgent:
    online_net = QNetwork(STATE_DIM, ACTION_DIM)
    target_net = QNetwork(STATE_DIM, ACTION_DIM)
    replay_buffer = ReplayBuffer(capacity=hp.buffer_capacity)
    return DQNAgent(
        online_net=online_net,
        target_net=target_net,
        replay_buffer=replay_buffer,
        gamma=hp.gamma,
        learning_rate=hp.learning_rate,
        batch_size=hp.batch_size,
        epsilon_start=hp.epsilon_start,
        epsilon_end=hp.epsilon_end,
        epsilon_decay=hp.epsilon_decay,
        target_update_freq=hp.target_update_freq,
    )


def run_single(hp: Hyperparameters, results_root: Path):
    out_dir = results_root / hp.name

    if (out_dir / "model.pt").exists():
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump(hp.model_dump(), f, indent=2)

    agent = build_agent(hp)
    train(agent=agent, episodes=hp.episodes, seed=hp.seed, results_dir=out_dir)

    agent.load(out_dir / "model.pt")
    metrics = evaluate_agent(agent, episodes=EVAL_EPISODES, seed=hp.seed + 1000)

    metrics = EvalResults.model_validate(metrics)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics.model_dump(), f, indent=2)

    render_episode(agent, seed=hp.seed + 2000, output_dir=out_dir / "videos")

    return metrics


def run_sweep(yaml_path: Path) -> None:
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
        cfg = HPSearchConfig.model_validate(raw)

    root = Path(cfg.results_root)
    root.mkdir(parents=True, exist_ok=True)

    for i, hp in enumerate(cfg.experiments, start=1):
        print(f"\n=== [{i}/{len(cfg.experiments)}] {hp.name} ===")
        try:
            metrics = run_single(hp, root)
            print(f"\t{metrics}")
        except Exception as e:
            print(f"\tERROR!!! failed: {e}")


def main():
    parser = argparse.ArgumentParser()
    _ = parser.add_argument(
        "--config-path",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    config_path = Path(args.config_path)

    run_sweep(config_path)


if __name__ == "__main__":
    main()
