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
TRAIN_SEED_OFFSET = 0
EVAL_SEED_OFFSET = 1000
RENDER_SEED_OFFSET = 2000


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
    train_episodes: int = Field(default=300, ge=1)
    val_episodes: int = Field(default=20, ge=1)


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


def run_train(
    results_dir: Path,
    config: Hyperparameters,
    ask_override: bool = True,
):
    model_pt = results_dir / "model.pt"
    agent = build_agent(config)
    if model_pt.exists() and ask_override:
        i = input("Override? [1-stop],2-load,3-override: ").strip().lower()
        if i == "2":
            agent.load(model_pt)
        elif i == "1" or i != "3":
            return
    train(
        agent=agent,
        episodes=config.train_episodes,
        seed=config.seed + TRAIN_SEED_OFFSET,
        results_dir=results_dir,
    )


def run_val(results_dir: Path, config: Hyperparameters):
    model_pt = results_dir / "model.pt"
    agent = build_agent(config)
    agent.load(model_pt)
    metrics = evaluate_agent(
        agent,
        episodes=config.val_episodes,
        seed=config.seed + EVAL_SEED_OFFSET,
    )
    metrics = EvalResults.model_validate(metrics)
    return metrics


def run_render(results_dir: Path, config: Hyperparameters):
    model_pt = results_dir / "model.pt"
    agent = build_agent(config)
    agent.load(model_pt)
    render_episode(
        agent,
        seed=config.seed + RENDER_SEED_OFFSET,
        output_dir=results_dir / "videos",
    )


def run_single(results_dir: Path, config: Hyperparameters):
    out_dir = results_dir / config.name

    if (out_dir / "model.pt").exists():
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump(config.model_dump(), f, indent=2)

    agent = build_agent(config)

    train(
        agent=agent,
        episodes=config.train_episodes,
        seed=config.seed + TRAIN_SEED_OFFSET,
        results_dir=out_dir,
    )

    agent.load(out_dir / "model.pt")
    metrics = evaluate_agent(
        agent,
        episodes=config.val_episodes,
        seed=config.seed + EVAL_SEED_OFFSET,
    )

    metrics = EvalResults.model_validate(metrics)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics.model_dump(), f, indent=2)

    render_episode(
        agent,
        seed=config.seed + RENDER_SEED_OFFSET,
        output_dir=out_dir / "videos",
    )

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
            metrics = run_single(root, hp)
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
