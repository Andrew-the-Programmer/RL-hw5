import argparse
from pathlib import Path

from src.agent import DQNAgent
from src.evaluate import evaluate_agent, render_episode
from src.network import QNetwork
from src.replay_buffer import ReplayBuffer
from src.train import train


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "render"],
        default="train",
    )
    return parser


def make_agent():
    state_dim = 2
    action_dim = 3

    online_net = QNetwork(state_dim, action_dim)
    target_net = QNetwork(state_dim, action_dim)

    replay_buffer = ReplayBuffer(capacity=1_000)

    agent = DQNAgent(
        online_net=online_net,
        target_net=target_net,
        replay_buffer=replay_buffer,
        gamma=0.90,
        learning_rate=1e-2,
        batch_size=128,
        epsilon_start=0.10,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        target_update_freq=1,
    )
    return agent


def main():
    parser = build_parser()
    args = parser.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    agent = make_agent()

    if args.mode == "train":
        train(
            agent=agent,
            episodes=300,
            seed=42,
            results_dir=results_dir,
        )
    elif args.mode == "evaluate":
        model_path = results_dir / "model.pt"
        agent.load(model_path)
        metrics = evaluate_agent(agent, episodes=20, seed=123)
        print(metrics)
    elif args.mode == "render":
        model_path = results_dir / "model.pt"
        agent.load(model_path)
        render_episode(agent, seed=123, output_dir=results_dir / "videos")


if __name__ == "__main__":
    main()
