import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from hp_search import (
    Hyperparameters,
    run_render,
    run_train,
    run_val,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_parser():
    parser = argparse.ArgumentParser()
    _ = parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "render"],
        default="train",
    )
    _ = parser.add_argument("--results-dir", type=Path, default=Path("results"))
    _ = parser.add_argument(
        "--config-path", type=Path, default=Path("results/config.json")
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    config_path = Path(args.config_path)
    mode = str(args.mode)

    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
        config = Hyperparameters.model_validate(raw)

    set_seed(config.seed)

    if mode == "train":
        run_train(results_dir, config)
    elif mode == "evaluate":
        metrics = run_val(results_dir, config)
        print(json.dumps(metrics.model_dump(), indent=2))
    elif mode == "render":
        run_render(results_dir, config)


if __name__ == "__main__":
    main()
