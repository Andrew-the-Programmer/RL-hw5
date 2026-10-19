import argparse
import json
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from hp_search import EvalResults, Hyperparameters


class Results(BaseModel):
    hp: Hyperparameters
    m: EvalResults


def get_results(root: Path) -> list[Results]:
    results: list[Results] = []

    for d in root.iterdir():
        hp_path, m_path = d / "hyperparameters.json", d / "metrics.json"

        if not (hp_path.exists() and m_path.exists()):
            continue

        hp_dict = json.loads(hp_path.read_text())
        hp = Hyperparameters.model_validate(hp_dict)
        m_dict = json.loads(m_path.read_text())
        m = EvalResults.model_validate(m_dict)

        results.append(Results(hp=hp, m=m))

    return results


def print_results(results: list[Results]) -> None:
    results.sort(
        key=lambda r: (r.m.successes, r.m.mean_reward),
        reverse=True,
    )

    base_mean_reward: float | None = None
    for r in results:
        if r.hp.name == "base":
            base_mean_reward = r.m.mean_reward

    rows: list[dict[str, str | float]] = []

    for r in results:
        m = r.m
        hp = r.hp
        delta = (
            f"{m.mean_reward - base_mean_reward:+.2f}"
            if base_mean_reward is not None
            else "-"
        )
        rows.append(
            {
                "name": hp.name,
                "succ": f"{m.successes}/{m.episodes}",
                "delta": delta,
                "mean": m.mean_reward,
                "std": m.std_reward,
                "min_reward": m.min_reward,
                "max_reward": m.max_reward,
            }
        )

    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format="%.2f"))

    best_result = results[0]
    print(f"\nBest: {best_result.hp.name}")
    print(json.dumps(best_result.hp.model_dump(), indent=2))
    print(json.dumps(best_result.m.model_dump(), indent=2))


def main():
    parser = argparse.ArgumentParser()
    _ = parser.add_argument(
        "--root",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    root = Path(args.root)

    results = get_results(root)
    print_results(results)


if __name__ == "__main__":
    main()
