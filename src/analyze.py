from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = Path("data/Grad Program Exit Survey Data (2).xlsx")
OUTPUT_DIR = Path("outputs")
OUTPUT_CSV = OUTPUT_DIR / "course_ranking.csv"
OUTPUT_PNG = OUTPUT_DIR / "course_ranking.png"


def normalize_finished(series: pd.Series) -> pd.Series:
    truthy = {"true", "1", "yes", "y", "t"}
    return series.astype(str).str.strip().str.lower().isin(truthy)


def find_group_column(columns: List[str], keyword: str) -> str:
    matches = [c for c in columns if "Groups" in c and keyword in c]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one column containing 'Groups' and '{keyword}', found {len(matches)}: {matches}"
        )
    return matches[0]


def split_courses(value: object) -> List[str]:
    if pd.isna(value):
        return []
    parts = [p.strip() for p in str(value).split(",")]
    return [p for p in parts if p]


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_excel(DATA_PATH, sheet_name=0)

    if "Finished" in df.columns:
        df = df[normalize_finished(df["Finished"])]

    df = df.reset_index(drop=True)
    n_responses = len(df)
    if n_responses == 0:
        raise ValueError("No responses available after filtering. Cannot compute ranking.")

    columns = [str(c) for c in df.columns]
    most_col = find_group_column(columns, "Most Beneficial")
    neutral_col = find_group_column(columns, "Neutral")
    least_col = find_group_column(columns, "Least Beneficial")

    course_stats: Dict[str, Dict[str, int]] = {}

    for _, row in df.iterrows():
        buckets = {
            "most": split_courses(row[most_col]),
            "neutral": split_courses(row[neutral_col]),
            "least": split_courses(row[least_col]),
        }

        for bucket_name, courses in buckets.items():
            for course in courses:
                if course not in course_stats:
                    course_stats[course] = {"most": 0, "neutral": 0, "least": 0}
                course_stats[course][bucket_name] += 1

    if not course_stats:
        raise ValueError("No course names found in ranking columns.")

    ranking_rows = []
    for course, counts in course_stats.items():
        score = (2 * counts["most"] + 1 * counts["neutral"] + 0 * counts["least"]) / n_responses
        ranking_rows.append(
            {
                "course": course,
                "score": score,
                "most": counts["most"],
                "neutral": counts["neutral"],
                "least": counts["least"],
                "N": n_responses,
            }
        )

    ranking_df = pd.DataFrame(ranking_rows)
    ranking_df = ranking_df.sort_values(
        by=["score", "most", "neutral", "course"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranking_df.insert(0, "rank", ranking_df.index + 1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ranking_df.to_csv(OUTPUT_CSV, index=False)

    top_15 = ranking_df.head(15)
    plt.figure(figsize=(12, 7))
    bars = plt.barh(top_15["course"], top_15["score"], color="#2b6cb0")
    plt.gca().invert_yaxis()
    plt.xlabel("Score = (2*Most + 1*Neutral + 0*Least) / N")
    plt.ylabel("Course")
    plt.title("Top 15 MAcc CORE Courses by Exit Survey Preference Score")

    for bar, score in zip(bars, top_15["score"]):
        plt.text(score + 0.01, bar.get_y() + bar.get_height() / 2, f"{score:.2f}", va="center")

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150)
    plt.close()

    print(f"Included responses (N): {n_responses}")
    print("Top 10 courses:")
    print(ranking_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
