"""
One-time ingestion entrypoint.

Downloads/generates public + synthetic datasets and persists all tables.
Subsequent runs of the app load from DB instead of regenerating.
"""

from data.generator import build_all_datasets


def main() -> None:
    datasets = build_all_datasets(force=True)
    print("Ingestion complete:")
    for key, df in datasets.items():
        print(f" - {key}: {len(df)} rows")


if __name__ == "__main__":
    main()
