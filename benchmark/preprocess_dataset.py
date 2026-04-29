from data.benchmarks import preprocess_and_cache_dataset
from experiment.arg_parsers import parse_preprocess_args


def main(dataset_name: str):
    # === Data preprocessing ==================================================
    preprocess_and_cache_dataset(dataset_name)


if __name__ == "__main__":
    args = parse_preprocess_args()
    main(args.dataset_name)
