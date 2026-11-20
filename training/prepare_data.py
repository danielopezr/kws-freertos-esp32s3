"""Download Speech Commands v2 and build the MFCC feature cache.

    python prepare_data.py --data-dir data/speech_commands --cache-dir data/features
"""

import argparse

from kws import dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/speech_commands")
    ap.add_argument("--cache-dir", default="data/features")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--augment-copies", type=int, default=1,
                    help="augmented copies of the training split added to the clean copy")
    args = ap.parse_args()

    dataset.download(args.data_dir)
    meta = dataset.build_features(args.data_dir, args.cache_dir, args.seed, args.augment_copies)
    for split, counts in meta["counts"].items():
        print(f"{split}: {sum(counts.values())} samples")


if __name__ == "__main__":
    main()
