import argparse
import glob
import json
import os
import random

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="/datasets/ScanNet")
    parser.add_argument(
        "--list", default="/workspace/ScanNet/Tasks/Benchmark/scannetv2_val.txt"
    )
    parser.add_argument("--output_dir", default="/workspace/data/cam2world")
    parser.add_argument("--seed", default=46, type=int)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    with open(args.list, "r") as f:
        scenes = [l.strip() for l in f.readlines()]

    for scene in scenes:
        poses = glob.glob(
            os.path.join(args.dataset_root, "scans", scene, "pose", "*.txt")
        )
        assert len(poses) > 0
        cam2world = None
        np.random.shuffle(poses)
        for p in poses:
            mat = np.loadtxt(p)
            cam2world = np.asarray(mat, dtype=np.float32).reshape((4, 4))
            if not np.any(np.isinf(cam2world)) and not np.any(np.isnan(cam2world)):
                break
            cam2world = None
        assert cam2world is not None

        with open(os.path.join(args.output_dir, f"{scene}.json"), "w") as f:
            json.dump(cam2world.tolist(), f)


if __name__ == "__main__":
    main()
