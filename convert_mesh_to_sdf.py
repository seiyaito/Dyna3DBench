import argparse
import json
import os

import numpy as np
import trimesh
from mesh_to_sdf import mesh_to_voxels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_mesh")
    parser.add_argument("-d", "--dim", default=256, type=int)
    parser.add_argument("-o", "--output_dir", default="./outputs")
    args = parser.parse_args()

    mesh = trimesh.load(args.input_mesh)

    os.makedirs(args.output_dir, exist_ok=True)

    basename = os.path.splitext(os.path.basename(args.input_mesh))[0]

    with open(os.path.join(args.output_dir, basename + ".json"), "w") as f:
        json.dump(
            {
                "dim": args.dim,
                "min": mesh.bounds[0].tolist(),
                "max": mesh.bounds[1].tolist(),
            },
            f,
        )

    sdf = mesh_to_voxels(mesh, args.dim)
    np.save(os.path.join(args.output_dir, basename + ".npy"), sdf.flatten())


if __name__ == "__main__":
    main()
