# MIT License
#
# Copyright (c) 2022 Zan Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Original file: https://github.com/scenediffuser/Scene-Diffuser/blob/main/preprocessing/prox/prox_scene.py
#
# Modified by Seiya Ito on 2023/05/13
# - Refactored code for improved readability and performance.

import argparse
import os
import time

import numpy as np
import trimesh

NUM_MAX_PTS = 100000


def read_ply_xyzrgbnormal(filename):
    """read XYZ RGB normals point cloud from filename PLY file"""
    assert os.path.isfile(filename), "File not found: {}".format(filename)
    scene = trimesh.load(filename)

    pc = scene.vertices
    color = scene.visual.vertex_colors
    normal = scene.vertex_normals

    vertices = np.concatenate((pc, color[:, 0:3], normal), 1)
    return vertices


def collect_one_scene_data_label(scene_dir, scene_id, preprocess_scenes_dir):
    ply_filename = os.path.join(scene_dir, scene_id + ".ply")

    points = read_ply_xyzrgbnormal(ply_filename)

    instance_labels = np.zeros((len(points), 1))
    semantic_labels = np.zeros((len(points), 1))
    data = np.concatenate((points, instance_labels, semantic_labels), 1)

    if data.shape[0] > NUM_MAX_PTS:
        choices = np.random.choice(data.shape[0], NUM_MAX_PTS, replace=False)
        data = data[choices]

    out_filename = os.path.join(preprocess_scenes_dir, scene_id + ".npy")
    np.save(out_filename, data)
    print(
        "Processed {}: shape of subsampled scene data {}".format(scene_id, data.shape)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process PLY files into numpy format with specified directories and scene IDs."
    )
    parser.add_argument(
        "--scene_dir",
        type=str,
        required=True,
        help="Directory where the scene PLY files are stored",
    )
    parser.add_argument(
        "--preprocess_scenes_dir",
        type=str,
        required=True,
        help="Output directory for preprocessed numpy files",
    )
    parser.add_argument(
        "--scene_id", type=str, required=True, help="Scene ID to process"
    )

    args = parser.parse_args()

    os.makedirs(args.preprocess_scenes_dir, exist_ok=True)

    start = time.time()
    collect_one_scene_data_label(
        args.scene_dir, args.scene_id, args.preprocess_scenes_dir
    )
    print("Total processing time: {:.2f}s".format(time.time() - start))
    print("Done!")
