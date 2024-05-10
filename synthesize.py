import argparse
import glob
import os
import re

import numpy as np
import open3d as o3d
from PIL import Image
from tqdm import tqdm


def read_motion_mesh(path, trans_mat):
    mesh = o3d.io.read_triangle_mesh(path)
    mesh = mesh.transform(trans_mat)
    return mesh


def render_scene(
    scene,
    dataset_dir,
    motion_mesh_dir,
    output_dir,
    motion_loop="infinite-reverse",
    num_samples=None,
    scene_suffix="",
    save_mask=False,
):

    if num_samples is None:
        num_samples = [0]

    info = {}
    with open(os.path.join(dataset_dir, scene, f"{scene}.txt"), "r") as f:
        for line in f.readlines():
            k, v = line.split("=")
            info[k.strip()] = v.strip()

    width = int(info["depthWidth"])
    height = int(info["depthHeight"])
    fx = float(info["fx_depth"])
    fy = float(info["fy_depth"])
    cx = float(info["mx_depth"])
    cy = float(info["my_depth"])

    poses = glob.glob(os.path.join(dataset_dir, scene, "pose/*.txt"))
    poses = sorted(
        poses, key=lambda s: int(re.search(r"\d+", os.path.basename(s)).group())
    )

    meshes = {}
    for i, p in enumerate(
        sorted(glob.glob(os.path.join(motion_mesh_dir, scene, "mesh*")))
    ):
        if not os.path.isdir(p):
            continue

        meshes[i] = sorted(
            glob.glob(os.path.join(p, "*.obj")),
            key=lambda s: int(re.search(r"\d+", os.path.basename(s)).group()),
        )

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height)

    scene_mesh = o3d.io.read_triangle_mesh(
        os.path.join(dataset_dir, scene, f"{scene}{scene_suffix}.ply")
    )
    vis.add_geometry(scene_mesh)

    verts = np.asarray(scene_mesh.vertices)
    center = (verts.max(axis=0) + verts.min(axis=0)) * 0.5
    center[2] = np.percentile(verts[:, 2], 1)
    trans_mat = np.eye(4)
    trans_mat[0:3, -1] += center

    view_control = vis.get_view_control()

    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    for n in num_samples:
        color_dir = os.path.join(
            output_dir,
            scene,
            f"{n}",
            "color",
        )
        depth_dir = os.path.join(output_dir, scene, f"{n}", "depth")
        mask_dir = os.path.join(output_dir, scene, f"{n}", "mask")

        for d in [color_dir, depth_dir]:
            os.makedirs(d, exist_ok=True)

        if save_mask and n > 0:
            os.makedirs(mask_dir, exist_ok=True)

        if n > 0:
            ids = {i: meshes[i] for i in np.random.choice(list(meshes.keys()), n)}
        else:
            ids = {}

        motion_meshes = {i: None for i in ids}
        for i, pose in enumerate(tqdm(poses, leave=False)):
            basename = os.path.splitext(os.path.basename(pose))[0]
            extrinsic = np.linalg.inv(np.loadtxt(pose))

            if np.any(np.isnan(extrinsic)):
                continue

            for j, m in ids.items():
                if motion_meshes[j] is not None:
                    vis.remove_geometry(motion_meshes[j])
                    motion_meshes[j] = None

                if motion_loop == "once":
                    if 0 <= i < len(meshes[j]):
                        idx = i
                        motion_meshes[j] = read_motion_mesh(meshes[j][idx], trans_mat)
                        vis.add_geometry(motion_meshes[j])
                        vis.update_geometry(motion_meshes[j])
                elif motion_loop == "infinite":
                    idx = i % len(meshes[j])
                    motion_meshes[j] = read_motion_mesh(meshes[j][idx], trans_mat)
                    vis.add_geometry(motion_meshes[j])
                    vis.update_geometry(motion_meshes[j])
                elif motion_loop == "infinite-reverse":
                    idx = i
                    if (idx // len(meshes)) % 2 == 0:
                        idx = idx % len(meshes[j])
                    else:
                        idx = len(meshes[j]) - idx % len(meshes[j]) - 1

                    motion_meshes[j] = read_motion_mesh(meshes[j][idx], trans_mat)
                    vis.add_geometry(motion_meshes[j])
                    vis.update_geometry(motion_meshes[j])
                else:
                    raise ValueError

            pinhole_parameters = view_control.convert_to_pinhole_camera_parameters()
            pinhole_parameters.intrinsic = intrinsic
            pinhole_parameters.extrinsic = extrinsic
            view_control.convert_from_pinhole_camera_parameters(pinhole_parameters)

            vis.poll_events()
            vis.update_renderer()

            vis.capture_screen_image(os.path.join(color_dir, f"{basename}.jpg"))
            vis.capture_depth_image(os.path.join(depth_dir, f"{basename}.png"))

            if save_mask and n > 0:
                dynamic = np.asarray(vis.capture_depth_float_buffer())
                for j, m in ids.items():
                    if motion_meshes[j] is not None:
                        vis.remove_geometry(motion_meshes[j])

                pinhole_parameters = view_control.convert_to_pinhole_camera_parameters()
                pinhole_parameters.intrinsic = intrinsic
                pinhole_parameters.extrinsic = extrinsic
                view_control.convert_from_pinhole_camera_parameters(pinhole_parameters)

                vis.poll_events()
                vis.update_renderer()

                static = np.asarray(vis.capture_depth_float_buffer())

                mask = np.isclose(static, dynamic)
                mask[np.isnan(static)] = 1
                mask = ~mask

                Image.fromarray(mask).save(os.path.join(mask_dir, f"{basename}.png"))

                for j, m in ids.items():
                    if motion_meshes[j] is not None:
                        vis.remove_geometry(motion_meshes[j])

    vis.destroy_window()
    del view_control
    del vis


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--scannet_root", default="/datasets/ScanNet/scans")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--scene_list", default=None)
    parser.add_argument("--scene_suffix", default="_vh_clean_2")
    parser.add_argument("--motion_mesh_dir", default="/datasets/Dyna3DBench/motions")
    parser.add_argument("--num_samples", default=[0, 1, 2], nargs="+", type=int)
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--motion_fps", type=int, default=30)
    parser.add_argument(
        "--motion_loop",
        default="infinite-reverse",
        choices=["infinite", "once", "infinite-reverse"],
    )
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--save_mask", action="store_true")
    parser.add_argument("--seed", type=int, default=46)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    np.random.seed(args.seed)

    if args.scene is not None:
        scenes = [args.scene]
    elif args.scene_list is not None:
        scenes = [line.strip() for line in open(args.scene_list, "r").readlines()]
    else:
        scenes = [
            os.path.basename(s) for s in glob.glob(os.path.join(args.scannet_root, "*"))
        ]

    for scene in tqdm(scenes):
        render_scene(
            scene,
            args.scannet_root,
            args.motion_mesh_dir,
            args.output_dir,
            num_samples=args.num_samples,
            motion_loop=args.motion_loop,
            scene_suffix=args.scene_suffix,
            save_mask=args.save_mask,
        )


if __name__ == "__main__":
    main()
