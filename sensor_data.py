# MIT License
#
# Copyright (c) 2017 Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber,
#                    Thomas Funkhouser, Matthias Niessner
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
# Original file: https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/SensorData.py
#
# Modified by Seiya Ito on 2023/06/19
# - Refactored code for improved readability and performance.
# - Added functionality to dump sensor data.
# - Added functionality to replace sensor data.

import os
import struct
import zlib

import cv2
import imageio.v2 as imageio
import numpy as np
import png
from tqdm import tqdm

COMPRESSION_TYPE_COLOR = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}
COMPRESSION_TYPE_DEPTH = {
    -1: "unknown",
    0: "raw_ushort",
    1: "zlib_ushort",
    2: "occi_ushort",
}


def get_key_from_value(d, val):
    keys = [k for k, v in d.items() if v == val]
    if keys:
        return keys[0]
    return None


class RGBDFrame:
    def __init__(
        self,
        camera_to_world=None,
        timestamp_color=None,
        timestamp_depth=None,
        color_size_bytes=None,
        depth_size_bytes=None,
        color_data=None,
        depth_data=None,
    ):
        self.camera_to_world = camera_to_world
        self.timestamp_color = timestamp_color
        self.timestamp_depth = timestamp_depth
        self.color_size_bytes = color_size_bytes
        self.depth_size_bytes = depth_size_bytes
        self.color_data = color_data
        self.depth_data = depth_data

    def load(self, file_handle):
        self.camera_to_world = np.asarray(
            struct.unpack("f" * 16, file_handle.read(16 * 4)), dtype=np.float32
        ).reshape(4, 4)
        self.timestamp_color = struct.unpack("Q", file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack("Q", file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.color_data = b"".join(
            struct.unpack(
                "c" * self.color_size_bytes, file_handle.read(self.color_size_bytes)
            )
        )
        self.depth_data = b"".join(
            struct.unpack(
                "c" * self.depth_size_bytes, file_handle.read(self.depth_size_bytes)
            )
        )

    # Added by Seiya Ito
    # dump RGB-D data
    def dump(self, file_handle):
        file_handle.write(struct.pack("<16f", *self.camera_to_world.flatten()))
        file_handle.write(struct.pack("<Q", self.timestamp_color))
        file_handle.write(struct.pack("<Q", self.timestamp_depth))
        file_handle.write(struct.pack("<Q", self.color_size_bytes))
        file_handle.write(struct.pack("<Q", self.depth_size_bytes))
        file_handle.write(
            struct.pack("<{}s".format(self.color_size_bytes), self.color_data.tobytes())
        )
        file_handle.write(
            struct.pack("<{}s".format(self.depth_size_bytes), self.depth_data)
        )

    def decompress_depth(self, compression_type):
        if compression_type == "zlib_ushort":
            return self.decompress_depth_zlib()
        else:
            raise

    def decompress_depth_zlib(self):
        return zlib.decompress(self.depth_data)

    def decompress_color(self, compression_type):
        if compression_type == "jpeg":
            return self.decompress_color_jpeg()
        else:
            raise

    def decompress_color_jpeg(self):
        color = cv2.imdecode(self.color_data, cv2.IMREAD_UNCHANGED)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        return color

    # Added by Seiya Ito
    # replace color data
    def replace_color(self, color, compression_type="jpeg"):
        if compression_type == "jpeg":
            color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            _, self.color_data = cv2.imencode(
                ".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, 100]
            )
            self.color_size_bytes = len(self.color_data)
        else:
            raise ValueError

    # Added by Seiya Ito
    # replace depth data
    def replace_depth(self, depth, compression_type="zlib_ushort"):
        if compression_type == "zlib_ushort":
            self.depth_data = zlib.compress(depth.flatten().tobytes())
            self.depth_size_bytes = len(self.depth_data)
        else:
            raise ValueError


class SensorData:
    def __init__(self):
        self.version = None
        self.strlen = None
        self.sensor_name = None
        self.intrinsic_color = None
        self.extrinsic_color = None
        self.intrinsic_depth = None
        self.extrinsic_depth = None
        self.color_compression_type = None
        self.depth_compression_type = None
        self.color_width = None
        self.color_height = None
        self.depth_width = None
        self.depth_height = None
        self.depth_shift = None
        self.num_frames = None
        self.frames = []

    def load(self, filename):
        with open(filename, "rb") as f:
            self.version = struct.unpack("I", f.read(4))[0]
            assert self.version == 4
            self.strlen = struct.unpack("Q", f.read(8))[0]
            self.sensor_name = b"".join(
                struct.unpack("c" * self.strlen, f.read(self.strlen))
            )
            self.intrinsic_color = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_color = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.intrinsic_depth = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_depth = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[
                struct.unpack("i", f.read(4))[0]
            ]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[
                struct.unpack("i", f.read(4))[0]
            ]
            self.color_width = struct.unpack("I", f.read(4))[0]
            self.color_height = struct.unpack("I", f.read(4))[0]
            self.depth_width = struct.unpack("I", f.read(4))[0]
            self.depth_height = struct.unpack("I", f.read(4))[0]
            self.depth_shift = struct.unpack("f", f.read(4))[0]
            self.num_frames = struct.unpack("Q", f.read(8))[0]
            self.frames = []
            for _ in tqdm(range(self.num_frames)):
                frame = RGBDFrame()
                frame.load(f)
                self.frames.append(frame)

    # Added by Seiya Ito
    # dump sensor data
    def dump(self, filename):
        with open(filename, "wb") as f:
            f.write(struct.pack("<I", self.version))
            f.write(struct.pack("<Q", self.strlen))
            assert len(self.sensor_name) == self.strlen
            f.write(struct.pack("<{}s".format(self.strlen), self.sensor_name))
            f.write(struct.pack("<16f", *self.intrinsic_color.flatten()))
            f.write(struct.pack("<16f", *self.extrinsic_color.flatten()))
            f.write(struct.pack("<16f", *self.intrinsic_depth.flatten()))
            f.write(struct.pack("<16f", *self.extrinsic_depth.flatten()))
            f.write(
                struct.pack(
                    "<i",
                    get_key_from_value(
                        COMPRESSION_TYPE_COLOR, self.color_compression_type
                    ),
                )
            )
            f.write(
                struct.pack(
                    "<i",
                    get_key_from_value(
                        COMPRESSION_TYPE_DEPTH, self.depth_compression_type
                    ),
                )
            )
            f.write(struct.pack("<I", self.color_width))
            f.write(struct.pack("<I", self.color_height))
            f.write(struct.pack("<I", self.depth_width))
            f.write(struct.pack("<I", self.depth_height))
            f.write(struct.pack("<f", self.depth_shift))
            f.write(struct.pack("<Q", self.num_frames))

            for frame in tqdm(self.frames):
                frame.dump(f)

    # Added by Seiya Ito
    # replace sensor data
    def replace(self, new):
        frame_ids = []
        for i in tqdm(range(self.num_frames)):
            c2w = self.frames[i].camera_to_world
            if np.isnan(c2w).sum() != 0 or np.isinf(c2w).sum() != 0:
                frame_ids.append(i)
                continue

            color = imageio.imread(os.path.join(new, "color", f"{i}.jpg"))
            color = cv2.resize(
                color, (self.color_width, self.color_height), cv2.INTER_LINEAR
            )
            self.frames[i].replace_color(color, "jpeg")
            depth = cv2.imread(
                os.path.join(new, "depth", f"{i}.png"), cv2.IMREAD_UNCHANGED
            )
            depth = cv2.resize(
                depth, (self.depth_width, self.depth_height), cv2.INTER_NEAREST
            )
            self.frames[i].replace_depth(depth, "zlib_ushort")

        for frame_id in reversed(frame_ids):
            del self.frames[frame_id]
            self.num_frames -= 1

    def export_depth_images(self, output_path, image_size=None, frame_skip=1):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print(
            "exporting", len(self.frames) // frame_skip, " depth frames to", output_path
        )
        for f in tqdm(range(0, len(self.frames), frame_skip)):
            depth_data = self.frames[f].decompress_depth(self.depth_compression_type)
            depth = np.fromstring(depth_data, dtype=np.uint16).reshape(
                self.depth_height, self.depth_width
            )
            if image_size is not None:
                depth = cv2.resize(
                    depth,
                    (image_size[1], image_size[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            with open(
                os.path.join(output_path, str(f) + ".png"), "wb"
            ) as fout:  # write 16-bit
                writer = png.Writer(
                    width=depth.shape[1], height=depth.shape[0], bitdepth=16
                )
                depth = depth.reshape(-1, depth.shape[1]).tolist()
                writer.write(fout, depth)

    def export_color_images(self, output_path, image_size=None, frame_skip=1):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print(
            "exporting", len(self.frames) // frame_skip, "color frames to", output_path
        )
        for f in tqdm(range(0, len(self.frames), frame_skip)):
            color = self.frames[f].decompress_color(self.color_compression_type)
            if image_size is not None:
                color = cv2.resize(
                    color,
                    (image_size[1], image_size[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            imageio.imwrite(os.path.join(output_path, str(f) + ".jpg"), color)

    def save_mat_to_file(self, matrix, filename):
        with open(filename, "w") as f:
            for line in matrix:
                np.savetxt(f, line[np.newaxis], fmt="%f")

    def export_poses(self, output_path, frame_skip=1):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print(
            "exporting", len(self.frames) // frame_skip, "camera poses to", output_path
        )
        for f in tqdm(range(0, len(self.frames), frame_skip)):
            self.save_mat_to_file(
                self.frames[f].camera_to_world,
                os.path.join(output_path, str(f) + ".txt"),
            )

    def export_intrinsics(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print("exporting camera intrinsics to", output_path)
        self.save_mat_to_file(
            self.intrinsic_color, os.path.join(output_path, "intrinsic_color.txt")
        )
        self.save_mat_to_file(
            self.extrinsic_color, os.path.join(output_path, "extrinsic_color.txt")
        )
        self.save_mat_to_file(
            self.intrinsic_depth, os.path.join(output_path, "intrinsic_depth.txt")
        )
        self.save_mat_to_file(
            self.extrinsic_depth, os.path.join(output_path, "extrinsic_depth.txt")
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("sens")
    parser.add_argument("--output_path", default="output.sens")
    parser.add_argument("--rendered_images", default=None)
    args = parser.parse_args()

    sd = SensorData()
    sd.load(args.sens)
    if args.rendered_images is not None:
        sd.replace(args.rendered_images)
    sd.dump(args.output_path)
