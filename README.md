# A Benchmark for 3D Reconstruction with Semantic Completion in Dynamic Environments

This is an official implementation of "[A Benchmark for 3D Reconstruction with Semantic Completion in Dynamic Environments](https://doi.org/10.1007/978-981-97-4249-3_7)" presented in [IW-FCV2024](https://sites.google.com/view/iw-fcv2024/home).

## Setting up the environment
We recommend using [Docker Compose](https://docs.docker.com/compose/) to set up the environment for this repository. Follow the steps below to get started.

1. Clone the repository.
    ```bash
    git clone https://github.com/seiyaito/Dyna3DBench.git
    cd Dyna3DBench
    ```
2. Use Docker Compose to build the images.
    ```bash
    docker compose build
    ```

This repository depends on other projects. If you prefer not to use Docker Compose, please check the respective repositories for detailed instructions.

- [Scene-Diffuser \[Huang+, CVPR2023\]](https://github.com/scenediffuser/Scene-Diffuser)
- [SCFusion \[Wu+, 3DV2020\]](https://github.com/ShunChengWu/SCFusion)

## Dataset Construction

### Expected Dataset Structure for ScanNet

Please follow the [instruction](https://github.com/ScanNet/ScanNet?tab=readme-ov-file#scannet-data) to download the ScanNet data.

```
ScanNet/
  scans/
    scene0000_00/
      pose/
        1.txt
        2.txt
        ...
      scene0000_00.sens
      scene0000_00.txt
      scene0000_00_vh_clean_2.ply
```

### Expected Structure for Huma Motion Generation

We employ Scene-Diffuser for Human Motion Generation. Please follow the [instructions](https://github.com/scenediffuser/Scene-Diffuser/tree/main?tab=readme-ov-file#data--checkpoints) to download the necessary files below.

```
data/
  checkpoints/
    2022-04-13_18-29-56_POINTTRANS_C_32768/model.pth
    2022-11-09_12-54-50_MotionGen_ddm_T200_lr1e-4_ep300/ckpt/model.pth
  PROX/
    body_segments/
      back.json
      body_mask.json
      ...
  SMPLX/
    models/
      SMPLX_FEMALE.npz
      SMPLX_FEMALE.pkl
      ...
```

### Dataset Path Settings for Docker Users

In order to share datasets between containers, the paths to the datasets are managed by environment variables.
Environment variables are assumed to be defined in `.env`. We provide `.env_sample` as a sample, so please change the paths to match your environment and place it as `.env` in the same directory as `compose.yml`.  
Note that if you just want to generate dynamic scenes, the only dataset you need is ScanNet.

### Human Motion Generation

This repository provides a script to apply Scene-Diffuser [Huang+, CVPR2023] to a dataset. Please check [scripts/run_scenediffuser.sh](scripts/run_scenediffuser.sh) for details.

```
docker compose run --rm scenediffuser scripts/run_scenediffuser.sh
```

### Scene-Motion Synthesis

This repository provides a script to synthesize static scene models and human motions. Please check [scripts/run_synthesis.sh](scripts/run_synthesis.sh) for details.

```
docker compose run --rm dyna3dbench scripts/run_synthesis.sh
```


## Semantic 3D Reconstruction

This repository provides a script to apply SCFusion [Wu+, 3DV2020] to a dataset. Please check [scripts/run_scfusion.sh](scripts/run_scfusion.sh) for details.

```
docker compose run --rm scfusion scripts/run_scfusion.sh \
  -d ${DYNA3DBENCH}/sens/0 \
  -o outputs/Dyna3DBench/0
```


## Citation
If you use our code or dataset in your work, please cite our paper:

```
@inproceedings{zhou2024dyna3dbench,
  author    = {Qinyuan Zhou and Seiya Ito and Kazuhiko Sumi},
  title     = {A Benchmark for 3D Reconstruction with Semantic Completion in Dynamic Environments},
  booktitle = {The 30th International Workshop on Frontiers of Computer Vision (IW-FCV)},
  pages     = {81–92},
  year      = {2024},
}
```

## LICENSE

This repository is licensed under the MIT License. See LICENSE for more details.

Since this repository depends on other projects, you would follow their respective licenses:

- [ScanNet](https://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf)
- [Scene-Diffuser](https://github.com/scenediffuser/Scene-Diffuser/blob/main/LICENSE)
- [SCFusion](https://github.com/ShunChengWu/SCFusion/blob/main/LICENSE)
- [PROX](https://prox.is.tue.mpg.de/license.html)
- [LEMO](https://github.com/sanweiliti/LEMO/blob/main/LICENSE)
