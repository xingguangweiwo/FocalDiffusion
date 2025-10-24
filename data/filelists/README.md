# FocalDiffusion file lists

Each text file enumerates the samples that compose a training split. The loader
now accepts the same whitespace separated format used by **Marigold**:

```
<relative_rgb_path> <relative_depth_path> [optional_extra_tokens]
```

* The first token points to an RGB image (PNG/JPG). When only an RGB frame and a
  depth map are provided the dataset will treat the RGB as the all-in-focus
  reference and synthesise a focal stack on the fly via the built-in circle of
  confusion simulator.
* The second token is the metric depth map. PNG/EXR/NPY/HDF5 are supported –
  HyperSim distributes depth in HDF5 containers by default, while the Marigold
  preprocessing scripts convert them to PNG for convenience.
* Additional tokens are optional. You can pass `key=value` pairs (e.g.
  `generate_focal_stack=false`) or provide an explicit stack directory as a
  third path. Integers are interpreted as the number of focal slices to load.

Lines that begin with `#` are ignored. Paths are resolved relative to the
`data_root` declared in the training YAML or, when mixing datasets, relative to
the per-source `data_root` entry.

### JSON lines

For complex situations you can still emit a JSON object per line. The same keys
listed in earlier revisions (`depth`, `all_in_focus`, `focal_stack_dir`,
`camera`, …) remain valid. JSON lines override any inference performed by the
plain-text parser, so you can explicitly request pre-rendered focal stacks or
specify the HDF5 dataset name when needed.

### HyperSim example

```
ai_001_001/ai_001_001/images/scene_cam_00_final_preview/frame.0000.color.jpg \
  ai_001_001/ai_001_001/images/scene_cam_00_geometry_hdf5/frame.0000.depth_meters.hdf5
ai_001_002/ai_001_002/images/scene_cam_00_final_preview/frame.0000.color.jpg \
  ai_001_002/ai_001_002/images/scene_cam_00_geometry_hdf5/frame.0000.depth_meters.hdf5
```

The shipped file lists keep HyperSim training samples inside
`ai_001_001/frame.0000`–`frame.0099` (100 entries) and move validation/test to
`ai_001_002/frame.0000`–`frame.0019` so every path exists in the raw download
without referencing indices above `frame.0099`.

### Virtual KITTI example

```
Scene01/clone/frames/rgb/Camera_0/rgb_00042.jpg \
  Scene01/clone/frames/depth/Camera_0/depth_00042.png
Scene01/clone/frames/rgb/Camera_0/rgb_00043.jpg \
  Scene01/clone/frames/depth/Camera_0/depth_00043.png
```

### Mixing HyperSim and Virtual KITTI

Define separate file lists per dataset and list them in the training config
under `train_sources` / `val_sources`. The loader will build a concatenated
dataset so both sources contribute to each epoch:

```yaml
data:
  train_sources:
    - name: hypersim
      data_root: /datasets/hypersim
      filelist: data/filelists/hypersim_train.txt
    - name: virtual_kitti
      data_root: /datasets/vkitti
      filelist: data/filelists/vkitti_train.txt
```

Validation/test splits follow the same structure.
