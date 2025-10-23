# FocalDiffusion file lists

Each text file enumerates the samples that compose a training split. The loader
now accepts the same whitespace separated format used by **Marigold**:

```
<relative_rgb_path> <relative_depth_path> [optional_extra_tokens]
```

* The first token points to an RGB image (PNG/JPG). When only an RGB frame and a
  depth map are provided the dataset will treat the RGB as the all-in-focus
  reference and synthesise a focal stack on the fly via the built-in circle of
  confusion simulator. This mirrors the MATLAB pipeline shown in the project
  discussion.
* The second token is the metric depth map. PNG/EXR/NPY/HDF5 are supported –
  HyperSim distributes depth in HDF5 containers by default, while the Marigold
  preprocessing scripts convert them to PNG for convenience.
* Additional tokens are optional. You can pass `key=value` pairs (e.g.
  `generate_focal_stack=false`) or provide an explicit stack directory as a
  third path. Integers are interpreted as the number of focal slices to load.

Lines that begin with `#` are ignored. Paths are resolved relative to the
`data_root` declared in the training YAML or, when mixing datasets, relative to
the per-source `data_root` entry.


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
