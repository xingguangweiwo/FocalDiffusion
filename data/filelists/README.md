# FocalDiffusion file lists

Each text file enumerates the relative paths that compose a focal stack sample.
Every non-comment line should follow the comma separated pattern:

```
<relative_path_to_stack>,<relative_path_to_depth_map>,<num_images_in_stack>
```

The paths are interpreted relative to the `data_root` entry configured in the
training YAML files. Lines beginning with `#` are ignored. The `num_images`
column is optional; when omitted the loader will fall back to the configured
`focal_stack_size`.

You can create separate lists for training, validation and testing. Example
entries for the supported datasets are provided below together with the
directory layouts expected by the loader.

## HyperSim

Assuming your pre-processed focal stacks follow the structure
`<data_root>/scenes/<scene_id>/images/<frame_id>/*.png` and the ground-truth
depth lives in `<data_root>/scenes/<scene_id>/depth/<frame_id>.depth.exr`, the
file list would look like:

```
scenes/ai_001_001/images/frame.0000,scenes/ai_001_001/depth/frame.0000.depth.exr,9
scenes/ai_001_002/images/frame.0005,scenes/ai_001_002/depth/frame.0005.depth.exr,9
```

## Virtual KITTI

Virtual KITTI stacks are usually generated per scene under
`<data_root>/SceneXX/frames/rgb/Camera_0`. Depth maps share the same frame id in
`frames/depth/Camera_0`. The corresponding file list lines become:

```
Scene01/frames/rgb/Camera_0/rgb00001,Scene01/frames/depth/Camera_0/depth00001.png,7
Scene02/frames/rgb/Camera_0/rgb00001,Scene02/frames/depth/Camera_0/depth00001.png,7
```

## Mixed datasets

When mixing HyperSim and Virtual KITTI (or any other source) you can keep a
single shared `data_root` and simply point different lines to the desired stack
directories. The loader only requires that each relative path resolves to a
directory containing the focal slices and that the matching depth map exists.
