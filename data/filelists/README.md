# FocalDiffusion file lists

Each text file enumerates the relative paths that compose a focal stack sample.
Two formats are supported:

1. **Legacy CSV** – every non-comment line follows the comma separated pattern:

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

2. **JSON lines** – for more complex setups (e.g. synthesising focal stacks
   on-the-fly or reading HyperSim HDF5 depth maps) you can emit a JSON object
   per line. The following keys are recognised:

   | Key | Description |
   | --- | ----------- |
   | `depth` / `depth_path` | Path to the metric depth map. Supports PNG, EXR, NPY and HDF5. |
   | `focal_stack` / `focal_stack_dir` | Directory containing pre-rendered focal slices. Optional when `all_in_focus` is provided. |
   | `all_in_focus` / `aif` | High-resolution sharp RGB image used to synthesise a focal stack through the built-in point spread function simulator. |
   | `focus_distances` | Array of focus distances (metres) corresponding to each slice. If omitted, log-spaced values within `focal_range` are generated. |
   | `camera` | Dictionary with camera metadata (`f_number`, `focal_length`, `pixel_size`, optionally `aperture`). |
   | `depth_dataset` | Dataset name inside an HDF5 file (defaults to common HyperSim aliases). |
   | `depth_scale` / `depth_shift` | Multiplicative/additive factors applied to the loaded depth map. |
   | `depth_range` | `[min, max]` clamp for the valid metric depth range. The loader also returns the range so diffusion outputs can be re-scaled. |
   | `generate_focal_stack` | Boolean overriding the global `generate_focal_stack` flag from the config. |
   | `transpose_depth` | Set to true when the stored depth map needs transposing to match the RGB orientation. |

   Example JSON entry combining an all-in-focus frame with a HyperSim HDF5 depth
   map:

   ```json
   {
     "all_in_focus": "ai_001_010/images/scene_cam_00_final_preview/frame.0031.color.jpg",
     "depth": "ai_001_010/images/scene_cam_00_geometry_hdf5/frame.0031.depth_meters.hdf5",
     "depth_dataset": "depth_meters",
     "focus_distances": [0.5, 1.0, 2.0, 4.0, 8.0],
     "camera": {"f_number": 8.0, "focal_length": 0.05, "pixel_size": 1.2e-5},
     "generate_focal_stack": true
   }
   ```

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
