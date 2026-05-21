# Package Architecture

`hawk_eye` is organized around the lifecycle of an aerial-imagery model:

- `hawk_eye/core/`: model wrappers, reusable network components, target typing, and asset management.
- `hawk_eye/data_generation/`: synthetic image and metadata generation.
- `hawk_eye/train/`: training entrypoints, datasets, collators, augmentations, and training utilities.
- `hawk_eye/inference/`: production model selection, benchmarking, and target-finding entrypoints.
- `third_party/`: vendored model and detection utilities. Do not place first-party code here.
- `tools/`: repository maintenance tooling such as quality gates.

Training datasets are exported through `hawk_eye.train.datasets` so training entrypoints
do not need to know the concrete subpackage path for each dataset implementation.
