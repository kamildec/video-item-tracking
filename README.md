# Video Item Tracking

Video Processing models used for item tracking.

## Pre-commit hooks

This repository supports pre-commit hooks. Hooks are defined in [pre-commit-config.yaml](.pre-commit-config.yaml). They check formatting and typing.

### Step 1. Install pre-commit
```
pip install pre-commit
```
### Step 2. Install git hook scripts
```
pre-commit install
```

### Disabling hooks
Ways of disabling the hooks:
- uninstalling with `pre-commit uninstall`
- skipping during commit by adding `--no-verify` flag.


## Scripts

Directory scripts contains auxiliary scripts for downloading datasets.
The resulting directory structure of the datasets should be (you don't have to download all of the datasets, just the ones you will need):

```
video-item-tracking
  └── datasets
     ├── <dataset-name-1>
     |  ├── annotations
     |  |   ├── train.json
     |  |   └── test.json
     |  ├── train
     |  └── test
     ├── <dataset-name-2>
     ...
```
### Download CVAT dataset
```
cd <video-item-tracking-directory>
./scripts/download_cvat.py --task-prefix=video_object_segmentation --username=YOUR_CVAT_USERNAME --password=YOUR_CVAT_PASSWORD --save-path=datasets/cvat
```

### Download COCO dataset
```
cd <video-item-tracking-directory>
./scripts/download_coco.sh
```

### Download OVIS dataset
```
cd <video-item-tracking-directory>
./scripts/download_ovis.sh
```