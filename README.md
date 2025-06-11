# HAR Datasets

This library provides support for popular HAR (human activity recognition) datasets including

- metadata descriptions in [DCAT-AP](https://www.dcat-ap.de/) and [Croissant](https://github.com/mlcommons/croissant)
- downloading from original source
- parsing into a centralized format
- preparation via config
- pytorch integration

# How To Use

```python
from har_datasets.dataset.get_har_dataset import get_har_dataset, HAR_DATASET_ID

dataset = get_har_dataset(dataset_id=HAR_DATASET_ID.UCI_HAR)

train_loader, test_loader, val_loader = dataset.get_dataloaders(
    batch_sizes=(32, 1, 1), shuffles=(True, False, False)
)
```

# Supported HAR Datasets

- [x] [UCI-HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
- [] [WISDM](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset)
- [] [RealWorld](https://www.uni-mannheim.de/dws/research/projects/activity-recognition/dataset/dataset-realworld/)
- [] [MotionSense](https://github.com/mmalekzadeh/motion-sense)
- [] [KU-HAR](https://data.mendeley.com/datasets/45f952y38r/5)
- [] [ExtraSensory](http://extrasensory.ucsd.edu/)
- [] [OPPORTUNITY](https://archive.ics.uci.edu/dataset/226/opportunity+activity+recognition)
- [] [PAMAP2](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)
- [] [Daphnet](https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait)
- [] [MHEALTH](hhttps://archive.ics.uci.edu/dataset/319/mhealth+dataset)
- [] [UTD-MHAD](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html)
- [] [USC-SIPI (USC-HAD)](https://sipi.usc.edu/had/)
- [] [HuGaDB](https://github.com/romanchereshnev/HuGaDB)
- [] [iSPL IMU-Stretch](https://github.com/thunguyenth/HAR_IMU_Stretch)
- [] [w-HAR](https://github.com/thunguyenth/HAR_IMU_Stretch)
- [] [SWELL](https://www.kaggle.com/datasets/qiriro/swell-heart-rate-variability-hrv)


- more coming soon

# Common Format

Since all HAR datasets do not share a common format, specific parsers are provided for each datasets to bring it into a common format which simplifies preparation. As common format a single csv is specified, which contains sensor channels as different columns. Additionally it contains the columns

- activity_id
- subj_id
- activity_block_id
- activity_name

# Preparation Config

For benchmarking, all HAR datasets should be prepared in the same manner, e.g. window size and displacement. For this purpose, a centralized config is needed. Additionally dataset-specific configs are required, e.g. to select only a subset of sensor channels. For this purpose, a hierarchical config system using [hydra](https://hydra.cc/docs/intro/) and specified with yaml is utilized. The config is then validated with [pydantic](https://docs.pydantic.dev/latest/) against a specified [schema](./src/har_datasets/schema/schema.py) for debugging purposes.

# Preparation Features

### Normalization

### Windowing

### Resampling

### Spectrogram Generation

### Differentiation

### Class Weights Computation

### Subject Cross Validation

# PyTorch Support

We provide [torch](https://pytorch.org/) datasets for easy integration into existing code bases.

# Example