# HAR Datasets

This library provides support for popular HAR (human activity recognition) datasets including

- metadata descriptions in [DCAT-AP](https://www.dcat-ap.de/) and [Croissant](https://github.com/mlcommons/croissant)
- downloading from original source with caching
- parsing into a centralized format with caching
- preparation via config with caching
- pytorch integration.

HAR datasets not included can be used by writing a custom config and parser, which can then be integrated easily.

# How to Install

```
pip install "git+https://github.com/maxbrzr/har-datasets.git"
```

This installs the library into the active environment.

# How To Use

```python
from har_datasets.dataset.har_dataset import HARDataset
from har_datasets.supported.getter import DatasetId, get_har_dataset_cfg_and_parser

cfg, parse = get_har_dataset_cfg_and_parser(dataset_id=DatasetId.UCI_HAR)
dataset = HARDataset(cfg=cfg, parse=parse)

train_loader, val_loader, test_loader = dataset.get_dataloaders()
```

For unsupported har datasets, a custom parse function and config can be implented and used instead.

# Supported HAR Datasets

- [x] [UCI-HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
- [x] [WISDM-12](https://www.cis.fordham.edu/wisdm/dataset.php)
- [X] [MHEALTH](https://archive.ics.uci.edu/dataset/319/mhealth+dataset)
- [X] [MotionSense](https://github.com/mmalekzadeh/motion-sense)
- [X] [OPPORTUNITY](https://archive.ics.uci.edu/dataset/226/opportunity+activity+recognition)
- [X] [PAMAP2](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)
- [X] [DSADS](https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities)
- [X] [Daphnet](https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait)
- [X] [KU-HAR](https://data.mendeley.com/datasets/45f952y38r/5)
- [X] [HARSense](https://ieee-dataport.org/open-access/harsense-statistical-human-activity-recognition-dataset)
- [] [SHL](http://www.shl-dataset.org/dataset/)
- [] [RealLifeHar](https://lbd.udc.es/research/real-life-HAR-dataset/)
- [] [ExtraSensory](http://extrasensory.ucsd.edu/)
- [] [RealWorld](https://www.uni-mannheim.de/dws/research/projects/activity-recognition/dataset/dataset-realworld/)
- [] [UTD-MHAD](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html)
- [] [USC-SIPI (USC-HAD)](https://sipi.usc.edu/had/)
- [] [HuGaDB](https://github.com/romanchereshnev/HuGaDB)
- [] [iSPL IMU-Stretch](https://github.com/thunguyenth/HAR_IMU_Stretch)
- [] [w-HAR](https://github.com/thunguyenth/HAR_IMU_Stretch)
- [] [SWELL](https://www.kaggle.com/datasets/qiriro/swell-heart-rate-variability-hrv)
- [] [WISDM-19](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset)
- [] [DG]()
- [] [HAPT](https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions) (update of UCI_HAR)

# Common Format

Since all HAR datasets do not share a common format, specific parsers are provided for each datasets to bring it into a common format which simplifies preparation. As common format a single csv is specified, which contains sensor channels as different columns. Additionally it contains the columns

| Column         | Type  |
|----------------|-------|
| timestamp      | datetime in ns |
| subject_id     | int   |
| activity_id    | int   |
| activity_name  | str   |
| session_id     | int   |

All parsers must ensure to output this format. 

# Config

A hierarchical config system built on [pydantic](https://docs.pydantic.dev/latest/) is used for both common and dataset-specific configuration.

### Common Config

- dataset directory
- resampling frequency 
- include derivative
- include spectrograms
- spectrogram hyperparams

### Dataset-Specific Config

- info
    - identifier (name)
    - download url
    - sampling frequency
    - activity names
    - sensor channel names
    - subject ids
- preprocessing
    - activity and subject selections
    - normalization type (same type for all channels but applied individually to sensor channels)
    - sliding window hyperparams
    - cache df, windows, spectrograms
- training
    - given split into train, val, test
    - split groups for subject-based cross validation
    - batch size, learning rate, number of epochs, seed
    - in_memory or read each sample from disk


# Preparation Features

- [x] Resampling
- [x] Normalization
- [x] Differentiation
- [x] Windowing
- [x] Spectrogram Generation
- [x] Class Weights Computation
- [x] Subject Cross Validation

# PyTorch Support

We provide [torch](https://pytorch.org/) datasets for easy integration into existing code bases.

# TODOS

- paper specifc time windows
- alle channel same
- configurable preprocessing piepline
- spectrogramm allows multiple transforms
- class resampling
- optional in mem or not, window chaching


