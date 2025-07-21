# WHAR Datasets

This library provides data handling support for popular WHAR (Wearbale Human Activity Recognition) datasets including

<!-- - metadata descriptions in [DCAT-AP](https://www.dcat-ap.de/) and [Croissant](https://github.com/mlcommons/croissant) -->
- downloading from original source
- parsing into a standardized data format
- configuration-driven preprocessing, splitting, normalization, and more
- integration with pytorch and tensorflow

# How to Install

```
pip install "git+https://github.com/teco-kit/whar-datasets.git"
```

This installs the library into the active environment.

# How To Use

```python
from whar_datasets.adapters.pytorch import PytorchAdapter
from whar_datasets.support.getter import WHARDatasetID, get_whar_cfg

cfg = get_whar_cfg(WHARDatasetID.UCI_HAR)
dataset = PytorchAdapter(cfg, override_cache=False)

train_loader, val_loader, test_loader = dataset.get_dataloaders(train_batch_size=32)
```

Not yet natively supported WHAR datasets can be integrated via a custom configuration (with parser).

# Currently Supported Datasets

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
- [] [HAPT](https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions) 
<!-- (update of UCI_HAR) -->


