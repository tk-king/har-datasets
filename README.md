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

### Currently Supported Datasets

| Supported | Name           | Year | What To Cite                                                                                       | Link |
|-----------|----------------|------|---------------------------------------------------------------------------------------------|------|
| ✅        | Daphnet        | 2009 | *Ambulatory monitoring of freezing of gait in Parkinson’s disease*                         | [Link](https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait) |
| ✅        | WISDM       | 2010 | *Activity Recognition using Cell Phone Accelerometers*                                      | [Link](https://www.cis.fordham.edu/wisdm/dataset.php) |
| ✅        | OPPORTUNITY    | 2010 | *Collecting complex activity datasets in highly rich networked sensor environments*                                                  | [Link](https://archive.ics.uci.edu/dataset/226/opportunity+activity+recognition) |
| ✅        | DSADS          | 2010 | *Comparative study on classifying human activities with miniature inertial and magnetic sensors*                                                       | [Link](https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities) |
| ⬜        | USC-HAD        | 2012 | *USC-HAD: A Daily Activity Dataset for Ubiquitous Activity Recognition Using Wearable Sensors*                     | [Link](https://sipi.usc.edu/had/) |
| ✅        | PAMAP2         | 2012 | *Introducing a New Benchmarked Dataset for Activity Monitoring*                                    | [Link](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring) |
| ⬜        | DA | 2012 | *Recognizing Human Activities User-independently on Smartphones Based on Accelerometer Data*         | [Link]() |
| ✅        | UCI-HAR        | 2013 | *A Public Domain Dataset for Human Activity Recognition using Smartphones*                                              | [Link](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) |
| ⬜        | PARDUSS | 2013 | *Towards physical activity recognition using smartphone sensors*         | [Link](https://www.utwente.nl/en/eemcs/ps/research/dataset/) |
| ⬜        | MobiFall | 2014 | *The MobiFall Dataset: Fall Detection and Classification with a Smartphone*         | [Link](https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2/#) |
| ✅        | MHEALTH        | 2014 | *mHealthDroid: A Novel Framework for Agile Development of Mobile Health Applications*                              | [Link](https://archive.ics.uci.edu/dataset/319/mhealth+dataset) |
| ⬜        | SWELL-KW          | 2014 | *The SWELL Knowledge Work Dataset for Stress and User Modeling Research*                      | [Link](https://www.kaggle.com/datasets/qiriro/swell-heart-rate-variability-hrv) |
| ⬜        | SAD | 2014 | *Fusion of Smartphone Motion Sensors for Physical Activity Recognition*         | [Link](https://www.utwente.nl/en/eemcs/ps/research/dataset/) |
| ⬜        | UTD-MHAD       | 2015 | *UTD-MHAD: A Multimodal Dataset for Human Action Recognition Utilizing a Depth Camera and a Wearable Inertial Sensor*                 | [Link](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html) |
| ⬜        | MobiAct | 2016 | *The MobiAct dataset: recognition of activities of daily living using smartphones*         | [Link](https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2/#) |
| ⬜        | ExtraSensory   | 2016 | *Recognizing Detailed Human Context In-the-Wild from Smartphones and Smartwatches*                        | [Link](http://extrasensory.ucsd.edu/) |
| ✅        | HAPT           | 2016 | *Transition-aware human activity recognition using smartphones.*         | [Link](https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions) |
| ⬜        | RealWorld      | 2016 | *On-body Localization of Wearable Devices: An Investigation of Position-Aware Activity*                                          | [Link](https://www.uni-mannheim.de/dws/research/projects/activity-recognition/dataset/dataset-realworld/) |
| ⬜        | UniMiB-SHAR | 2017 | *Unimib shar: a dataset for human activity recognition using acceleration data from smartphones*         | [Link](http://www.sal.disco.unimib.it/technologies/unimib-shar/) |
| ⬜        | UMAFall | 2017 | *UMAFall: Fall Detection Dataset*         | [Link](https://figshare.com/articles/dataset/UMA_ADL_FALL_Dataset_zip/4214283) |
| ⬜        | SHL            | 2018 | *The University of Sussex-Huawei Locomotion and Transportation Dataset for Multimodal Analytics with Mobile Devices*                                                      | [Link](http://www.shl-dataset.org/dataset/) |
| ⬜        | HuGaDB         | 2018 | *HuGaDB: Human Gait Database for Activity Recognition from Wearable Inertial Sensor Networks*                                                               | [Link](https://github.com/romanchereshnev/HuGaDB) |
| ⬜        | WISDM-19       | 2019 | -                     | [Link](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset) |
| ✅        | MotionSense    | 2019 | *Mobile Sensor Data Anonymization*                                      | [Link](https://github.com/mmalekzadeh/motion-sense) |
| ⬜        | RealLifeHAR    | 2020 | *A Public Domain Dataset for Real-Life Human Activity Recognition Using Smartphone Sensors*                                                          | [Link](https://lbd.udc.es/research/real-life-HAR-dataset/) |
| ⬜        | w-HAR          | 2020 | *w-HAR: An Activity Recognition Dataset and Framework Using Low-Power Wearable Devices*                                       | [Link](https://github.com/gmbhat/human-activity-recognition) |
| ✅        | KU-HAR         | 2021 | *KU-HAR: An open dataset for heterogeneous human activity recognition*                         | [Link](https://data.mendeley.com/datasets/45f952y38r/5) |
| ✅        | HARSense       | 2021 | *HARSense: Statistical Human Activity Recognition Dataset*                                | [Link](https://ieee-dataport.org/open-access/harsense-statistical-human-activity-recognition-dataset) |
| ⬜        | iSPL | 2022 | *An Investigation on Deep Learning-Based Activity Recognition Using IMUs and Stretch Sensors*         | [Link](https://github.com/thunguyenth/HAR_IMU_Stretch) |




