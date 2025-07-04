from whar_datasets.core.config import (
    Common,
    Dataset,
    GivenSplit,
    WHARConfig,
    Info,
    Preprocessing,
    Selections,
    SlidingWindow,
    Split,
    SubjCrossValSplit,
    Training,
)

cfg_dsads = WHARConfig(
    common=Common(
        datasets_dir="./datasets",
    ),
    dataset=Dataset(
        info=Info(
            id="dsads",
            download_url="https://archive.ics.uci.edu/static/public/256/daily+and+sports+activities.zip",
            sampling_freq=25,
            num_of_subjects=8,
            num_of_activities=19,
            num_of_channels=45,
        ),
        preprocessing=Preprocessing(
            selections=Selections(
                activity_names=[
                    "sitting",  # A1
                    "standing",  # A2
                    "lying on back",  # A3
                    "lying on right side",  # A4
                    "ascending stairs",  # A5
                    "descending stairs",  # A6
                    "standing in an elevator still",  # A7
                    "moving around in an elevator",  # A8
                    "walking in a parking lot",  # A9
                    "walking on treadmill (flat, 4 km/h)",  # A10
                    "walking on treadmill (15° incline, 4 km/h)",  # A11
                    "running on treadmill (8 km/h)",  # A12
                    "exercising on a stepper",  # A13
                    "exercising on a cross trainer",  # A14
                    "cycling on exercise bike (horizontal)",  # A15
                    "cycling on exercise bike (vertical)",  # A16
                    "rowing",  # A17
                    "jumping",  # A18
                    "playing basketball",  # A19
                ],
                sensor_channels=[
                    # Trunk (T)
                    "T_xacc",
                    "T_yacc",
                    "T_zacc",
                    "T_xgyro",
                    "T_ygyro",
                    "T_zgyro",
                    "T_xmag",
                    "T_ymag",
                    "T_zmag",
                    # Right Arm (RA)
                    "RA_xacc",
                    "RA_yacc",
                    "RA_zacc",
                    "RA_xgyro",
                    "RA_ygyro",
                    "RA_zgyro",
                    "RA_xmag",
                    "RA_ymag",
                    "RA_zmag",
                    # Left Arm (LA)
                    "LA_xacc",
                    "LA_yacc",
                    "LA_zacc",
                    "LA_xgyro",
                    "LA_ygyro",
                    "LA_zgyro",
                    "LA_xmag",
                    "LA_ymag",
                    "LA_zmag",
                    # Right Leg (RL)
                    "RL_xacc",
                    "RL_yacc",
                    "RL_zacc",
                    "RL_xgyro",
                    "RL_ygyro",
                    "RL_zgyro",
                    "RL_xmag",
                    "RL_ymag",
                    "RL_zmag",
                    # Left Leg (LL)
                    "LL_xacc",
                    "LL_yacc",
                    "LL_zacc",
                    "LL_xgyro",
                    "LL_ygyro",
                    "LL_zgyro",
                    "LL_xmag",
                    "LL_ymag",
                    "LL_zmag",
                ],
            ),
            sliding_window=SlidingWindow(window_time=2.56, overlap=0),
        ),
        training=Training(
            split=Split(
                given_split=GivenSplit(
                    train_subj_ids=list(range(1, 5)),
                    val_subj_ids=list(range(5, 7)),
                    test_subj_ids=list(range(7, 9)),
                ),
                subj_cross_val_split=SubjCrossValSplit(
                    subj_id_groups=[[1, 2], [3, 4], [5, 6], [7, 8]],
                ),
            ),
        ),
    ),
)
