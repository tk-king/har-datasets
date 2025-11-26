from enum import Enum
from whar_datasets.support.getter import WHARDatasetID

class SensorLocation(Enum):
    HIP = 0
    HAND = 1
    CHEST = 2
    LEFT_ANKLE = 3
    RIGHT_ANKLE = 4
    RIGHT_ARM = 5
    LEFT_ARM = 6


class SensorType(Enum):
    ACC_X = 0
    ACC_Y = 1
    ACC_Z = 2
    GYRO_X = 3
    GYRO_Y = 4
    GYRO_Z = 5
    MAG_X = 6
    MAG_Y = 7
    MAG_Z = 8
    ECG = 9
    BODY_ACC_X = 10
    BODY_ACC_Y = 11
    BODY_ACC_Z = 12
    BODY_GYRO_X = 13
    BODY_GYRO_Y = 14
    BODY_GYRO_Z = 15
    ATTITUE_ROLL = 16
    ATTITUE_PITCH = 17
    ATTITUE_YAW = 18
    GRAVITY_X = 19
    GRAVITY_Y = 20
    GRAVITY_Z = 21
    ROTATION_RATE_X = 22
    ROTATION_RATE_Y = 23
    ROTATION_RATE_Z = 24
    USERACCLERATION_X = 25
    USERACCLERATION_Y = 26
    USERACCLERATION_Z = 27


UCI_HAR_SENSOR_TYPES = [
    (SensorLocation.HIP, SensorType.ACC_X),
    (SensorLocation.HIP, SensorType.ACC_Y),
    (SensorLocation.HIP, SensorType.ACC_Z),
    (SensorLocation.HIP, SensorType.BODY_ACC_X),
    (SensorLocation.HIP, SensorType.BODY_ACC_Y),
    (SensorLocation.HIP, SensorType.BODY_ACC_Z),
    (SensorLocation.HIP, SensorType.BODY_GYRO_X),
    (SensorLocation.HIP, SensorType.BODY_GYRO_Y),
    (SensorLocation.HIP, SensorType.BODY_GYRO_Z)
]

WIDSM_SENSOR_TYPES = [ # For WISDM, the authors name the position as "pocket"
    (SensorLocation.HIP, SensorType.ACC_X),
    (SensorLocation.HIP, SensorType.ACC_Y),
    (SensorLocation.HIP, SensorType.ACC_Z),
]

MOTIONSENSE_SENSOR_TYPES = [
    (SensorLocation.HIP, SensorType.ATTITUE_ROLL),
    (SensorLocation.HIP, SensorType.ATTITUE_PITCH),
    (SensorLocation.HIP, SensorType.ATTITUE_YAW),
    (SensorLocation.HIP, SensorType.GRAVITY_X),
    (SensorLocation.HIP, SensorType.GRAVITY_Y),
    (SensorLocation.HIP, SensorType.GRAVITY_Z),
    (SensorLocation.HIP, SensorType.ROTATION_RATE_X),
    (SensorLocation.HIP, SensorType.ROTATION_RATE_Y),
    (SensorLocation.HIP, SensorType.ROTATION_RATE_Z),
    (SensorLocation.HIP, SensorType.USERACCLERATION_X),
    (SensorLocation.HIP, SensorType.USERACCLERATION_Y),
    (SensorLocation.HIP, SensorType.USERACCLERATION_Z),
]

PAMAP2_SENSOR_TYPES = [
    (SensorLocation.HAND, SensorType.ACC_X),
    (SensorLocation.HAND, SensorType.ACC_Y),
    (SensorLocation.HAND, SensorType.ACC_Z),
    (SensorLocation.HAND, SensorType.GYRO_X),
    (SensorLocation.HAND, SensorType.GYRO_Y),
    (SensorLocation.HAND, SensorType.GYRO_Z),
    (SensorLocation.HAND, SensorType.MAG_X),
    (SensorLocation.HAND, SensorType.MAG_Y),
    (SensorLocation.HAND, SensorType.MAG_Z)
]

OPPORTUNITY_SENSOR_TYPES = [

]

MHEALTH_SENSOR_TYPES = [
    (SensorLocation.CHEST, SensorType.ACC_X),
    (SensorLocation.CHEST, SensorType.ACC_Y),
    (SensorLocation.CHEST, SensorType.ACC_Z),
    (SensorLocation.CHEST, SensorType.ECG),
    (SensorLocation.CHEST, SensorType.ECG),
    (SensorLocation.LEFT_ANKLE, SensorType.ACC_X),
    (SensorLocation.LEFT_ANKLE, SensorType.ACC_Y),
    (SensorLocation.LEFT_ANKLE, SensorType.ACC_Z),
    (SensorLocation.LEFT_ANKLE, SensorType.GYRO_X),
    (SensorLocation.LEFT_ANKLE, SensorType.GYRO_Y),
    (SensorLocation.LEFT_ANKLE, SensorType.GYRO_Z),
    (SensorLocation.LEFT_ANKLE, SensorType.MAG_X),
    (SensorLocation.LEFT_ANKLE, SensorType.MAG_Y),
    (SensorLocation.LEFT_ANKLE, SensorType.MAG_Z),
    (SensorLocation.RIGHT_ARM, SensorType.ACC_X),
    (SensorLocation.RIGHT_ARM, SensorType.ACC_Y),
    (SensorLocation.RIGHT_ARM, SensorType.ACC_Z),
    (SensorLocation.RIGHT_ARM, SensorType.GYRO_X),
    (SensorLocation.RIGHT_ARM, SensorType.GYRO_Y),
    (SensorLocation.RIGHT_ARM, SensorType.GYRO_Z),
    (SensorLocation.RIGHT_ARM, SensorType.MAG_X),
    (SensorLocation.RIGHT_ARM, SensorType.MAG_Y),
    (SensorLocation.RIGHT_ARM, SensorType.MAG_Z)
]

DSADS_SENSOR_TYPES = [
    (SensorLocation.CHEST, SensorType.ACC_X),
    (SensorLocation.CHEST, SensorType.ACC_Y),
    (SensorLocation.CHEST, SensorType.ACC_Z),
    (SensorLocation.CHEST, SensorType.GYRO_X),
    (SensorLocation.CHEST, SensorType.GYRO_Y),
    (SensorLocation.CHEST, SensorType.GYRO_Z),
    (SensorLocation.CHEST, SensorType.MAG_X),
    (SensorLocation.CHEST, SensorType.MAG_Y),
    (SensorLocation.CHEST, SensorType.MAG_Z),

    (SensorLocation.RIGHT_ARM, SensorType.ACC_X),
    (SensorLocation.RIGHT_ARM, SensorType.ACC_Y),
    (SensorLocation.RIGHT_ARM, SensorType.ACC_Z),
    (SensorLocation.RIGHT_ARM, SensorType.GYRO_X),
    (SensorLocation.RIGHT_ARM, SensorType.GYRO_Y),
    (SensorLocation.RIGHT_ARM, SensorType.GYRO_Z),
    (SensorLocation.RIGHT_ARM, SensorType.MAG_X),
    (SensorLocation.RIGHT_ARM, SensorType.MAG_Y),
    (SensorLocation.RIGHT_ARM, SensorType.MAG_Z),

    (SensorLocation.LEFT_ARM, SensorType.ACC_X),
    (SensorLocation.LEFT_ARM, SensorType.ACC_Y),
    (SensorLocation.LEFT_ARM, SensorType.ACC_Z),
    (SensorLocation.LEFT_ARM, SensorType.GYRO_X),
    (SensorLocation.LEFT_ARM, SensorType.GYRO_Y),
    (SensorLocation.LEFT_ARM, SensorType.GYRO_Z),
    (SensorLocation.LEFT_ARM, SensorType.MAG_X),
    (SensorLocation.LEFT_ARM, SensorType.MAG_Y),
    (SensorLocation.LEFT_ARM, SensorType.MAG_Z),

    (SensorLocation.RIGHT_ANKLE, SensorType.ACC_X),
    (SensorLocation.RIGHT_ANKLE, SensorType.ACC_Y),
    (SensorLocation.RIGHT_ANKLE, SensorType.ACC_Z),
    (SensorLocation.RIGHT_ANKLE, SensorType.GYRO_X),
    (SensorLocation.RIGHT_ANKLE, SensorType.GYRO_Y),
    (SensorLocation.RIGHT_ANKLE, SensorType.GYRO_Z),
    (SensorLocation.RIGHT_ANKLE, SensorType.MAG_X),
    (SensorLocation.RIGHT_ANKLE, SensorType.MAG_Y),
    (SensorLocation.RIGHT_ANKLE, SensorType.MAG_Z),

    (SensorLocation.LEFT_ANKLE, SensorType.ACC_X),
    (SensorLocation.LEFT_ANKLE, SensorType.ACC_Y),
    (SensorLocation.LEFT_ANKLE, SensorType.ACC_Z),
    (SensorLocation.LEFT_ANKLE, SensorType.GYRO_X),
    (SensorLocation.LEFT_ANKLE, SensorType.GYRO_Y),
    (SensorLocation.LEFT_ANKLE, SensorType.GYRO_Z),
    (SensorLocation.LEFT_ANKLE, SensorType.MAG_X),
    (SensorLocation.LEFT_ANKLE, SensorType.MAG_Y),
    (SensorLocation.LEFT_ANKLE, SensorType.MAG_Z)
]

KU_HAR_SENSOR_TYPES = [
    (SensorLocation.HIP, SensorType.ACC_X),
    (SensorLocation.HIP, SensorType.ACC_Y),
    (SensorLocation.HIP, SensorType.ACC_Z),
    (SensorLocation.HIP, SensorType.GYRO_X),
    (SensorLocation.HIP, SensorType.GYRO_Y),
    (SensorLocation.HIP, SensorType.GYRO_Z),
]

HAR_SENSE_SENSOR_TYPES = [

]

datasets_to_types = {
    WHARDatasetID.UCI_HAR: UCI_HAR_SENSOR_TYPES,
    WHARDatasetID.WISDM: WIDSM_SENSOR_TYPES,
    WHARDatasetID.MOTION_SENSE: MOTIONSENSE_SENSOR_TYPES,
    WHARDatasetID.PAMAP2: PAMAP2_SENSOR_TYPES,
    WHARDatasetID.OPPORTUNITY: OPPORTUNITY_SENSOR_TYPES,
    WHARDatasetID.MHEALTH: MHEALTH_SENSOR_TYPES,
    WHARDatasetID.DSADS: DSADS_SENSOR_TYPES,
    WHARDatasetID.KU_HAR: KU_HAR_SENSOR_TYPES,
    WHARDatasetID.HAR_SENSE: HAR_SENSE_SENSOR_TYPES,
}


def get_sensor_types(dataset_id: WHARDatasetID):
    if type(dataset_id) is str:
        dataset_id = WHARDatasetID(dataset_id)
    selected_dataset = datasets_to_types[dataset_id]

    sensor_locations = list(x[0].value for x in selected_dataset)
    sensor_types = list(x[1].value for x in selected_dataset)
    return sensor_locations, sensor_types
