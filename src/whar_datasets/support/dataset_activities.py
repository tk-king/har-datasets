from typing import Dict, Iterable, List, Sequence, Tuple, Union

from whar_datasets.support.getter import WHARDatasetID

# Original: ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
CLASS_NAMES_UCI_HAR = ["walking", "walking_upstairs", "walking_downstairs", "sitting", "standing", "laying"]

# Original: ['Walking', 'Jogging', 'Upstairs', 'Downstairs', 'Sitting', 'Standing']
#CLASS_NAMES_WISDM = ["walking", "running", "jogging", "walking_upstairs", "walking_downstairs", "sitting", "standing", "laying"]
CLASS_NAMES_WISDM = ["walking", "jogging", "walking_upstairs", "walking_downstairs", "sitting", "standing"]

# Original: "downstairs", "upstairs", "walking", "jogging", "sitting", "standing",
CLASS_NAMES_MOTIONSENSE = ["walking_downstairs", "walking_upstairs", "walking", "jogging", "sitting", "standing"]


# Original: "Stand", "Walk", "Sit", "Lie"
CLASS_NAMES_OPPORTUNITY = ["standing", "walking", "sitting", "laying"]


# Original: "lying", "sitting", "standing", "walking", "running", "cycling", "nordic walking", "watching TV", "computer work", "car driving", "ascending stairs", "descending stairs", "vacuum cleaning", "ironing", "folding laundry", "house cleaning", "playing soccer", "rope jumping",
CLASS_NAMES_PAMAP2 = ["laying", "sitting", "standing", "walking", "running", "cycling", "nordic_walking", "watching_tv", "computer_work", "car_driving", "walking_upstairs", "walking_downstairs", "vacuum_cleaning", "ironing", "folding_laundry", "house_cleaning", "playing_soccer", "rope_jumping"]

# Original: "Standing still", "Sitting and relaxing", "Lying down", "Walking", "Climbing stairs", "Waist bends forward", "Frontal elevation of arms", "Knees bending (crouching)", "Cycling", "Jogging", "Running", "Jump front and back",
CLASS_NAMES_MHEALTH = ["null", "standing", "sitting", "laying", "walking", "walking_upstairs", "waist_bends_forward", "frontal_elevation_of_arms", "knees_bending_crouching", "cycling", "jogging", "running", "jump_front_and_back"]

# Original: "sitting", "standing", "lying on back", "lying on right side", "ascending stairs", "descending stairs", "standing in an elevator still", "moving around in an elevator", "walking in a parking lot", "walking on treadmill (flat, 4 km/h)", "walking on treadmill (15° incline, 4 km/h)", "running on treadmill (8 km/h)", "exercising on a stepper", "exercising on a cross trainer", "cycling on exercise bike (horizontal)", "cycling on exercise bike (vertical)", "rowing", "jumping", "playing basketball"
CLASS_NAMES_DSADS = ["sitting", "standing", "laying", "laying", "walking_upstairs", "walking_downstairs", "standing_in_an_elevator_still", "moving_around_in_an_elevator", "walking", "walking_threadmill", "walking_threadmill", "running_threadmill", "exercise_stepper", "exercise_cross_trainer", "cycling_on_exercise_bike_horizontal", "cycling_on_exercise_bike_vertical", "rowing", "jumping", "playing_basketball"]

# TODO: FIX
# Stand", "Sit", "Talk-sit", "Talk-stand", "Stand-sit", "Lay", "Lay-stand", "Pick", "Jump", "Push-up", "Sit-up", "Walk", "Walk-backward", "Walk-circle", "Run", "Stair-up", "Stair-down", "Table-tennis"
CLASS_NAMES_KU_HAR = ["standing", "sitting", "talking_sitting", "talking_standing", "Stand-sit", "laying", "laying_standing", "picking", "jumping", "push_up", "sit_up", "walking", "walking_backward", "walking_circle", "running", "stair_up", "stair_down", "table_tennis"]

# Original: "Walking", "Standing", "Upstairs", "Downstairs", "Running", "Sitting"
CLASS_NAMES_HAR_SENSE = ["walking", "standing", "walking_upstairs", "walking_downstairs", "running", "sitting"]


def get_class_names(dataset_id: WHARDatasetID):
    class_names = {
        WHARDatasetID.UCI_HAR: CLASS_NAMES_UCI_HAR,
        WHARDatasetID.WISDM: CLASS_NAMES_WISDM,
        WHARDatasetID.MOTION_SENSE: CLASS_NAMES_MOTIONSENSE,
        WHARDatasetID.OPPORTUNITY: CLASS_NAMES_OPPORTUNITY,
        WHARDatasetID.PAMAP2: CLASS_NAMES_PAMAP2,
        WHARDatasetID.MHEALTH: CLASS_NAMES_MHEALTH,
        WHARDatasetID.DSADS: CLASS_NAMES_DSADS,
        WHARDatasetID.KU_HAR: CLASS_NAMES_KU_HAR,
        WHARDatasetID.HAR_SENSE: CLASS_NAMES_HAR_SENSE,
    }
    return class_names[dataset_id]


def sanitize_class_names(class_names):
    sanitization_map = {
        "null": "Doing nothing",
        "walking": "Walking",
        "jogging": "Jogging",
        "standing": "Standing",
        "sitting": "Sitting",
        "laying": "Laying",
        "running": "Running",
        "cycling": "Cycling",
        "nordic_walking": "Nordic Walking",
        "watching_tv": "Watching TV",
        "computer_work": "Computer Work",
        "car_driving": "Car Driving",
        "walking_upstairs": "Walking Upstairs",
        "walking_downstairs": "Walking Downstairs",
        "vacuum_cleaning": "Vacuum Cleaning",
        "ironing": "Ironing",
        "folding_laundry": "Folding Laundry",
        "house_cleaning": "House Cleaning",
        "playing_soccer": "Playing Soccer",
        "rope_jumping": "Rope Jumping",
        "waist_bends_forward": "Bending forward using the waist",
        "frontal_elevation_of_arms": "Elevating the arms in the front",
        "knees_bending_crouching": "Bending the knees and crouching",
        "jump_front_and_back": "Jumping front and back"
        
    }
    return [sanitization_map.get(name, name) for name in class_names]


def _normalize_dataset_id(dataset_id: Union[str, WHARDatasetID]) -> WHARDatasetID:
    if isinstance(dataset_id, WHARDatasetID):
        return dataset_id
    return WHARDatasetID(dataset_id)


def build_activity_alignment(
    dataset_ids: Sequence[Union[str, WHARDatasetID]],
    target_classes: Iterable[str] | None = None,
    strategy: str = "intersection",
) -> Tuple[List[str], Dict[WHARDatasetID, List[int]]]:
    """Align activities across datasets and return canonical classes and mappings.

    Args:
        dataset_ids: Datasets to align (e.g., [WHARDatasetID.PAMAP2, WHARDatasetID.DSADS]).
        target_classes: Optional explicit canonical class list. If provided, strategy is ignored.
        strategy: "intersection" (default) keeps only activities present in all datasets;
            "union" keeps all activities observed across datasets (ordered by first dataset, then new ones).

    Returns:
        canonical_classes: Ordered list of canonical activity names.
        mappings: Dict from dataset_id to a list mapping the dataset's class index to canonical index
            (or -1 if that activity is dropped).
    """

    if not dataset_ids:
        raise ValueError("dataset_ids must not be empty")

    ds_ids: List[WHARDatasetID] = [_normalize_dataset_id(ds) for ds in dataset_ids]
    names_by_ds: Dict[WHARDatasetID, List[str]] = {ds: get_class_names(ds) for ds in ds_ids}

    if target_classes is not None:
        canonical = list(target_classes)
    else:
        base = list(names_by_ds[ds_ids[0]])
        if strategy == "intersection":
            canonical = [
                name for name in base if all(name in names_by_ds[ds] for ds in ds_ids[1:])
            ]
        elif strategy == "union":
            canonical = base[:]
            for ds in ds_ids[1:]:
                for name in names_by_ds[ds]:
                    if name not in canonical:
                        canonical.append(name)
        else:
            raise ValueError('strategy must be either "intersection" or "union"')

    mappings: Dict[WHARDatasetID, List[int]] = {}
    for ds in ds_ids:
        names = names_by_ds[ds]
        mappings[ds] = [canonical.index(name) if name in canonical else -1 for name in names]

    return canonical, mappings
