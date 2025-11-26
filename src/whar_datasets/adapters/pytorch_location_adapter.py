from whar_datasets.adapters.pytorch import PytorchAdapter
from whar_datasets.core.config import WHARConfig
from whar_datasets.support.sensor_types import get_sensor_types


class PyTorchLocationAdapter(PytorchAdapter):
    def __init__(self, cfg: WHARConfig, override_cache: bool = False, only_imu: bool = False):
        super().__init__(cfg, override_cache)
        sensor_locations, sensor_types = get_sensor_types(cfg.dataset_id)
        self.sensor_locations = sensor_locations
        self.sensor_types = sensor_types

    def __getitem__(self, index: int):
        super_item = super().__getitem__(index)
        return (*super_item, self.sensor_locations, self.sensor_types)
