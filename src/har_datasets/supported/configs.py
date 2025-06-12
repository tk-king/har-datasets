from har_datasets.config.config import HARConfig
from omegaconf import OmegaConf
from hydra import initialize, compose


def get_config_uci_har(config_dir: str = "../config") -> HARConfig:
    with initialize(version_base=None, config_path=config_dir):
        cfg = compose(config_name="cfg")
        cfg = OmegaConf.to_container(cfg, resolve=True)  # type: ignore
        cfg = HARConfig(**cfg)  # type: ignore

    return cfg  # type: ignore
