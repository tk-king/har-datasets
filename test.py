from typing import Any, Dict
import hydra
from omegaconf import OmegaConf
from har_datasets.schema.schema import Config
from har_datasets.old.dataparser import DataParser


@hydra.main(config_path="config", config_name="cfg", version_base=None)
def main(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = Config(**cfg)
    print(cfg.common)
    print(cfg.dataset)

    dp = DataParser(cfg)


if __name__ == "__main__":
    main()
