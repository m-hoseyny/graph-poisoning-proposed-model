import hydra
from omegaconf import DictConfig
from train import train
from test import test

@hydra.main(version_base='1.1', config_path="../configs", config_name="config")
def main(cfg : DictConfig) -> None:
    if cfg.general.test_only is False:
        train(cfg)
    else:
        test(cfg)


if __name__ == "__main__":
    main()
    


