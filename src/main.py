import hydra
from omegaconf import DictConfig
from train import train

@hydra.main(version_base='1.1', config_path="../configs", config_name="config")
def main(cfg : DictConfig) -> None:
    if cfg.general.test_only is None:
        train(cfg)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
    


