import hydra
from omegaconf import DictConfig
from train import train, train_regressor
from test import test

@hydra.main(version_base='1.1', config_path="../configs", config_name="config")
def main(cfg : DictConfig) -> None:
    print(cfg)
    if cfg.general.test_only is False:
        if cfg.general.edge_model in ('classifier', 'binary_classifier'):
            train(cfg)
        else:
            train_regressor(cfg)
    else:
        test(cfg)


if __name__ == "__main__":
    main()
    


