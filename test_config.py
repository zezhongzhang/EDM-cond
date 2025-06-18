import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="configs", config_name="main", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == "__main__":
    main()