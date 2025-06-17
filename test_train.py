import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))        # works fine
    print(cfg.dataset.path)              # now this works

if __name__ == "__main__":
    main()