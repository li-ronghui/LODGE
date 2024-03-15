import importlib


def get_model(cfg, datamodule, phase="train"):
    modeltype = cfg.model.model_type
    if modeltype in ["mld", "dld"]:
        return get_module(cfg, datamodule)
    else:
        raise ValueError(f"Invalid model type {modeltype}.")


def get_module(cfg, datamodule):
    modeltype = cfg.model.model_type
    model_module = importlib.import_module(
        f".modeltype.{cfg.model.model_type}", package="dld.models")     # mld
    Model = model_module.__getattribute__(f"{modeltype}")
    return Model(cfg=cfg, datamodule=datamodule)


