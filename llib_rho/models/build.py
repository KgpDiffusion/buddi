
from .regressors.buddi import build_diffusion_transformer as build_buddi

def build_model(model_cfg):
    model_type = model_cfg.type
    model = build_buddi(model_cfg.diffusion_transformer)
    return model