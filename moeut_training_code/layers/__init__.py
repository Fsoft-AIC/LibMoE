# flake8: noqa: F401
from .cvmm import cvmm, cvmm_prepare_sel

import os
import importlib
# all_moe_types = ["moe_layer", "moe_layer_shared", "moe_layer_film", "moe_layer_xmoe", "moe_layer_perturbed",
#                 "moe_layer_deepseek", "moe_layer_deepseekgate", "moe_layer_deepseek_nogroup", "moe_layer_deepseek_sigmoidonly",
#                 "moe_layer_deepseek_no_gatingbias", "moe_layer_film_deepseek", "moe_layer_film_deepseek_after",
#                 "moe_layer_jain_fairness", "moe_layer_film_isolation", "moe_layer_film_ensemble_deepseek_after"]

def get_moe_type(moe_type: str):
    # assert moe_type in all_moe_types
    module_name = f"layers.{moe_type}"
    module = importlib.import_module(module_name)

    return getattr(module, "MoE")

_default_moe_type = os.environ.get("MOE_TYPE", "moe_layer")
MoE = get_moe_type(_default_moe_type)