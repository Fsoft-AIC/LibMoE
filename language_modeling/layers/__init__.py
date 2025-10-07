# flake8: noqa: F401
from .cvmm import cvmm, cvmm_prepare_sel

import os
import importlib

def get_moe_type(moe_type: str):
    # assert moe_type in all_moe_types
    module_name = f"layers.{moe_type}"
    module = importlib.import_module(module_name)

    return getattr(module, "MoE")

_default_moe_type = os.environ.get("MOE_TYPE", "moe_layer")
MoE = get_moe_type(_default_moe_type)