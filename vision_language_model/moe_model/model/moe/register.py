import os
import glob

MOE_REGISTRY = {}
def register_moe(*names):
    '''
    Registry new MoE class
    
    '''
    def decorate(cls):
        for name in names:
            if name in MOE_REGISTRY:
                if MOE_REGISTRY[name] != cls:
                    assert name not in MOE_REGISTRY, f"Model named '{name}' conflicts with existing model! Please register with a non-conflicting alias instead. \n {cls} \n Models: {MOE_REGISTRY}"
            MOE_REGISTRY[name] = cls
    return decorate

def get_moe(model_name):
    try:
        return MOE_REGISTRY[model_name]
    except KeyError:
        raise ValueError(f"Attempted to load moe method'{model_name}', but no model for this name found! Supported model names: {', '.join(MOE_REGISTRY.keys())}")