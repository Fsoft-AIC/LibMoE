# import os
# num_threads = 8
# index = int(os.environ['RANK']) * num_threads
# cpu_affinity = "{}-{}".format(index, (index + num_threads) - 1)
# os.environ['OMP_NUM_THREADS'] = "{}".format(num_threads)
# os.environ['KMP_AFFINITY'] = "granularity=fine,explicit,proclist=[{}]".format(cpu_affinity)
# os.environ['NCCL_DEBUG'] = "trace"

import os
from typing import Optional
import framework
from framework.task import task_db
import torch
import numpy as np
import random
import json
from framework import dataset
import tasks

torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled = False

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def register_args(parser: framework.helpers.ArgumentParser):
    task_db.register_args(parser)
    # parser.add_argument('--local_rank', type=int)       # for deepspeed
    parser.add_argument("-state_size", default=128)
    parser.add_argument("-task", default="tuple")
    parser.add_argument("-dropout", default=0.0)
    parser.add_argument("-embedding_size", default="none", parser=parser.int_or_none_parser)
    parser.add_argument("-transformer.n_heads", default=4)
    parser.add_argument("-transformer.variant", default="standard")
    parser.add_argument("-transformer.ff_multiplier", default=2.0)
    parser.add_argument("-transformer.encoder_n_layers", default=3)
    parser.add_argument("-transformer.attention_dropout", default=0.0)
    parser.add_argument("-load_pretrained_model", type=str)
    parser.add_argument("-test_pretrained", default=1)
    parser.add_argument("-train_baseline", default=False, help="Train the model on easy task and test on hard,"
                                                               "no masking")
    parser.add_argument("-test_only", default=False)
    parser.add_argument("-nan_detect", default=False)
    parser.add_argument("-fs_cache_pattern", default="*", parser=parser.str_or_none_parser)



def initialize(restore: Optional[str] = None):
    helper = framework.helpers.TrainingHelper(wandb_project_name="lm",
                                              register_args=register_args, extra_dirs=["export", "model_weights", "tmp"],
                                              log_async=True, restore=restore)

    dataset.init_fs_cache(helper.args.fs_cache_pattern)
    task = task_db.get_task(helper.args.task)

    task = task(helper)
    return helper, task


def main():
    set_seed(42)
    helper, task = initialize()

    if helper.args.nan_detect:
        torch.autograd.set_detect_anomaly(True)

    # test moe type
    from layers import MoE
    print("="*100)
    print(type(MoE), MoE)
    print("="*100)

    if helper.args.load_pretrained_model:
        assert not helper.args.train_baseline

        print("Loading pretrained model...")

        pretrained = os.path.expanduser(helper.args.load_pretrained_model)
        if not helper.args.load_pretrained_model.endswith(".pth"):
            pretrained = os.path.join(pretrained, str(helper.args.sweep_id_for_grid_search), "model.pth")

        assert os.path.isfile(pretrained), f"Failed to load pretrained weights. File {pretrained} not found."

        if helper.dist_env.is_master():
            task.load_weights(pretrained)

        helper.distibute_model_weights()
        print("Done.")

    if helper.args.test_only:
        res = task.validate()
        helper.log(res)
        print("Validate returned:")
        print(json.dumps(res))
        print("-------------------")
    else:
        if helper.args.test_pretrained and helper.args.load_pretrained_model:
            helper.log({f"load_validation/{k}": v for k, v in task.validate().items()})

        if helper.args.train_baseline:
            task.set_baseline_mode()

        task.train()

        print("Training finished. Saving model...")
        task.save_weights()

    task.finish()
    helper.finish()


if __name__ == "__main__":
    main()
