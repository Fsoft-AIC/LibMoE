import os
import lib
import json
import copy
from typing import List, Optional
from tqdm import tqdm
import argparse


my_dir = os.path.dirname(__file__)
main_dir = os.path.abspath(my_dir+"/../..")
my_rel_dir = os.path.relpath(my_dir, main_dir)
curr_dir = os.getcwd()


def get_info(tasks: str, patch_ckpt=None, save_dir: str = None, bs: Optional[int] = None, local_rank=0):
    model_name = os.path.basename(patch_ckpt)
    res_path = f"{save_dir}/result-{model_name}.json"

    if not os.path.isfile(res_path):
        if patch_ckpt is not None:
            ckpt_path = patch_ckpt

        if bs is None:
            bs = ""
        else:
            bs = f"--batch_size {bs}"

        cmd = f"python3 main_deepspeed.py --name validate2 --log tb --restore {ckpt_path} --test_only 1 -reset 1 -lm.eval.enabled 1 {tasks} --keep_alive 0 {bs}"
        print("Validate command: ", cmd)
        out = lib.run_command(cmd)
        lines = out.splitlines()
        start_line = lines.index('Validate returned:')
        end_line = None
        for i in range(start_line, len(lines)):
            if lines[i].startswith("-------"):
                end_line = i
                break

        assert end_line is not None

        res = "\n".join(lines[start_line+1:end_line])
        os.chdir(curr_dir)

        with open(res_path, "w") as f:
            f.write(res)

    with open(res_path, "r") as f:
        res = json.load(f)

    return res


if __name__ == "__main__":
    global local_rank
    parser = argparse.ArgumentParser(description="Run tests with specified parameters.")
    parser.add_argument('--local_rank', type=int, required=False)       # for deepspeed
    parser.add_argument('--tasks', type=str, required=True, help='The tasks to run.')
    parser.add_argument('--path_weight', type=str, required=True, help='The path to the model weight file.')
    parser.add_argument('--save_dir', type=str, required=True, help='The directory to save reult')
    parser.add_argument('--bs', type=int, required=True, help='Batch size for testing.')

    args = parser.parse_args()

    get_info(args.tasks, args.path_weight, args.save_dir, args.bs)