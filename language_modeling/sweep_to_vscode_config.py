#!/usr/bin/python3

import yaml
import sys
import subprocess
import os

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <the yaml file to run>")
    sys.exit(-1)

with open(sys.argv[1]) as f:
    config = yaml.safe_load(f)

args = []

for p, pval in config["parameters"].items():
    if p in ["log", "sweep_id_for_grid_search"]:
        continue

    args.append("-" + p)
    if "value" in pval:
        assert "values" not in pval
        args.append(pval["value"])
    elif "values" in pval:
        if len(pval["values"]) == 1:
            args.append(pval["values"][0])
        else:
            while True:
                print(f"Choose value for \"{p}\"")
                for i, v in enumerate(pval["values"]):
                    print(f"  {i+1}: {v}")

                choice = input("> ")
                if not choice.isdigit() or int(choice) < 1 or int(choice) > len(pval["values"]):
                    print("Invalid choice.")
                    continue

                args.append(pval["values"][int(choice) - 1])
                break

if "name" not in config["parameters"]:
    args.append("-name")
    args.append(os.path.basename(sys.argv[1]).replace(".yaml", ""))


print("Arg list:")
line = ""
for i in range(0, len(args), 2):
    curr_arg = f"\"{args[i]}\", \"{args[i+1]}\", "
    if (len(line) + len(curr_arg)) > 100:
        print(line)
        line = curr_arg
    else:
        line = line + curr_arg

if line:
    print(line)

    # print(f"  {args[i]}: {args[i+1]}")
# print(", ".join(["\""+str(a)+"\"" for a in args]))
