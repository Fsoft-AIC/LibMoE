import json
import sys

if len(sys.argv) != 2:
    raise Exception(f"Usage: {sys.argv[0]} <configuration name>")

name = sys.argv[1]
# name = "Fullmoe"

cfg = json.load(open('.vscode/launch.json'))["configurations"]
matchcfg = None

for c in cfg:
    if c['name'] == name:
        if matchcfg is not None:
            raise Exception(f"Multiple configurations with name {name}")
        matchcfg = c


if matchcfg is None:
    raise Exception(f"No configuration with name {name}")


print(
"""
program: main.py
command:
  - ${env}
  - python3
  - ${program}
  - ${args}
method: grid
metric:
  name: validation/mean_accuracy
  goal: maximize
parameters:
  log:
    value: wandb"""
)

skip_next = False
for a in matchcfg["args"]:
    if a.startswith("-"):
        skip_next = a in {"-log", "-reset"}
        if not skip_next:
            print(f"  {a[1:]}:")
    else:
        if not skip_next:
            print(f"    value: {a}")
