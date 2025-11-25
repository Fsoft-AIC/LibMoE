import subprocess
from typing import Optional


def run_command(cmd: str, get_stderr: bool = False, allow_failure: bool = False) -> Optional[str]:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE if get_stderr else None,
                            shell=True, stdin=subprocess.PIPE)
    # input.encode() if input is not None else None
    res = proc.communicate(None)
    stdout = res[0].decode()
    if proc.returncode != 0:
        if allow_failure:
            return None
        stderr = res[1].decode()
        raise RuntimeError(f"Command {cmd} failed with return code {proc.returncode} and stderr: {stderr}")
    return stdout
