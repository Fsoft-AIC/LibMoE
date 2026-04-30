import subprocess
import sys
import threading
from typing import BinaryIO, List, Optional


def _read_pipe(pipe: BinaryIO, chunks: List[bytes], stream: BinaryIO) -> None:
    while True:
        data = pipe.read(4096)
        if not data:
            break

        chunks.append(data)
        stream.write(data)
        stream.flush()


def run_command(cmd: str, get_stderr: bool = False, allow_failure: bool = False,
                stream_output: bool = False) -> Optional[str]:
    capture_stderr = get_stderr or not allow_failure or stream_output
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE if capture_stderr else None,
                            shell=True, stdin=subprocess.PIPE)
    # input.encode() if input is not None else None
    if stream_output:
        stdout_chunks: List[bytes] = []
        stderr_chunks: List[bytes] = []
        threads = [
            threading.Thread(target=_read_pipe, args=(proc.stdout, stdout_chunks, sys.stdout.buffer)),
        ]
        if proc.stderr is not None:
            threads.append(threading.Thread(target=_read_pipe, args=(proc.stderr, stderr_chunks, sys.stderr.buffer)))

        for thread in threads:
            thread.start()

        proc.wait()

        for thread in threads:
            thread.join()

        stdout = b"".join(stdout_chunks).decode(errors="replace")
        stderr = b"".join(stderr_chunks).decode(errors="replace") if proc.stderr is not None else ""
    else:
        res = proc.communicate(None)
        stdout = res[0].decode()
        stderr = res[1].decode() if res[1] is not None else ""

    if proc.returncode != 0:
        if allow_failure:
            return None
        raise RuntimeError(f"Command {cmd} failed with return code {proc.returncode} and stderr: {stderr}")
    return stdout
