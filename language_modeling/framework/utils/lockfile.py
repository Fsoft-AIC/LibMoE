import os
import fcntl
import time
import errno


class LockFile:
    def __init__(self, fname: str):
        self._fname = fname
        self._fd = None

    def acquire(self):
        old_mask = os.umask(0)
        self._fd=os.open(self._fname, os.O_CREAT | os.O_WRONLY, mode=0o777)
        os.umask(old_mask)

        while True:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except IOError as e:
                if e.errno != errno.EAGAIN:
                    raise
                else:
                    time.sleep(0.1)

    def release(self):
        fcntl.flock(self._fd, fcntl.LOCK_UN)
        os.close(self._fd)
        self._fd = None

    def __enter__(self):
        self.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class MultiUserGlobalLockFile(LockFile):
    def __init__(self, fname: str):
        old_umask = os.umask(0)
        os.makedirs("/tmp/arf_locks", exist_ok=True, mode=0o777)
        os.umask(old_umask)

        super().__init__(f"/tmp/arf_locks/{fname}")

# class LockFile:
#     pass

# class MultiUserGlobalLockFile(LockFile):
#     pass
