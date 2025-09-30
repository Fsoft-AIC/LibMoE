import os
import torch
from typing import Any
from ... import utils


class CacheLoaderMixin:
    VERSION = 0
    cache_dir = "./cache"

    ONCE_DONE = set()

    @classmethod
    def once_per_config(cls, fn):
        def new_fn(self, *args, **kwargs):
            my_id = self.config_id() + fn.__name__
            res = None
            if my_id not in cls.ONCE_DONE:
                res = fn(self, *args, **kwargs)
                cls.ONCE_DONE.add(my_id)
            return res

        return new_fn

    def config_id(self) -> str:
        raise NotImplementedError()

    def generate_dataset(self) -> Any:
        raise NotImplementedError()

    def load_cache(self):
        dirname = os.path.join(self.cache_dir, self.__class__.__name__)
        os.makedirs(dirname, exist_ok=True)

        prefix = os.path.join(dirname, self.config_id())

        # with utils.LockFile(f"{prefix}.lock"):
        fname = f"{prefix}.pth"
        if os.path.isfile(fname):
            try:
                data = torch.load(fname)
                if data["version"] == self.VERSION:
                    return data["data"]
            except:
                print("Cache file corrupt.")
                pass

        print("Generating dataset...")
        res = self.generate_dataset()
        print("Saving cache...")
        torch.save({"data": res, "version": self.VERSION}, fname)
        print("Done.")
        return res
