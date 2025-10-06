import torch.utils.data


class LimitDatasetLength:
    def __init__(self, dataset: torch.utils.data.Dataset, max_length: int) -> None:
        super().__init__()

        self.dataset = dataset
        self._new_length = min(max_length, len(dataset))

    def __len__(self) -> int:
        return self._new_length

    def __getitem__(self, index):
        return self.dataset[index]

    def __getattr__(self, item):
        return getattr(self.dataset, item)
