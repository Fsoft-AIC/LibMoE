import torch.utils.data


class LimitSequenceLength:
    def __init__(self, dataset: torch.utils.data.Dataset, max_length: int, get_length=lambda x: x["in_len"]) -> None:
        super().__init__()

        self.dataset = dataset
        self.indices = [i for i in range(len(dataset)) if get_length(dataset[i]) <= max_length]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def __getattr__(self, item):
        return getattr(self.dataset, item)
