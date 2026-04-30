import torch.utils.data
from .. import data_structures


class SequenceDataset(torch.utils.data.Dataset):
    in_vocabulary: data_structures.vocabulary.Vocabulary
    out_vocabulary: data_structures.vocabulary.Vocabulary
