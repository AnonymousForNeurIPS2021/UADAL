
import torch.utils.data
from builtins import object


class PairedData(object):
    def __init__(self, data_loader_A, data_loader_B, max_dataset_size):
        self.data_loader_A = data_loader_A
        self.data_loader_B = data_loader_B
        self.stop_A = False
        self.stop_B = False
        self.max_dataset_size = max_dataset_size

    def __iter__(self):
        self.stop_A = False
        self.stop_B = False
        self.data_loader_A_iter = iter(self.data_loader_A)
        self.data_loader_B_iter = iter(self.data_loader_B)
        self.iter = 0
        return self

    def __next__(self):
        A, A_paths = None, None
        B, B_paths = None, None
        try:
            A, A_paths = next(self.data_loader_A_iter)
        except StopIteration:
            if A is None or A_paths is None:
                self.stop_A = True
                self.data_loader_A_iter = iter(self.data_loader_A)
                A, A_paths = next(self.data_loader_A_iter)
        try:
            B, B_paths = next(self.data_loader_B_iter)
        except StopIteration:
            if B is None or B_paths is None:
                self.stop_B = True
                self.data_loader_B_iter = iter(self.data_loader_B)
                B, B_paths = next(self.data_loader_B_iter)

        if (self.stop_A and self.stop_B) or self.iter > self.max_dataset_size:
            self.stop_A = False
            self.stop_B = False
            raise StopIteration()
        else:
            self.iter += 1
            return {'img_s': A, 'label_s': A_paths,
                    'img_t': B, 'label_t': B_paths, }


class UnalignedDataLoader():
    def initialize(self, A, B, batchSize, sampler=None):
        dataset_A = A
        dataset_B = B

        num_workers = 2
        if sampler is not None:
            data_loader_A = torch.utils.data.DataLoader(
                dataset_A,
                batch_size=batchSize,
                sampler=sampler,
                num_workers=num_workers, pin_memory=True)
        else:
            data_loader_A = torch.utils.data.DataLoader(
                dataset_A,
                batch_size=batchSize,
                shuffle=True,
                num_workers=num_workers, pin_memory=True)

        data_loader_B = torch.utils.data.DataLoader(
            dataset_B,
            batch_size=batchSize,
            shuffle=True,
            num_workers=num_workers, pin_memory=True)

        self.dataset_A = dataset_A
        self.dataset_B = dataset_B
        flip = False
        self.paired_data = PairedData(data_loader_A, data_loader_B, float("inf"))

    def name(self):
        return 'UnalignedDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return min(max(len(self.dataset_A), len(self.dataset_B)), self.opt.max_dataset_size)
