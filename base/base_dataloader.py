import numpy as np
from copy import deepcopy
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

class BaseDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers, val_split = 0.0, sampler=None):
        self.shuffle = shuffle
        self.dataset = dataset
        self.nbr_examples = len(dataset)
        if sampler is not None:
            self.train_sampler = sampler
            print('sampler: {}'.format(sampler))
            self.val_sampler = None
        elif val_split: self.train_sampler, self.val_sampler = self._split_sampler(val_split)
        else: self.train_sampler, self.val_sampler = None, None

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers,
            'pin_memory': True,
            'drop_last':True
            # 'sampler':sampler
        }
        super(BaseDataLoader, self).__init__(sampler=self.train_sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        self.shuffle = False

        split_indx = int(self.nbr_examples * split)
        np.random.seed(0)

        indxs = np.arange(self.nbr_examples)
        np.random.shuffle(indxs)
        train_indxs = indxs[split_indx:]
        val_indxs = indxs[:split_indx]
        self.nbr_examples = len(train_indxs)

        train_sampler = SubsetRandomSampler(train_indxs)
        val_sampler = SubsetRandomSampler(val_indxs)
        return train_sampler, val_sampler

    def get_val_loader(self):
        if self.val_sampler is None:
            return None
        #self.init_kwargs['batch_size'] = 1
        return DataLoader(sampler=self.val_sampler, **self.init_kwargs)

class DataPrefetcher(object):
    def __init__(self, loader, device, stop_after=None):
        self.loader = loader
        self.dataset = loader.dataset
        self.stream = torch.cuda.Stream()
        self.stop_after = stop_after
        self.next_input = None
        self.next_input_a = None
        self.next_target = None
        self.mkpt0 = None
        self.mkpt1 = None
        self.m_cnt = None
        self.device = device

    def __len__(self):
        return len(self.loader)

    def preload(self):
        try:
            self.next_input, self.next_target,self.next_input_a,self.mkpt0,self.mkpt1,self.m_cnt = next(self.loaditer)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.m_cnt = None
            self.mkpt0 = None
            self.mkpt1 = None
            self.next_input_a = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(device=self.device, non_blocking=True)
            self.next_target = self.next_target.cuda(device=self.device, non_blocking=True)
            self.next_input_a = self.next_input_a.cuda(device=self.device, non_blocking=True)

    def __iter__(self):
        count = 0
        self.loaditer = iter(self.loader)
        self.preload()
        while self.next_input is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            input_a = self.next_input_a
            target = self.next_target
            mkpt0 = self.mkpt0
            m_cnt=self.m_cnt
            mkpt1 = self.mkpt1
            self.preload()
            count += 1
            yield input, target,input_a, mkpt0, mkpt1,m_cnt
            if type(self.stop_after) is int and (count > self.stop_after):
                break
