import torch
import random
from torch.utils.data import Sampler

class ClimateBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, climate_extractor):

        self.batch_size = batch_size
        self.climate_extractor = climate_extractor

        self.climate_indices = {}
        for idx in range(len(dataset)):
            climate = climate_extractor(dataset[idx])['meta']['desc']['climate']
            if climate not in self.climate_indices:
                self.climate_indices[climate] = []
            self.climate_indices[climate].append(idx)

    def __iter__(self):

        all_batches = []

        for climate, indices in self.climate_indices.items():
            shuffled = indices.copy()
            random.shuffle(shuffled)

            for i in range(0, len(shuffled), self.batch_size):
                batch = shuffled[i:i+self.batch_size]

                if len(batch) == self.batch_size:
                    all_batches.append(batch)
                elif len(batch) > 0:

                    remain = self.batch_size - len(batch)
                    batch += random.sample(shuffled, remain)
                    all_batches.append(batch)

        random.shuffle(all_batches)

        for batch in all_batches:
            yield batch

    def __len__(self):

        total_batches = 0
        for indices in self.climate_indices.values():
            total_batches += len(indices) // self.batch_size
            if len(indices) % self.batch_size != 0:
                total_batches += 1
        return total_batches
