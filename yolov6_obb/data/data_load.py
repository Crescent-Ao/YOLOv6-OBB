import os
from .datasets import TrainValDataset
from torch.utils.data import DataLoader, distributed
from yolov6_obb.utils.torch_utils import torch_distributed_zero_first
from loguru import logger
def create_dataloader(
    anno_file_name,
    img_size,
    batch_size,
    rank = -1,
    workers = 8,
    shuffle = False,
    data_dict = None,
):
    logger.info("Create dataloader")
    dataset = TrainValDataset(
        anno_file_name,
        batch_size = batch_size,
        stride=32,
        img_size=img_size,
        rank = rank,
        data_dict=data_dict
    )
    # with torch_distributed_zero_first(rank):
    #     dataset = TrainValDataset(
    #         anno_file_name,
    #         batch_size = batch_size,
    #         stride=32,
    #         img_size=img_size,
    #         rank = rank,
    #         data_dict=data_dict
    #     )
    batch_size =  min(batch_size,len(dataset))
    workers = min(
        [
            os.cpu_count() // int(os.getenv("WORLD_SIZE", 1)),
            batch_size if batch_size > 1 else 0,
            workers
        ]
    )
    sampler = (
        None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    )
    logger.info(sampler)
    logger.info(len(dataset))
    # dataloader =  DataLoader(dataset, batch_size = batch_size, num_workers = 4, pin_memory = True,\
    #     collate_fn = TrainValDataset.collate_fn)
    dataloader = TrainValDataLoader(
        dataset, 
        batch_size = batch_size,
        shuffle = shuffle and sampler is None,
        num_workers=workers,
        sampler = sampler,
        pin_memory = True,
        collate_fn = TrainValDataset.collate_fn,)
    logger.info(dataloader)
    return (dataloader, dataset)
class TrainValDataLoader(DataLoader):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()
    def __len__(self):
        return len(self.batch_sampler.sampler)
    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)
            
class _RepeatSampler:
    """Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
