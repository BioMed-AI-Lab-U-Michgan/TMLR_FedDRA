import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer, seed_everything

class DataModule(pl.LightningDataModule):
    def __init__(self, dataset, collate_fn, transforms, data_pct, batch_size, num_workers, crop_size=224, client_index = -1, dataset_name = "others"):
        super().__init__()

        self.dataset = dataset
        self.collate_fn = collate_fn
        self.transforms = transforms
        self.data_pct = data_pct
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop_size = crop_size
        self.client_index = client_index
        self.dataset_name = dataset_name
        print("client_index:", self.client_index)
    def train_dataloader(self):
        if self.transforms:
            transform = self.transforms(True, self.crop_size)
        else:
            transform = None
        
        if self.dataset_name == "MultimodalPretrainingDataset":
            
            dataset = self.dataset(
            is_client = self.client_index, split="train", transform=transform, data_pct=self.data_pct)
            print(f"length of dataset of client{self.client_index} {len(dataset)}")
        else:
            
            dataset = self.dataset(
            split="train", transform=transform, data_pct=self.data_pct)

        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        if self.transforms:
            transform = self.transforms(False, self.crop_size)
        else:
            transform = None
        if self.dataset_name == "MultimodalPretrainingDataset":
            
            dataset = self.dataset(
            is_client = self.client_index, split="valid", transform=transform, data_pct=self.data_pct)
            
        else:
            
            dataset = self.dataset(
            split="valid", transform=transform, data_pct=self.data_pct)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=32,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        if self.transforms:
            transform = self.transforms(False, self.crop_size)
        else:
            transform = None
        if self.dataset_name == "MultimodalPretrainingDataset":
            # This is for calculating client weights and monitoring only
            '''
            dataset = self.dataset(
            is_client = self.client_index, split="valid", transform=transform, data_pct=self.data_pct)
            '''
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!using training data for calculating client weights and evaluation")
            dataset = self.dataset(
            is_client = self.client_index, split="train", transform=transform, data_pct=self.data_pct)
            # for sample a part of training data for testing
            from torch.utils.data import Subset
            import random
            seed_everything(42)
            total_size = len(dataset)
            subset_size = int(0.1 * total_size)
            indices = list(range(total_size))
            random.shuffle(indices)
            subset_indices = indices[:subset_size]
            dataset = Subset(dataset, subset_indices)
            
        else:
            
            dataset = self.dataset(
            split="test", transform=transform, data_pct=self.data_pct)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )