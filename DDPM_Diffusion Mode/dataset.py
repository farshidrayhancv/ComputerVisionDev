import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import AdamW
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from datasets import load_dataset
from accelerate import Accelerator
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random 
import timeit
# from datasets import load_dataset  # Make sure to import the appropriate library for dataset loading

# class ButterflyDataset(Dataset):
#     def __init__(self, dataset_path, img_size=128):
#         self.dataset = load_dataset(dataset_path, split="train")
#         self.img_size = img_size
#         self.preprocess = transforms.Compose([
#             transforms.Resize((img_size, img_size)),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5], [0.5])
#         ])
#         self.transform = self._transform

#     def _transform(self, examples):
#         images = [self.preprocess(image.convert("RGB")) for image in examples["image"]]
#         return {"images": images}

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         return self.transform(self.dataset[idx])

# Example usage:
butterfly_dataset = "huggan/smithsonian_butterflies_subset"
train_dataloader = DataLoader(butterfly_dataset, batch_size=5, shuffle=True)

print(train_dataloader.batch_sampler)
