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

class ButterflyTrainer:
    def __init__(self, model, train_dataloader, num_epochs, learning_rate, num_timesteps, random_seed):
        self.model = model
        self.train_dataloader = train_dataloader
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.num_timesteps = num_timesteps
        self.random_seed = random_seed

    def train(self):
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)

        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=len(self.train_dataloader) * self.num_epochs
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(device)

        start = timeit.default_timer()
        for epoch in tqdm(range(self.num_epochs), position=0, leave=True):
            self.model.train()
            train_running_loss = 0

            for idx, batch in enumerate(tqdm(self.train_dataloader, position=0, leave=True)):
                clean_images = batch["images"].to(device)
                noise = torch.randn(clean_images.shape).to(device)
                last_batch_size = len(clean_images)

                timesteps = torch.randint(0, self.num_timesteps, (last_batch_size,)).to(device)
                noisy_images = self.add_noise(clean_images, noise, timesteps)

                optimizer.zero_grad()
                noise_pred = self.model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                train_running_loss += loss.item()

            train_loss = train_running_loss / (idx + 1)
            train_learning_rate = lr_scheduler.get_last_lr()[0]

            print("-" * 30)
            print(f"Train Loss EPOCH: {epoch + 1}: {train_loss:.4f}")
            print(f"Train Learning Rate EPOCH: {epoch + 1}: {train_learning_rate}")

            if epoch % 10 == 0:
                self.sample_image_generation()

            print("-" * 30)

        stop = timeit.default_timer()
        print(f"Training Time: {stop - start:.2f}s")
        self.sample_image_generation()

    def add_noise(self, images, noise, timesteps):
        # Implement the noise addition logic here
        pass

    def sample_image_generation(self):
        # Implement image generation logic here
        pass

if __name__ == "__main__":
    RANDOM_SEED = 42
    IMG_SIZE = 128
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    NUM_GENERATE_IMAGES = 9
    NUM_TIMESTEPS = 1000

    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset_path = "huggan/smithsonian_butterflies_subset"
    butterfly_dataset = ButterflyDataset(dataset_path, img_size=IMG_SIZE)
    train_dataloader = DataLoader(butterfly_dataset, batch_size=BATCH_SIZE, shuffle=True)

    unet_model = UNet2DModel(sample_size=IMG_SIZE, in_channels=3, out_channels=3,
                             layers_per_block=2, block_out_channels=(128, 128, 256, 256, 512, 512),
                             down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D",
                                               "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
                             up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"))

    trainer = ButterflyTrainer(unet_model, train_dataloader, NUM_EPOCHS, LEARNING_RATE, NUM_TIMESTEPS, RANDOM_SEED)
    trainer.train()
