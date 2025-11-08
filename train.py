import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import argparse
from tqdm import tqdm

from models import UNet3D, UNet3D_Simple, DeepLabV3Plus3D, DeepLabV3Plus3D_Simple
from data_loader import BratsDataset, BratsTransform
from config import Config


class Trainer:
    def __init__(self, model_name, data_dir, log_dir, val_split=0.2):
        self.cfg = Config()
        self.model_name = model_name
        self.data_dir = data_dir
        self.log_dir = log_dir
        self.val_split = val_split
        self.device = self.cfg.DEVICE

        # Initialize model
        self.model = self._init_model()
        self.model.to(self.device)

        # Loss and optimizer
        self.criterion = self._init_loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5)

        # Data loaders
        self.train_loader, self.val_loader = self._init_data_loaders()

        # Logger
        self.writer = SummaryWriter(log_dir)

    def _init_model(self):
        if self.model_name == 'unet':
            return UNet3D(
                in_channels=self.cfg.IN_CHANNELS,
                out_channels=self.cfg.OUT_CHANNELS,
                features=self.cfg.UNET_FEATURES
            )
        elif self.model_name == 'unet_simple':
            return UNet3D_Simple(
                in_channels=self.cfg.IN_CHANNELS,
                out_channels=self.cfg.OUT_CHANNELS,
                features=[16, 32, 64, 128]
            )
        elif self.model_name == 'deeplab':
            return DeepLabV3Plus3D(
                in_channels=self.cfg.IN_CHANNELS,
                out_channels=self.cfg.OUT_CHANNELS,
                aspp_channels=self.cfg.DEEPLAB_ASPP_CHANNELS
            )
        elif self.model_name == 'deeplab_simple':
            return DeepLabV3Plus3D_Simple(
                in_channels=self.cfg.IN_CHANNELS,
                out_channels=self.cfg.OUT_CHANNELS
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def _init_loss(self):
        class CombinedLoss(nn.Module):
            def __init__(self):
                super(CombinedLoss, self).__init__()
                self.ce = nn.BCEWithLogitsLoss()  # Change to BCE for multi-label

            def dice_loss(self, pred, target):
                smooth = 1.0
                pred = torch.sigmoid(pred)

                # Flatten spatial dimensions
                pred_flat = pred.view(pred.size(0), pred.size(1), -1)
                target_flat = target.view(target.size(0), target.size(1), -1)

                intersection = (pred_flat * target_flat).sum(2)
                union = pred_flat.sum(2) + target_flat.sum(2)

                dice = (2. * intersection + smooth) / (union + smooth)
                return 1 - dice.mean()

            def forward(self, pred, target):
                bce_loss = self.ce(pred, target)
                dice_loss = self.dice_loss(pred, target)
                return bce_loss + dice_loss

        return CombinedLoss()

    def _init_data_loaders(self):
        # Create full dataset
        full_dataset = BratsDataset(self.data_dir, transform=BratsTransform(size=self.cfg.IMAGE_SIZE), mode='train')

        # Split into train and validation
        val_size = int(len(full_dataset) * self.val_split)
        train_size = len(full_dataset) - val_size

        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        # Apply different transforms
        train_dataset.dataset.transform = BratsTransform(size=self.cfg.IMAGE_SIZE, augment=True)
        val_dataset.dataset.transform = BratsTransform(size=self.cfg.IMAGE_SIZE, augment=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.BATCH_SIZE,
            shuffle=True,
            num_workers=2
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=2
        )

        return train_loader, val_loader

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

            # Log batch loss
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('train/batch_loss', loss.item(), global_step)

        return total_loss / len(self.train_loader)

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        self.writer.add_scalar('val/epoch_loss', avg_loss, epoch)

        return avg_loss

    def train(self):
        best_val_loss = float('inf')

        print(f"Training {self.model_name} on {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")

        for epoch in range(self.cfg.NUM_EPOCHS):
            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss = self.validate(epoch)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(epoch, val_loss)

            # Save checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, val_loss)

    def save_model(self, epoch, loss):
        os.makedirs('checkpoints', exist_ok=True)
        model_path = f'checkpoints/{self.model_name}_best_epoch_{epoch}_loss_{loss:.4f}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, model_path)
        print(f"Saved best model to {model_path}")

    def save_checkpoint(self, epoch, loss):
        os.makedirs('checkpoints', exist_ok=True)
        checkpoint_path = f'checkpoints/{self.model_name}_checkpoint_epoch_{epoch}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)


def main():
    parser = argparse.ArgumentParser(description='Train BraTS Segmentation Models')
    parser.add_argument('--model', type=str, required=True,
                        choices=['unet', 'unet_simple', 'deeplab', 'deeplab_simple'],
                        help='Model to train')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to BraTS data directory')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Path to log directory')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')

    args = parser.parse_args()

    trainer = Trainer(args.model, args.data_dir, args.log_dir, args.val_split)
    trainer.train()


if __name__ == '__main__':
    main()