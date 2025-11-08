# config.py
import torch


class Config:
    # Model settings
    IN_CHANNELS = 4  # T1, T1ce, T2, FLAIR
    OUT_CHANNELS = 3  # WT, TC, ET

    # Training settings
    BATCH_SIZE = 1  # Reduced from 2 to 1 for 3D data
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data settings
    IMAGE_SIZE = (256, 256, 256)  # Common size for BraTS

    # Model specific settings
    UNET_FEATURES = [32, 64, 128, 256]
    DEEPLAB_ASPP_CHANNELS = 256