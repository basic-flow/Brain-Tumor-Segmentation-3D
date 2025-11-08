# data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import nibabel as nib
from scipy import ndimage
import random


class BratsDataset(Dataset):
    def __init__(self, data_dir, transform=None, mode='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.samples = self._load_samples()

    def _load_samples(self):
        """Load sample paths from BraTS directory structure"""
        samples = []

        # Get all case directories
        case_dirs = [d for d in os.listdir(self.data_dir)
                     if os.path.isdir(os.path.join(self.data_dir, d))]

        for case_id in case_dirs:
            case_path = os.path.join(self.data_dir, case_id)
            files = os.listdir(case_path)

            # Find modality files (BraTS naming can vary)
            sample = {'case_id': case_id}

            # Look for each modality file
            for mod in ['t1', 't1ce', 't2', 'flair']:
                mod_files = [f for f in files if mod in f.lower() and 'seg' not in f.lower()]
                if mod_files:
                    sample[mod] = os.path.join(case_path, mod_files[0])
                else:
                    print(f"Warning: No {mod} file found in {case_id}")

            # Look for segmentation file (only for training)
            if self.mode != 'test':
                seg_files = [f for f in files if 'seg' in f.lower()]
                if seg_files:
                    sample['seg'] = os.path.join(case_path, seg_files[0])
                else:
                    print(f"Warning: No segmentation file found in {case_id}")

            # Only add sample if all modalities are present
            if all(mod in sample for mod in ['t1', 't1ce', 't2', 'flair']):
                if self.mode == 'test' or 'seg' in sample:
                    samples.append(sample)

        print(f"Loaded {len(samples)} samples from {self.data_dir}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load modalities
        modalities = []
        for mod in ['t1', 't1ce', 't2', 'flair']:
            img = nib.load(sample[mod])
            data = img.get_fdata().astype(np.float32)
            modalities.append(data)

        # Stack modalities [4, H, W, D]
        image = np.stack(modalities, axis=0)

        # Load segmentation mask if available
        if self.mode != 'test':
            seg = nib.load(sample['seg'])
            mask = seg.get_fdata().astype(np.float32)
            # Convert to 3 channels: WT, TC, ET
            mask = self._convert_brats_labels(mask)
        else:
            mask = np.zeros((3, *image.shape[1:]))

        # Apply transforms
        if self.transform:
            image, mask = self.transform(image, mask)

        return {
            'image': torch.FloatTensor(image),
            'mask': torch.FloatTensor(mask),
            'case_id': sample['case_id']
        }

    def _convert_brats_labels(self, mask):
        """Convert BraTS labels to WT, TC, ET"""
        # BraTS labels:
        # 0: background, 1: necrotic, 2: edema, 3: non-enhancing, 4: enhancing

        wt = np.zeros_like(mask)  # Whole Tumor: 1, 2, 3, 4
        tc = np.zeros_like(mask)  # Tumor Core: 1, 3, 4
        et = np.zeros_like(mask)  # Enhancing Tumor: 4

        wt[(mask == 1) | (mask == 2) | (mask == 3) | (mask == 4)] = 1
        tc[(mask == 1) | (mask == 3) | (mask == 4)] = 1
        et[mask == 4] = 1

        # Stack along channel dimension [3, H, W, D]
        return np.stack([wt, tc, et], axis=0)


class BratsTransform:
    def __init__(self, size=(128, 128, 128), augment=False):
        self.size = size
        self.augment = augment

    def __call__(self, image, mask):
        # Basic preprocessing
        image = self.normalize_modalities(image)

        # Resize - FIXED VERSION
        image = self.resize_volume(image, self.size, is_mask=False)
        mask = self.resize_volume(mask, self.size, is_mask=True)

        # Data augmentation
        if self.augment:
            image, mask = self.augment_data(image, mask)

        return image, mask

    def normalize_modalities(self, image):
        """Normalize each modality independently using robust normalization"""
        for i in range(image.shape[0]):
            modality = image[i]
            # Use non-zero voxels for normalization (common in medical imaging)
            non_zero_mask = modality > 0
            if non_zero_mask.any():
                non_zero_values = modality[non_zero_mask]
                mean = non_zero_values.mean()
                std = non_zero_values.std()
                modality = (modality - mean) / (std + 1e-8)
            image[i] = modality
        return image

    def resize_volume(self, volume, target_size, is_mask=False):
        """Resize 3D volume to target size - FIXED VERSION"""
        # Get current dimensions
        current_depth, current_height, current_width = volume.shape[1], volume.shape[2], volume.shape[3]
        target_depth, target_height, target_width = target_size

        # Calculate resize factors for spatial dimensions only
        depth_factor = target_depth / current_depth
        height_factor = target_height / current_height
        width_factor = target_width / current_width

        # Create zoom factors array matching input dimensions
        # For 4D input (channels, depth, height, width): [1, depth_factor, height_factor, width_factor]
        # For 3D input (channels, depth, height, width): [1, depth_factor, height_factor, width_factor]
        zoom_factors = [1, depth_factor, height_factor, width_factor]

        # Use appropriate interpolation order
        order = 0 if is_mask else 1

        # Apply zoom
        resized = ndimage.zoom(volume, zoom_factors, order=order)
        return resized

    def augment_data(self, image, mask):
        """Simple 3D data augmentation"""
        # Random flip
        if random.random() > 0.5:
            axis = random.choice([2, 3, 4])  # Spatial axes (channels is axis 1)
            image = np.flip(image, axis=axis).copy()
            mask = np.flip(mask, axis=axis).copy()

        # Random rotation (0, 90, 180, 270 degrees) in axial plane
        if random.random() > 0.5:
            angle = random.choice([0, 90, 180, 270])
            # Rotate in height-width plane (axes 2 and 3)
            for i in range(image.shape[0]):  # For each channel in image
                image[i] = ndimage.rotate(image[i], angle, axes=(1, 2), reshape=False, order=1)
            for i in range(mask.shape[0]):  # For each channel in mask
                mask[i] = ndimage.rotate(mask[i], angle, axes=(1, 2), reshape=False, order=0)

        # Random intensity shift
        if random.random() > 0.5:
            shift = random.uniform(-0.1, 0.1)
            image = image + shift

        return image, mask