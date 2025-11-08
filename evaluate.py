import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import torch.nn.functional as F
from tqdm import tqdm


class MedicalSegmentationMetrics:
    def __init__(self, num_classes=3, class_names=['WT', 'TC', 'ET']):
        self.num_classes = num_classes
        self.class_names = class_names

    def dice_score(self, pred, target):
        smooth = 1.0
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

    def hausdorff_distance(self, pred, target):
        try:
            pred_np = pred.cpu().numpy().astype(bool)
            target_np = target.cpu().numpy().astype(bool)

            if not np.any(pred_np) or not np.any(target_np):
                return np.inf

            hd1 = directed_hausdorff(pred_np, target_np)[0]
            hd2 = directed_hausdorff(target_np, pred_np)[0]
            return max(hd1, hd2)
        except:
            return np.inf

    def sensitivity_specificity(self, pred, target):
        tp = ((pred == 1) & (target == 1)).sum().float()
        tn = ((pred == 0) & (target == 0)).sum().float()
        fp = ((pred == 1) & (target == 0)).sum().float()
        fn = ((pred == 0) & (target == 1)).sum().float()

        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        return sensitivity, specificity

    def evaluate_model(self, model, dataloader, device):
        model.eval()
        metrics = {
            'dice': {name: [] for name in self.class_names},
            'hausdorff': {name: [] for name in self.class_names},
            'sensitivity': {name: [] for name in self.class_names},
            'specificity': {name: [] for name in self.class_names}
        }

        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating'):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)

                outputs = torch.sigmoid(model(images))
                preds = (outputs > 0.5).float()

                for i, class_name in enumerate(self.class_names):
                    # Dice
                    dice = self.dice_score(preds[:, i], masks[:, i])
                    metrics['dice'][class_name].append(dice.item())

                    # Hausdorff
                    hd = self.hausdorff_distance(preds[:, i], masks[:, i])
                    metrics['hausdorff'][class_name].append(hd)

                    # Sensitivity & Specificity
                    sens, spec = self.sensitivity_specificity(preds[:, i], masks[:, i])
                    metrics['sensitivity'][class_name].append(sens.item())
                    metrics['specificity'][class_name].append(spec.item())

        # Compute averages
        results = {}
        for metric_name, class_metrics in metrics.items():
            results[metric_name] = {}
            for class_name, values in class_metrics.items():
                if metric_name == 'hausdorff':
                    # Filter out infinite values for Hausdorff
                    finite_values = [v for v in values if np.isfinite(v)]
                    results[metric_name][class_name] = np.mean(finite_values) if finite_values else np.inf
                else:
                    results[metric_name][class_name] = np.mean(values)

        return results