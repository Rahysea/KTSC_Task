# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings

warnings.filterwarnings("ignore")
import logging
import os
import sys
import json
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import sklearn.metrics as metrics
import monai
import argparse
import random
import pandas as pd
import csv
import time
from torchvision.utils import make_grid
from torch.utils.data import WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from monai.data import decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric
from monai.transforms import (
    Activations, AsDiscrete, Compose, LoadImaged, RandRotate90d,
    Resized, ScaleIntensityd, NormalizeIntensityd, RandFlipd, RandGaussianNoised
)

'''
Binary classification data file format (modify dataload function as needed):
filename (case_name)  ground_truth  split_name
111 0 train
222 1 validation  
333 1 test
'''


def parse_arguments():
    """Parse command line arguments for the training script."""
    parser = argparse.ArgumentParser()

    # Essential parameters (modify for different tasks)
    parser.add_argument('--experiment_suffix', type=str, default='test-20250803-batch-192-3',
                        help="Custom suffix for experiment name (empty by default)")
    parser.add_argument('--architecture', type=str, default="monai.DenseNet121_EMA",
                        help="Model architecture: monai.DenseNet121, monai.DenseNet121_EMA, monai.HighResNet, monai.SEResNet101, "
                             "monai.SEResNet152, monai.SEResNet18, monai.SEResNet34, monai.SEResNet50, "
                             "monai.SEResNeXt101, monai.SEResNeXt50")
    parser.add_argument('--root_dir', type=str, default="")
    parser.add_argument('--save_dir', type=str, default=r"")
    parser.add_argument('--data_file', type=str, default=r"")

    # Data configuration (modify if using non-standard format)
    parser.add_argument('--split_column', type=str, default='split')
    parser.add_argument('--case_column', type=str, default='filename')
    parser.add_argument('--split_names', type=list, default=['Train', 'Val', 'Test'],
                        help="Split names, default: ['Train', 'Val', 'Test']")
    parser.add_argument('--images_folder_name', type=str, default='images')

    # Commonly modified parameters
    parser.add_argument('--gpu_id', type=str, default='0,1,2,3,4,5,6,7,8')
    parser.add_argument('--T_max', type=int, default=10,
                        help="CosineAnnealingLR period, if -1, uses max_epoch / 10")
    parser.add_argument('--default_threshold', type=float, default=-1,
                        help="Classification threshold, default 0.5, if -1 auto-calculates optimal threshold")
    parser.add_argument('--loss_function', type=str, default='CrossEntropyLoss',
                        help="Loss function: BCEWithLogitsLoss (binary), CrossEntropyLoss, "
                             "MultiLabelSoftMarginLoss (multi-label)")
    parser.add_argument('--multimodal', type=bool, default=False, help="Whether using multimodal data")
    parser.add_argument('--modality_names', type=list, default=['DCE_PVP'],
                        help="Multimodal data names")
    parser.add_argument('--binary_classification', type=bool, default=True, help="Whether binary classification")
    parser.add_argument('--binary_label_column', type=str, default='ground_truth',
                        help="Binary label column name")
    parser.add_argument('--multi_class_columns', type=list, default=['class_1', 'class_2'],
                        help="Multi-class label columns for one-hot encoding")
    parser.add_argument('--in_channels', type=int, default=1, help="Input channels")
    parser.add_argument('--out_channels', type=int, default=2, help="Output channels")
    parser.add_argument('--batch_size', type=int, default=12, help="Batch size")
    parser.add_argument('--num_workers', type=int, default=2, help="Threads for loading data")
    parser.add_argument('--balancing_strategy', type=str, default='balance',
                        help='balance: class balance weights, balance_pn: positive-negative balance, '
                             'balance_enhance: combined positive-negative and class balance. '
                             'Binary classification can only use balance')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--test_set_available', type=bool, default=True,
                        help="Whether test set is available")
    parser.add_argument('--checkpoint_path', type=str, default="",
                        help="Path to checkpoint for inference")

    # Rarely modified parameters
    parser.add_argument('--validation_interval', type=int, default=5, help="Epochs between validation")
    parser.add_argument('--max_epochs', type=int, default=2000, help="Maximum epochs")
    parser.add_argument('--spatial_dims', type=int, default=3, help="Data dimensions")
    parser.add_argument('--spatial_size_h', type=int, default=192)
    parser.add_argument('--spatial_size_w', type=int, default=192)
    parser.add_argument('--spatial_size_d', type=int, default=192)
    parser.add_argument('--optimizer', default="AdamW", help="Optimizer: Adam, SGD, AdamW")
    parser.add_argument('--scheduler', default="CosineAnnealingLR",
                        help="Scheduler: CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR")
    parser.add_argument('--seed', type=int, default=2025, help="Random seed for python, numpy and pytorch")

    # Data augmentation
    parser.add_argument('--rotation_probability', type=float, default=0.8, help="Rotation probability")

    return parser.parse_args()


def save_args_to_json(args, file_path):
    """
    Save command line arguments to JSON file.

    Args:
        args: argparse.Namespace, command line arguments
        file_path: str, JSON file path to save
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w') as f:
        json.dump(vars(args), f, indent=4)


def apply_sigmoid_threshold(x, threshold=0.5):
    """Apply sigmoid threshold to convert probability to binary classification."""
    return 1 if x >= threshold else 0


def calculate_binary_metrics(labels, predictions, custom_threshold=None):
    """
    Calculate metrics for binary classification only.

    Args:
        labels: 0/1 ground truth labels
        predictions: probability of class 1
        custom_threshold: manual threshold, otherwise uses Youden index from ROC

    Returns:
        [Accuracy, Sensitivity, Specificity, AUC, best_threshold]
    """
    true_labels = labels
    pred_values = predictions
    num_patients = len(true_labels)

    # Calculate AUC and optimal threshold
    if sum(true_labels) == num_patients or sum(true_labels) == 0:
        auc_score = 0
        print('Warning: Only one class present in data')
    else:
        auc_score = metrics.roc_auc_score(true_labels, pred_values)

    # Calculate optimal threshold using Youden index
    fpr, tpr, thresholds = metrics.roc_curve(true_labels, pred_values)
    youden_index = tpr + (1 - fpr)
    best_threshold = thresholds[youden_index == np.max(youden_index)][0]

    # Handle edge case where threshold > 1
    if best_threshold > 1:
        best_threshold = 0.5

    # Use custom threshold if provided
    if custom_threshold is not None:
        best_threshold = custom_threshold

    # Calculate confusion matrix components
    tp = tn = fp = fn = 0

    for i in range(num_patients):
        true_label = true_labels[i]
        pred_prob = pred_values[i]
        pred_label = apply_sigmoid_threshold(pred_prob, best_threshold)

        if true_label == 1 and true_label == pred_label:
            tp += 1  # True positive
        elif true_label == 0 and true_label == pred_label:
            tn += 1  # True negative
        elif true_label == 1 and pred_label == 0:
            fn += 1  # False negative
        elif true_label == 0 and pred_label == 1:
            fp += 1  # False positive

    epsilon = 1e-16  # Avoid division by zero
    sensitivity = tp / (tp + fn + epsilon)
    specificity = tn / (tn + fp + epsilon)
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)

    return [accuracy, sensitivity, specificity, auc_score, best_threshold]


def calculate_multiclass_thresholds(true_labels, predictions):
    """
    Calculate optimal thresholds for multi-label classification.

    Args:
        true_labels: ground truth labels
        predictions: predicted probabilities

    Returns:
        List of optimal thresholds per class
    """
    optimal_thresholds = []
    for i in range(predictions.shape[1]):
        fpr, tpr, thresholds = metrics.roc_curve(true_labels[:, i], predictions[:, i])
        youden_index = tpr + (1 - fpr)
        optimal_threshold = thresholds[youden_index == np.max(youden_index)]
        if len(optimal_threshold) > 0:
            optimal_thresholds.append(optimal_threshold[0])
        else:
            optimal_thresholds.append(0.5)  # Default threshold
    return optimal_thresholds


def convert_probabilities_to_binary(true_labels, predictions, device):
    """
    Convert probabilities to binary using thresholds for multi-label classification.

    Args:
        true_labels: ground truth labels
        predictions: predicted probabilities
        device: computation device

    Returns:
        Binary predictions and thresholds used
    """
    thresholds = calculate_multiclass_thresholds(true_labels.cpu().numpy(), predictions.cpu().numpy())
    binary_predictions = torch.zeros_like(predictions)

    for i in range(predictions.shape[1]):
        binary_predictions[:, i] = (predictions[:, i] > thresholds[i]).float()

    return binary_predictions.to(device), thresholds


def set_random_seed(seed, worker_id):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True


def custom_collate_function(batch):
    """
    Custom collate function for batch processing.

    Args:
        batch: batch data

    Returns:
        Collated batch data
    """
    images = [item['img'] for item in batch]
    labels = [item['label'] for item in batch]
    names = [item['name'] for item in batch]

    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.float32)

    return {'img': images, 'label': labels, 'name': names}


def load_data(args, annotation_df, annotation_list):
    """
    Load data based on arguments.

    Args:
        args: configuration arguments
        annotation_df: data file dataframe
        annotation_list: list of cases to load

    Returns:
        image_list, label_list, name_list
    """
    image_list = []
    label_list = []
    name_list = []

    def get_image_paths(case_name, is_multimodal, modality_names):
        """Get image paths for a case, handling multimodal data."""
        image_paths = []
        modalities = modality_names if is_multimodal else [case_name]

        for modality in modalities:
            img_path = f'{args.root_dir}/{args.images_folder_name}/{case_name}/{modality}.nii.gz'
            if os.path.exists(img_path):
                image_paths.append(img_path)
            else:
                print(f'Warning: {img_path} not found!')
        return image_paths

    for case in annotation_list:
        # Get image paths
        image_list.append(get_image_paths(case, args.multimodal, args.modality_names))

        # Get labels
        if args.binary_classification:
            label_value = int(
                annotation_df[annotation_df[args.case_column] == case][args.binary_label_column].values[0])
            label_list.append(label_value)
        else:
            label_values = [int(annotation_df[annotation_df[args.case_column] == case][label].values[0])
                            for label in args.multi_class_columns]
            label_list.append(label_values)

        name_list.append(case)

    return image_list, label_list, name_list


def calculate_class_accuracies(predictions, true_labels):
    """
    Calculate accuracy for each class in multi-class classification.

    Args:
        predictions: binary predictions
        true_labels: ground truth labels

    Returns:
        accuracies: list of accuracies per class
        mean_accuracy: average accuracy
    """
    accuracies = []
    for i in range(predictions.shape[1]):
        correct = (predictions[:, i] == true_labels[:, i]).sum().item()
        total = predictions.shape[0]
        accuracy = correct / total if total > 0 else 0
        accuracies.append(accuracy)

    mean_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
    return accuracies, mean_accuracy


def calculate_sensitivity_specificity(predictions, true_labels):
    """
    Calculate sensitivity and specificity for each class.

    Args:
        predictions: binary predictions
        true_labels: ground truth labels

    Returns:
        sensitivities: list of sensitivities per class
        specificities: list of specificities per class
    """
    sensitivities = []
    specificities = []

    for i in range(predictions.shape[1]):
        true_positives = ((predictions[:, i] == 1) & (true_labels[:, i] == 1)).sum().item()
        true_negatives = ((predictions[:, i] == 0) & (true_labels[:, i] == 0)).sum().item()
        false_positives = ((predictions[:, i] == 1) & (true_labels[:, i] == 0)).sum().item()
        false_negatives = ((predictions[:, i] == 0) & (true_labels[:, i] == 1)).sum().item()

        epsilon = 1e-16
        sensitivity = true_positives / (true_positives + false_negatives + epsilon)
        specificity = true_negatives / (true_negatives + false_positives + epsilon)

        sensitivities.append(sensitivity)
        specificities.append(specificity)

    return sensitivities, specificities


def log_metrics_to_csv_and_tensorboard(epoch, accuracy_metrics, sensitivity, specificity, auc,
                                       mean_accuracy, writer, phase, csv_file_path, components=None):
    """
    Log metrics to CSV file and TensorBoard.

    Args:
        epoch: current epoch number
        accuracy_metrics: accuracy values
        sensitivity: sensitivity values
        specificity: specificity values
        auc: AUC value
        mean_accuracy: mean accuracy
        writer: TensorBoard SummaryWriter
        phase: phase name (train/val/test)
        csv_file_path: path to save CSV file
        components: list of component names for multi-class
    """
    csv_file_path = os.path.join(csv_file_path, f"{phase}_metrics.csv")

    # Binary classification handling
    if isinstance(accuracy_metrics, (int, float)):
        if not os.path.exists(csv_file_path):
            with open(csv_file_path, mode='w', newline='') as file:
                writer_csv = csv.writer(file)
                writer_csv.writerow([
                    'Epoch', f'{phase}_accuracy', f'{phase}_sensitivity',
                    f'{phase}_specificity', f'{phase}_auc', f'mean_{phase}_accuracy'
                ])

        with open(csv_file_path, mode='a', newline='') as file:
            writer_csv = csv.writer(file)
            writer_csv.writerow([
                epoch + 1, accuracy_metrics, sensitivity, specificity, auc, mean_accuracy
            ])

        # TensorBoard logging
        writer.add_scalar(f"{phase}/accuracy", accuracy_metrics, epoch + 1)
        writer.add_scalar(f"{phase}/sensitivity", sensitivity, epoch + 1)
        writer.add_scalar(f"{phase}/specificity", specificity, epoch + 1)
        writer.add_scalar(f"{phase}/auc", auc, epoch + 1)
        writer.add_scalar(f"{phase}/mean_accuracy", mean_accuracy, epoch + 1)

    # Multi-class classification handling
    else:
        if components is None:
            raise ValueError("Components must be provided for multi-class classification")

        if not os.path.exists(csv_file_path):
            with open(csv_file_path, mode='w', newline='') as file:
                writer_csv = csv.writer(file)
                headers = ['Epoch']
                for component in components:
                    headers.extend([
                        f'{phase}_accuracy[{component}]',
                        f'{phase}_sensitivity[{component}]',
                        f'{phase}_specificity[{component}]'
                    ])
                headers.extend([f'{phase}_auc', f'mean_{phase}_accuracy'])
                writer_csv.writerow(headers)

        with open(csv_file_path, mode='a', newline='') as file:
            writer_csv = csv.writer(file)
            row = [epoch + 1]
            row.extend(accuracy_metrics)
            row.extend(sensitivity)
            row.extend(specificity)
            row.extend([auc, mean_accuracy])
            writer_csv.writerow(row)

        # TensorBoard logging
        for idx, component in enumerate(components):
            writer.add_scalar(f"{phase}/accuracy_{component}", accuracy_metrics[idx], epoch + 1)
            writer.add_scalar(f"{phase}/sensitivity_{component}", sensitivity[idx], epoch + 1)
            writer.add_scalar(f"{phase}/specificity_{component}", specificity[idx], epoch + 1)
        writer.add_scalar(f"{phase}/auc", auc, epoch + 1)
        writer.add_scalar(f"{phase}/mean_accuracy", mean_accuracy, epoch + 1)


def save_hyperparameters(writer, args):
    """
    Save hyperparameters to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter
        args: configuration arguments
    """
    args_dict = vars(args)
    param_info = "<br>".join([f"{key}: {value}" for key, value in args_dict.items()])
    writer.add_text('Hyperparameters', param_info)


def create_balanced_sampler(args, files):
    """
    Create balanced sampler for handling class imbalance.

    Args:
        args: configuration arguments
        files: list of data files

    Returns:
        WeightedRandomSampler or None
    """
    if args.binary_classification:
        if args.balancing_strategy == 'balance':
            labels = [item["label"] for item in files]
            class_counts = [labels.count(c) for c in set(labels)]
            class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
            sample_weights = class_weights[labels]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
        else:
            sampler = None
    else:
        labels = [item["label"] for item in files]
        labels_tensor = torch.tensor(labels, dtype=torch.float)

        if args.balancing_strategy == 'balance':
            class_counts = labels_tensor.sum(dim=0)
            class_weights = 1.0 / (class_counts + 1e-6)
            sample_weights = torch.sum(labels_tensor * class_weights, dim=1)

        elif args.balancing_strategy == 'balance_pn':
            class_pos_counts = labels_tensor.sum(dim=0)
            class_neg_counts = labels_tensor.size(0) - class_pos_counts
            pos_weights = 1.0 / (class_pos_counts + 1e-6)
            neg_weights = 1.0 / (class_neg_counts + 1e-6)

            sample_weights = torch.zeros(labels_tensor.size(0))
            for i in range(labels_tensor.size(1)):
                sample_weights += (labels_tensor[:, i] * pos_weights[i] +
                                   (1 - labels_tensor[:, i]) * neg_weights[i])

        elif args.balancing_strategy == 'balance_enhance':
            class_counts = labels_tensor.sum(dim=0)
            class_weights = 1.0 / (class_counts + 1e-6)
            class_pos_counts = labels_tensor.sum(dim=0)
            class_neg_counts = labels_tensor.size(0) - class_pos_counts
            pos_weights = 1.0 / (class_pos_counts + 1e-6)
            neg_weights = 1.0 / (class_neg_counts + 1e-6)

            sample_weights = torch.zeros(labels_tensor.size(0))
            for i in range(labels_tensor.size(1)):
                sample_weights += ((labels_tensor[:, i] * pos_weights[i] +
                                    (1 - labels_tensor[:, i]) * neg_weights[i]) *
                                   class_weights[i])
        else:
            sampler = None
            return sampler

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

    return sampler


def validate_arguments(args):
    """
    Validate configuration arguments for consistency.

    Args:
        args: configuration arguments

    Raises:
        ValueError: if arguments are inconsistent
    """
    # Check multimodal data and input channels match
    if args.multimodal and args.in_channels != len(args.modality_names):
        raise ValueError(
            f"Input channels ({args.in_channels}) don't match number of modalities ({len(args.modality_names)})"
        )

    # Warn if inference mode with batch size != 1
    if args.checkpoint_path and args.batch_size != 1:
        warnings.warn("In inference mode, batch_size should be 1 for reliable results")

    # Binary classification can only use 'balance' strategy
    if args.binary_classification and args.balancing_strategy not in ['balance', '']:
        warnings.warn("Binary classification should use 'balance' strategy or none")


def log_images_to_tensorboard(writer, phase, images, labels, names, epoch):
    """
    Log images and corresponding labels to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter
        phase: phase name (train/val/test)
        images: image tensor
        labels: label tensor
        names: patient names
        epoch: current epoch
    """
    try:
        # Create image grid (assuming depth is last dimension)
        if len(images.shape) == 5:  # [batch, channels, height, width, depth]
            grid = make_grid(images[:, :, :, :, images.shape[4] // 2], nrow=4, padding=2, normalize=True)
        else:
            grid = make_grid(images, nrow=4, padding=2, normalize=True)

        writer.add_image(f"images/{phase}_epoch_{epoch + 1}", grid, epoch + 1)

        # Log labels and names as text
        labels_str = [str(label.tolist()) for label in labels]
        labels_names_str = [f"Patient: {name}, Label: {label}" for name, label in zip(names, labels_str)]
        labels_text = "<br>".join(labels_names_str)
        writer.add_text(f"images/{phase}_labels_epoch_{epoch + 1}", labels_text, epoch + 1)

    except Exception as e:
        print(f"Warning: Could not log images to TensorBoard: {e}")


def select_model_architecture(args):
    """
    Select and initialize model based on architecture argument.

    Args:
        args: configuration arguments

    Returns:
        Initialized model

    Raises:
        ValueError: if architecture is not supported
    """
    architecture_map = {
        'monai.DenseNet121': monai.networks.nets.DenseNet121,
        'monai.DenseNet121_EMA': monai.networks.nets.DenseNet121_EMA,
        'monai.HighResNet': monai.networks.nets.HighResNet,
        'monai.SEResNet101': monai.networks.nets.SEResNet101,
        'monai.SEResNet152': monai.networks.nets.SEResNet152,
    }

    if args.architecture not in architecture_map:
        raise ValueError(f"Unsupported architecture: {args.architecture}")

    model_class = architecture_map[args.architecture]

    if args.architecture == 'monai.DenseNet121' or 'monai.DenseNet121_EMA':
        model = model_class(
            spatial_dims=args.spatial_dims,
            in_channels=args.in_channels,
            out_channels=args.out_channels
        )
    else:
        model = model_class(
            spatial_dims=args.spatial_dims,
            in_channels=args.in_channels,
            num_classes=args.out_channels
        )

    print(f"Using {args.architecture} model")
    return model


def main():
    """Main training and evaluation function."""
    args = parse_arguments()
    validate_arguments(args)
    set_random_seed(args.seed, 0)

    # Calculate T_max for scheduler
    T_max = args.T_max if args.T_max != -1 else int(args.max_epochs / 10)

    # Adjust parameters for inference mode
    if args.checkpoint_path:
        args.batch_size = 1
        args.num_workers = 1
        args.validation_interval = 1
        args.max_epochs = 1

    # Create experiment name
    experiment_time = time.strftime("%Y%m%d", time.localtime())
    experiment_name = (f"{experiment_time}-{args.architecture}-{args.balancing_strategy}-"
                       f"{args.loss_function}-{args.spatial_size_d}-{args.spatial_size_h}-"
                       f"{args.spatial_size_w}-{args.rotation_probability}-{T_max}-"
                       f"{args.experiment_suffix}")

    # Create save directory and save arguments
    save_path = os.path.join(args.save_dir, experiment_name)
    os.makedirs(save_path, exist_ok=True)
    save_args_to_json(args, os.path.join(save_path, "args.json"))

    # Setup logging
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Setup device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    annotation_df = pd.read_excel(args.data_file)

    # Training set
    train_cases = annotation_df[annotation_df[args.split_column] == args.split_names[0]][args.case_column].tolist()
    train_images, train_labels, train_names = load_data(args, annotation_df, train_cases)

    # Validation set
    val_cases = annotation_df[annotation_df[args.split_column] == args.split_names[1]][args.case_column].tolist()
    val_images, val_labels, val_names = load_data(args, annotation_df, val_cases)

    # Test set (if available)
    if args.test_set_available:
        test_cases = annotation_df[annotation_df[args.split_column] == args.split_names[2]][args.case_column].tolist()
        test_images, test_labels, test_names = load_data(args, annotation_df, test_cases)

    # Create file lists
    train_files = [{"img": img, "label": label, "name": name}
                   for img, label, name in zip(train_images, train_labels, train_names)]
    val_files = [{"img": img, "label": label, "name": name}
                 for img, label, name in zip(val_images, val_labels, val_names)]
    if args.test_set_available:
        test_files = [{"img": img, "label": label, "name": name}
                      for img, label, name in zip(test_images, test_labels, test_names)]

    # Define transforms
    train_transforms = Compose([
        LoadImaged(keys=["img"], ensure_channel_first=True),
        ScaleIntensityd(keys=["img"]),
        Resized(keys=["img"], spatial_size=(args.spatial_size_h, args.spatial_size_w, args.spatial_size_d)),
        RandRotate90d(keys=["img"], prob=args.rotation_probability, spatial_axes=[0, 1]),
        RandFlipd(keys=["img"], spatial_axis=[0, 1], prob=0.8),
        RandGaussianNoised(keys=["img"], prob=0.8, mean=0.0, std=0.1)
    ])

    val_transforms = Compose([
        LoadImaged(keys=["img"], ensure_channel_first=True),
        ScaleIntensityd(keys=["img"]),
        Resized(keys=["img"], spatial_size=(args.spatial_size_h, args.spatial_size_w, args.spatial_size_d)),
    ])

    # Create balanced sampler
    sampler = create_balanced_sampler(args, train_files)

    # Create data loaders
    train_dataset = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler if not args.checkpoint_path else None,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=custom_collate_function
    )

    val_dataset = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=custom_collate_function
    )

    if args.test_set_available:
        test_dataset = monai.data.Dataset(data=test_files, transform=val_transforms)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=custom_collate_function
        )

    # Initialize model
    model = select_model_architecture(args).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")

    # Load checkpoint for inference
    if args.checkpoint_path:
        try:
            checkpoint = torch.load(args.checkpoint_path, map_location=device)
            if isinstance(model, torch.nn.DataParallel):
                model.module.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            print(f"Successfully loaded model from {args.checkpoint_path}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return

    # Loss function
    loss_functions = {
        'CrossEntropyLoss': torch.nn.CrossEntropyLoss(),
        'BCEWithLogitsLoss': torch.nn.BCEWithLogitsLoss(),
        'MultiLabelSoftMarginLoss': torch.nn.MultiLabelSoftMarginLoss()
    }
    loss_function = loss_functions.get(args.loss_function, torch.nn.CrossEntropyLoss())
    print(f"Using {args.loss_function} loss function")

    # Optimizer
    optimizers = {
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD,
        "AdamW": torch.optim.AdamW
    }
    optimizer_class = optimizers.get(args.optimizer, torch.optim.AdamW)
    optimizer = optimizer_class(model.parameters(), args.learning_rate)
    print(f"Using {args.optimizer} optimizer")

    # Learning rate scheduler
    if args.scheduler == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-6)
    elif args.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=1e-6)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(args.max_epochs / 10),
            gamma=0.1
        )
    print(f"Using {args.scheduler} scheduler")

    # Metrics
    auc_metric = ROCAUCMetric()

    # Training setup
    best_metric = -1
    best_auc = -1
    best_auc_epoch = -1

    if args.test_set_available:
        best_metric_test = -1
        best_auc_test = -1
        best_auc_epoch_test = -1

    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'runs', experiment_name))
    save_hyperparameters(writer, args)

    # Training loop
    for epoch in range(args.max_epochs):
        print("-" * 50)
        print(f"Epoch {epoch + 1}/{args.max_epochs}")

        # Inference mode (no training)
        if args.checkpoint_path:
            model.eval()
            with torch.no_grad():
                all_predictions = torch.tensor([], dtype=torch.float32, device=device)
                all_labels = torch.tensor([], dtype=torch.long, device=device)
                all_names = []

                for train_data in train_loader:
                    images, labels = train_data["img"].to(device), train_data["label"].to(device)
                    outputs = model(images)
                    all_predictions = torch.cat([all_predictions, outputs], dim=0)
                    all_labels = torch.cat([all_labels, labels], dim=0)
                    all_names.extend(train_data["name"])

                if args.binary_classification:
                    probabilities = torch.sigmoid(all_predictions)[:, 1]
                    binary_predictions = (
                                probabilities >= args.default_threshold).int() if args.default_threshold != -1 else (
                                probabilities >= 0.5).int()
                    accuracy, sensitivity, specificity, auc_score, threshold = calculate_binary_metrics(
                        all_labels.cpu().numpy(), probabilities.cpu().numpy()
                    )
                    mean_accuracy = accuracy
                else:
                    probabilities = torch.sigmoid(all_predictions)
                    if args.default_threshold != -1:
                        binary_predictions = (probabilities >= args.default_threshold).int()
                    else:
                        binary_predictions, threshold = convert_probabilities_to_binary(
                            all_labels, probabilities, device
                        )
                    class_accuracies, mean_accuracy = calculate_class_accuracies(binary_predictions, all_labels)
                    sensitivities, specificities = calculate_sensitivity_specificity(binary_predictions, all_labels)
                    auc_metric(binary_predictions, all_labels)
                    auc_score = auc_metric.aggregate("macro")
                    auc_metric.reset()

                print(f"Train - Accuracy: {mean_accuracy:.4f}, AUC: {auc_score:.4f}")

                # Save predictions
                results_df = pd.DataFrame({
                    "name": all_names,
                    "ground_truth": all_labels.cpu().numpy().flatten(),
                    "predicted_probability": probabilities.cpu().numpy().flatten() if args.binary_classification else [
                        str(p) for p in probabilities.cpu().numpy()],
                    "prediction": binary_predictions.cpu().numpy().flatten()
                })
                results_df.to_excel(os.path.join(save_path, f"predictions_epoch_{epoch + 1}_train.xlsx"), index=False)

            print("Inference mode - skipping training")
            break

        # Training mode
        else:
            model.train()
            epoch_loss = 0
            step = 0
            all_names = []

            for batch_data in train_loader:
                step += 1
                images, labels = batch_data["img"].to(device), batch_data["label"].to(device)
                all_names.extend(batch_data["name"])

                optimizer.zero_grad()
                outputs = model(images)

                # Calculate loss
                if args.loss_function == 'BCEWithLogitsLoss' and args.binary_classification:
                    loss = loss_function(outputs[:, 1].float(), labels.float())
                else:
                    loss = loss_function(outputs, labels.long() if args.binary_classification else labels)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                epoch_len = len(train_dataset) // train_loader.batch_size
                if step % 10 == 0:  # Print every 10 steps
                    print(f"Step {step}/{epoch_len}, Loss: {loss.item():.4f}")

                writer.add_scalar("train/loss_step", loss.item(), epoch * epoch_len + step)

            # Update learning rate
            scheduler.step()
            epoch_loss /= step
            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else \
            scheduler.optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch + 1} - Average Loss: {epoch_loss:.4f}, Learning Rate: {current_lr:.6f}")
            writer.add_scalar("train/loss_epoch", epoch_loss, epoch + 1)
            writer.add_scalar("train/learning_rate", current_lr, epoch + 1)

            # Log training images (for binary classification and single modality)
            if (epoch + 1) % args.validation_interval == 0 and not args.multimodal and args.binary_classification:
                log_images_to_tensorboard(writer, "train", images, labels, all_names, epoch)

        # Validation
        if (epoch + 1) % args.validation_interval == 0:
            model.eval()
            with torch.no_grad():
                all_predictions = torch.tensor([], dtype=torch.float32, device=device)
                all_labels = torch.tensor([], dtype=torch.long, device=device)
                all_names = []

                for val_data in val_loader:
                    images, labels = val_data["img"].to(device), val_data["label"].to(device)
                    outputs = model(images)
                    all_predictions = torch.cat([all_predictions, outputs], dim=0)
                    all_labels = torch.cat([all_labels, labels], dim=0)
                    all_names.extend(val_data["name"])

                # Calculate metrics
                if args.binary_classification:
                    probabilities = torch.sigmoid(all_predictions)[:, 1]
                    binary_predictions = (
                                probabilities >= args.default_threshold).int() if args.default_threshold != -1 else (
                                probabilities >= 0.5).int()
                    accuracy, sensitivity, specificity, auc_score, threshold = calculate_binary_metrics(
                        all_labels.cpu().numpy(), probabilities.cpu().numpy()
                    )
                    mean_accuracy = accuracy
                    log_metrics_to_csv_and_tensorboard(
                        epoch, accuracy, sensitivity, specificity, auc_score, mean_accuracy,
                        writer, "val", save_path
                    )
                else:
                    probabilities = torch.sigmoid(all_predictions)
                    if args.default_threshold != -1:
                        binary_predictions = (probabilities >= args.default_threshold).int()
                    else:
                        binary_predictions, threshold = convert_probabilities_to_binary(
                            all_labels, probabilities, device
                        )
                    class_accuracies, mean_accuracy = calculate_class_accuracies(binary_predictions, all_labels)
                    sensitivities, specificities = calculate_sensitivity_specificity(binary_predictions, all_labels)
                    auc_metric(binary_predictions, all_labels)
                    auc_score = auc_metric.aggregate("macro")
                    auc_metric.reset()

                    log_metrics_to_csv_and_tensorboard(
                        epoch, class_accuracies, sensitivities, specificities, auc_score, mean_accuracy,
                        writer, "val", save_path, args.multi_class_columns
                    )

                # Save model checkpoint
                torch.save(model.state_dict(), os.path.join(save_path, f"epoch_{epoch + 1}.pth"))

                # Update best metrics
                if mean_accuracy > best_metric:
                    best_metric = mean_accuracy
                if auc_score > best_auc:
                    best_auc = auc_score
                    best_auc_epoch = epoch + 1
                    # Save best model
                    torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))

                print(f"Validation - Accuracy: {mean_accuracy:.4f}, AUC: {auc_score:.4f}")

                # Save validation predictions
                results_df = pd.DataFrame({
                    "name": all_names,
                    "ground_truth": all_labels.cpu().numpy().flatten(),
                    "predicted_probability": probabilities.cpu().numpy().flatten() if args.binary_classification else [
                        str(p) for p in probabilities.cpu().numpy()],
                    "prediction": binary_predictions.cpu().numpy().flatten()
                })
                results_df.to_excel(os.path.join(save_path, f"predictions_epoch_{epoch + 1}_val.xlsx"), index=False)

                # Log validation images
                if not args.multimodal and args.binary_classification:
                    log_images_to_tensorboard(writer, "val", images, labels, all_names, epoch)

        # Test set evaluation
        if args.test_set_available and (epoch + 1) % args.validation_interval == 0:
            model.eval()
            with torch.no_grad():
                all_predictions = torch.tensor([], dtype=torch.float32, device=device)
                all_labels = torch.tensor([], dtype=torch.long, device=device)
                all_names = []

                for test_data in test_loader:
                    images, labels = test_data["img"].to(device), test_data["label"].to(device)
                    outputs = model(images)
                    all_predictions = torch.cat([all_predictions, outputs], dim=0)
                    all_labels = torch.cat([all_labels, labels], dim=0)
                    all_names.extend(test_data["name"])

                # Calculate test metrics
                if args.binary_classification:
                    probabilities = torch.sigmoid(all_predictions)[:, 1]
                    binary_predictions = (
                                probabilities >= args.default_threshold).int() if args.default_threshold != -1 else (
                                probabilities >= 0.5).int()
                    accuracy, sensitivity, specificity, auc_score, _ = calculate_binary_metrics(
                        all_labels.cpu().numpy(), probabilities.cpu().numpy()
                    )
                    mean_accuracy = accuracy
                    log_metrics_to_csv_and_tensorboard(
                        epoch, accuracy, sensitivity, specificity, auc_score, mean_accuracy,
                        writer, "test", save_path
                    )
                else:
                    probabilities = torch.sigmoid(all_predictions)
                    if args.default_threshold != -1:
                        binary_predictions = (probabilities >= args.default_threshold).int()
                    else:
                        binary_predictions, threshold = convert_probabilities_to_binary(
                            all_labels, probabilities, device
                        )
                    class_accuracies, mean_accuracy = calculate_class_accuracies(binary_predictions, all_labels)
                    sensitivities, specificities = calculate_sensitivity_specificity(binary_predictions, all_labels)
                    auc_metric(binary_predictions, all_labels)
                    auc_score = auc_metric.aggregate("macro")
                    auc_metric.reset()

                    log_metrics_to_csv_and_tensorboard(
                        epoch, class_accuracies, sensitivities, specificities, auc_score, mean_accuracy,
                        writer, "test", save_path, args.multi_class_columns
                    )

                # Update best test metrics
                if mean_accuracy > best_metric_test:
                    best_metric_test = mean_accuracy
                if auc_score > best_auc_test:
                    best_auc_test = auc_score
                    best_auc_epoch_test = epoch + 1

                print(f"Test - Accuracy: {mean_accuracy:.4f}, AUC: {auc_score:.4f}")

                # Save test predictions
                results_df = pd.DataFrame({
                    "name": all_names,
                    "ground_truth": all_labels.cpu().numpy().flatten(),
                    "predicted_probability": probabilities.cpu().numpy().flatten() if args.binary_classification else [
                        str(p) for p in probabilities.cpu().numpy()],
                    "prediction": binary_predictions.cpu().numpy().flatten()
                })
                results_df.to_excel(os.path.join(save_path, f"predictions_epoch_{epoch + 1}_test.xlsx"), index=False)

    # Final summary
    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETED")
    print(f"{'=' * 60}")
    print(f"Best Validation AUC: {best_auc:.4f} at epoch {best_auc_epoch}")
    if args.test_set_available:
        print(f"Best Test AUC: {best_auc_test:.4f} at epoch {best_auc_epoch_test}")

    writer.close()
    print(f"Results saved to: {save_path}")


if __name__ == "__main__":
    main()