# KTSC_Task
kidney tumor segmentation and classification

This document provides a step-by-step guide to perform kidney tumor segmentation using the `nnUNet` framework and classification tasks using the `MONAI` framework.

## Prerequisites

1. **Install nnUNet Framework**: Follow the instructions in the [nnUNet GitHub repository](https://github.com/MIC-DKFZ/nnUNet) to set up the nnUNet framework.
2. **Install MONAI Framework**: Follow the instructions in the [MONAI GitHub repository](https://github.com/Project-MONAI/MONAI) to install the MONAI framework.

## 1. Segmentation Task (Using nnUNet Framework)

For the kidney tumor segmentation task, you need to use pretrained weights from the `nnUNet` framework. These weights have been uploaded to [Google Drive](https://drive.google.com/drive/folders/1qzDT5dkYyyiN8ebwx53VzzGEKxMhAiZa?usp=drive_link).

### Steps:
1. Download the pretrained weight files from the provided Google Drive link.
2. Place the weight files in the `nnUNet_results` directory:
3. Perform Segmentation with nnUNet:
    ```bash
   nnUNet_predict -i <input_folder> -o <output_folder> -d 040 -c 3d_fullres
   ```
   
## 2. Classification Task (Using MONAI Framework)

### Steps:
1. You need to replace the files in the `monai/networks/nets` directory with the provided files to ensure that the DenseNet121_EMA model architecture is matched.
2. After replacing the model files, use the `train.py` script for training and inference.
