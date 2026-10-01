\# Breast Cancer Histopathology Classification



A deep learning pipeline for binary classification of breast cancer histopathology images using multiple vision architectures and hybrid feature fusion.



\## Overview



This project investigates deep feature extraction and feature fusion approaches for histopathology image classification.



The pipeline combines features extracted from:



\- Vision Transformer (ViT)

\- Swin Transformer

\- ConvNeXt



The extracted deep representations are fused and classified using an RBF-kernel Support Vector Machine (SVM).



\## Pipeline



```text

Histopathology Images

&#x20;       │

&#x20;       ▼

Dataset Preprocessing

&#x20;       │

&#x20;       ▼

┌─────────────────────────────┐

│  ViT Feature Extraction     │

│  Swin Feature Extraction    │

│  ConvNeXt Feature Extraction│

└─────────────────────────────┘

&#x20;       │

&#x20;       ▼

Hybrid Feature Fusion

&#x20;       │

&#x20;       ▼

RBF SVM Classification

&#x20;       │

&#x20;       ▼

Model Evaluation
Models

Vision Transformer



A Vision Transformer (ViT) based feature extractor is used to obtain deep visual representations from histopathology images.



Swin Transformer



A Swin Transformer feature extractor provides hierarchical visual representations using shifted-window attention.



ConvNeXt



ConvNeXt is used as an additional deep feature extractor to capture complementary visual information.



Hybrid Feature Fusion



Features extracted from the three architectures are combined into a unified representation before classification.



RBF SVM



An RBF-kernel Support Vector Machine is used as the final classifier on the fused deep features.



Dataset



The project uses breast cancer histopathology images for binary classification:



IDC Negative

IDC Positive



Dataset preprocessing and patch extraction utilities are included in the dataset/ directory.



Evaluation



The project includes:



Accuracy

Precision

Recall

F1-score

ROC-AUC

Balanced Accuracy

Confusion Matrix

ROC Curve

Training / validation curves



The evaluation pipeline also includes Grad-CAM based interpretability analysis.



Results



The final experimental results include:



Metric	Score

Accuracy	91.46%

Precision	91.69%

Recall	91.14%

F1-Score	91.42%

ROC-AUC	96.82%

Balanced Accuracy	91.46%



The best validation accuracy during training was approximately 90.97%.



Project Structure

transformers\_model/

│

├── app.py

├── main.py

├── quick\_test.py

├── requirements.txt

│

├── configs/

│   └── config.py

│

├── dataset/

│   ├── dataset\_builder.py

│   ├── dataset\_loader.py

│   └── patch\_extractor.py

│

├── models/

│   ├── vit\_model.py

│   ├── swin\_model.py

│   ├── convnext\_model.py

│   └── feature\_fusion.py

│

├── training/

│   ├── train\_pipeline.py

│   └── train\_svm.py

│

├── evaluation/

│   ├── metrics.py

│   ├── visualization.py

│   ├── gradcam.py

│   └── gradcam\_main.py

│

├── figures/

│

├── results/

│

└── requirements.txt

Technologies

Python

PyTorch

Vision Transformer

Swin Transformer

ConvNeXt

Scikit-learn

Support Vector Machine

NumPy

Matplotlib

Grad-CAM

Reproducibility



The project is organized as a modular pipeline separating:



Dataset preprocessing

Feature extraction

Feature fusion

Model training

Classification

Evaluation

Interpretability analysis



Generated datasets, model checkpoints, logs, and generated Grad-CAM outputs are excluded from version control where appropriate.



Author



Fatma Betül Pehlivan







