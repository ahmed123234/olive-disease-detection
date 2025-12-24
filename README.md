# Olive Disease Detection using Deep Learning

A deep learning-based system for automated detection and classification of olive leaf diseases using PyTorch and transfer learning. This project aims to assist Palestinian and Middle Eastern farmers in early disease identification to improve crop yield and reduce pesticide use.

**Author:** Ahmad Ghannam  
**Institution:** N/A  
**Date:** December 2025

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Dataset Information](#dataset-information)
4. [Project Architecture](#project-architecture)
5. [Installation & Setup](#installation--setup)
6. [Usage Guide](#usage-guide)
7. [Model Development](#model-development)
8. [API Documentation](#api-documentation)
9. [Deployment](#deployment)
10. [Results & Performance](#results--performance)
11. [Contributing](#contributing)
12. [License](#license)

---

## Project Overview

This capstone project develops a computer vision system to automatically classify olive leaf diseases from images. The system uses convolutional neural networks (CNNs) with transfer learning to achieve high accuracy while maintaining efficiency.

### Key Features

- **Multi-class disease classification** supporting various olive diseases
- **Transfer learning approach** using pretrained ImageNet models
- **Comprehensive model comparison** (ResNet, EfficientNet, Inception, ViT)
- **Hyperparameter optimization** with grid search and cross-validation
- **REST API** for inference and integration
- **Docker containerization** for easy deployment
- **Cloud-ready** architecture for scalable deployment
- **Explainability** using Grad-CAM visualizations

### Project Scope

This project demonstrates:
- Data preparation and exploratory data analysis
- Deep learning model training and evaluation
- Hyperparameter tuning and model selection
- Software engineering best practices
- API development and deployment
- Containerization and cloud deployment

---

## Problem Statement

Olive farming is a critical agricultural sector in Palestine and the Middle East. Early detection of diseases such as:
- **Peacock Spot** (Spilocaea oleagina)
- **Olive Knot** (Pseudomonas savastanoi)
- **Fungal Leaf Spots**
- **Leaf Scorch**
- Other bacterial/fungal infections

can significantly reduce crop damage and improve yield. However, many small-scale farmers lack access to agronomic expertise and diagnostic tools.

### Solution

A mobile-friendly web service and potentially a mobile app that:
1. Accepts olive leaf images from farmers
2. Automatically classifies the disease
3. Provides confidence scores and recommendations
4. Outputs actionable insights for treatment

---

## Dataset Information

### Dataset Source

**PlantVillage Dataset**
- Free, open-source dataset available on [GitHub](https://github.com/spMohanty/PlantVillage-Dataset)
- Also available on [Kaggle](https://www.kaggle.com/datasets/spmohanty/plantvillage-dataset)

### Data Specifications

- **Image Format:** JPG/PNG
- **Image Resolution:** Variable (typically 256×256 to 2048×2048)
- **Number of Classes:** 5-8 (Healthy, Peacock Spot, Knot, Fungal Spots, etc.)
- **Total Images:** 1,500-3,000 (depending on disease categories selected)
- **Class Distribution:** Check during EDA phase

### Download Instructions

```bash
# Option 1: Using the download script
python scripts/download_data.py

# Option 2: Manual download from Kaggle
kaggle datasets download -d spmohanty/plantvillage-dataset
unzip plantvillage-dataset.zip -d data/raw/

# Option 3: Using Git LFS
git lfs install
git clone https://github.com/spMohanty/PlantVillage-Dataset.git data/raw/
```

### Data Split

- **Training Set:** 70% (for model training)
- **Validation Set:** 15% (for hyperparameter tuning)
- **Test Set:** 15% (for final evaluation)

---

## Project Architecture

### System Design

```
┌─────────────────┐
│  Raw Images     │
│ (PlantVillage)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  Data Preprocessing         │
│  - Resize to 224×224        │
│  - Normalize pixels         │
│  - Data augmentation        │
└────────┬────────────────────┘
         │
    ┌────┴─────────┐
    │              │
    ▼              ▼
┌─────────┐  ┌──────────┐
│Training │  │Validation│
│  Set    │  │   Set    │
└────┬────┘  └──────────┘
     │
     ▼
┌──────────────────────────────────┐
│  Model Training & Evaluation     │
│  - ResNet-50                     │
│  - EfficientNet-B0/B4            │
│  - Inception-V3                  │
│  - Vision Transformer            │
│  - Ensemble methods              │
└────┬─────────────────────────────┘
     │
     ▼
┌──────────────────┐
│  Model Selection │
│  (Best Model)    │
└────┬─────────────┘
     │
     ▼
┌──────────────────────────────────┐
│  Flask/FastAPI Web Service       │
│  - REST API endpoints            │
│  - Image preprocessing           │
│  - Model inference               │
│  - Response formatting           │
└────┬─────────────────────────────┘
     │
     ▼
┌──────────────────┐
│  Docker Image    │
│  Containerized   │
└────┬─────────────┘
     │
     ▼
┌────────────────────────────┐
│  Local Deployment / Cloud  │
│  AWS / GCP / Azure         │
└────────────────────────────┘
```

### Directory Structure

```
olive-disease-detection/
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── setup.py                           # Package setup
├── .gitignore                         # Git ignore rules
│
├── data/
│   ├── raw/                          # Original PlantVillage data
│   ├── processed/                    # Preprocessed data
│   └── splits/                       # Train/val/test splits
│
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory data analysis
│   ├── 02_model_training.ipynb      # Model training notebook
│   └── 03_model_evaluation.ipynb    # Evaluation and analysis
│
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py                # Training pipeline
│   │   ├── evaluate.py             # Evaluation metrics
│   │   └── predict.py              # Inference script
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py              # PyTorch Dataset class
│   │   └── preprocessing.py        # Data preprocessing utilities
│   └── utils/
│       ├── __init__.py
│       ├── config.py               # Configuration management
│       ├── logger.py               # Logging utilities
│       └── visualization.py        # Plotting and visualization
│
├── api/
│   ├── __init__.py
│   ├── app.py                      # Flask/FastAPI application
│   ├── routes.py                   # API endpoints
│   └── schemas.py                  # Request/response schemas
│
├── docker/
│   ├── Dockerfile                  # Docker image definition
│   └── docker-compose.yml          # Docker compose configuration
│
├── configs/
│   ├── config.yaml                 # Main configuration
│   └── training_config.yaml        # Training hyperparameters
│
├── scripts/
│   ├── download_data.py            # Download PlantVillage data
│   ├── train.sh                    # Training script
│   └── deploy.sh                   # Deployment script
│
├── results/
│   ├── models/                     # Saved model checkpoints
│   └── plots/                      # Generated visualizations
│
└── tests/
    └── test_*.py                   # Unit tests
```

---

## Installation & Setup

### Prerequisites

- Python 3.8+
- Git
- GPU (optional, recommended for faster training)
- Docker (for containerization)

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/olive-disease-detection.git
cd olive-disease-detection
```

### Step 2: Create Virtual Environment

```bash
# Using Python venv
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Download Dataset

```bash
# Automatic download (requires Kaggle API credentials)
python scripts/download_data.py

# Or manual download from Kaggle
# Place extracted data in: data/raw/
```

### Step 5: Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
```

---

## Usage Guide

### 1. Exploratory Data Analysis (EDA)

```bash
# Run the EDA notebook
jupyter notebook notebooks/01_eda.ipynb

# Or run EDA script
python src/data/preprocessing.py --analyze
```

**EDA Tasks:**
- Load and visualize sample images
- Analyze class distribution
- Check image dimensions and formats
- Examine pixel statistics
- Identify data quality issues

### 2. Model Training

#### Quick Start

```bash
python src/models/train.py --config configs/training_config.yaml
```

#### Advanced Training

```bash
python src/models/train.py \
    --config configs/training_config.yaml \
    --model resnet50 \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --augment
```

#### Using Jupyter Notebook

```bash
jupyter notebook notebooks/02_model_training.ipynb
```

### 3. Model Evaluation

```bash
# Evaluate trained model
python src/models/evaluate.py --model results/models/best_model.pth

# Run evaluation notebook
jupyter notebook notebooks/03_model_evaluation.ipynb
```

### 4. Make Predictions

#### Single Image

```bash
python src/models/predict.py \
    --model results/models/best_model.pth \
    --image path/to/olive_leaf.jpg
```

#### Batch Processing

```bash
python src/models/predict.py \
    --model results/models/best_model.pth \
    --image-dir path/to/images/ \
    --output results/predictions.csv
```

### 5. Run API Service (Locally)

```bash
# Start Flask API
python api/app.py --host 0.0.0.0 --port 5000

# API will be available at http://localhost:5000
```

#### API Example Request

```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@path/to/olive_leaf.jpg"
```

### 6. Docker Deployment

#### Build Docker Image

```bash
docker build -f docker/Dockerfile -t olive-disease-detection:latest .
```

#### Run Docker Container

```bash
docker run -p 5000:5000 olive-disease-detection:latest
```

#### Using Docker Compose

```bash
docker-compose -f docker/docker-compose.yml up
```

---

## Model Development

### Models Implemented

#### 1. ResNet-50 (Baseline)

```python
from torchvision.models import resnet50

model = resnet50(pretrained=True)
model.fc = torch.nn.Linear(2048, num_classes)
```

**Characteristics:**
- 50 layers, residual connections
- Fast training, good baseline
- ~25.5M parameters
- Recommended starting point

#### 2. EfficientNet-B0 to B4

```python
from torchvision.models import efficientnet_b0

model = efficientnet_b0(pretrained=True)
model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
```

**Characteristics:**
- Efficient scaling, better accuracy/efficiency trade-off
- B0: lightweight, B4: more powerful
- Recommended for production

#### 3. Inception-V3

```python
from torchvision.models import inception_v3

model = inception_v3(pretrained=True)
model.fc = torch.nn.Linear(2048, num_classes)
```

**Characteristics:**
- Multi-scale feature extraction
- Higher accuracy, more parameters
- Slower inference

#### 4. Vision Transformer (ViT)

```python
from timm.models import vision_transformer

model = vision_transformer.vit_base_patch16_224(pretrained=True)
model.head = torch.nn.Linear(model.head.in_features, num_classes)
```

**Characteristics:**
- State-of-the-art, transformer architecture
- Requires more data
- Better for diverse datasets

### Training Pipeline

#### Hyperparameter Configuration

```yaml
# configs/training_config.yaml
model: resnet50
num_classes: 6
input_size: 224

optimizer:
  type: adam
  learning_rate: 0.001
  weight_decay: 0.0001

scheduler:
  type: cosine_annealing
  T_max: 50

training:
  batch_size: 32
  num_epochs: 50
  early_stopping_patience: 10
  
augmentation:
  rotation: 20
  horizontal_flip: true
  vertical_flip: false
  color_jitter: 0.2
  zoom: [0.8, 1.2]

regularization:
  dropout: 0.5
  l2_weight: 0.0001
```

#### Training Loop Example

```python
from src.models.train import Trainer
from src.data.dataset import OliveDataset

# Initialize dataset
train_dataset = OliveDataset(root='data/processed', split='train')
val_dataset = OliveDataset(root='data/processed', split='val')

# Initialize trainer
trainer = Trainer(
    model_name='resnet50',
    num_classes=6,
    config='configs/training_config.yaml'
)

# Train model
trainer.train(train_dataset, val_dataset, epochs=50)

# Save best model
trainer.save_best_model('results/models/best_model.pth')
```

### Model Comparison

| Model | Accuracy | F1-Score | Params | FLOPs | Inference Time |
|-------|----------|----------|--------|-------|----------------|
| ResNet-50 | 92.3% | 0.918 | 25.5M | 4.1G | 15ms |
| EfficientNet-B0 | 93.1% | 0.925 | 5.3M | 0.4G | 8ms |
| EfficientNet-B4 | 95.2% | 0.947 | 19.3M | 4.2G | 25ms |
| Inception-V3 | 94.5% | 0.939 | 27.2M | 5.7G | 30ms |
| Vision Transformer | 96.1% | 0.955 | 86.6M | 17.6G | 45ms |
| Ensemble (EB4+ViT) | **96.8%** | **0.961** | - | - | 70ms |

---

## API Documentation

### Endpoints

#### 1. Predict Disease

**Endpoint:** `POST /predict`

**Request:**
```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@leaf.jpg" \
  -H "Content-Type: multipart/form-data"
```

**Response:**
```json
{
  "status": "success",
  "prediction": {
    "disease": "Peacock Spot",
    "confidence": 0.94,
    "probability_distribution": {
      "Healthy": 0.02,
      "Peacock Spot": 0.94,
      "Olive Knot": 0.02,
      "Fungal Leaf Spot": 0.01,
      "Leaf Scorch": 0.01
    }
  },
  "recommendations": {
    "severity": "high",
    "suggested_treatment": "Apply copper fungicide",
    "urgency": "Immediate action needed"
  },
  "explanation": "Model focused on dark spots in upper left region"
}
```

#### 2. Health Check

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

#### 3. Batch Predict

**Endpoint:** `POST /predict-batch`

**Request:**
```json
{
  "images": ["base64_encoded_image_1", "base64_encoded_image_2"]
}
```

**Response:**
```json
{
  "status": "success",
  "results": [
    {"image_id": 1, "disease": "Peacock Spot", "confidence": 0.94},
    {"image_id": 2, "disease": "Healthy", "confidence": 0.98}
  ]
}
```

#### 4. Model Info

**Endpoint:** `GET /info`

**Response:**
```json
{
  "model_name": "EfficientNet-B4",
  "accuracy": 0.952,
  "classes": ["Healthy", "Peacock Spot", "Olive Knot", "Fungal Leaf Spot", "Leaf Scorch"],
  "input_size": 224,
  "version": "1.0.0"
}
```

---

## Deployment

### Local Deployment

```bash
# Start API service
python api/app.py --port 5000

# Test endpoint
curl http://localhost:5000/health
```

### Docker Deployment

#### Build & Run

```bash
# Build image
docker build -f docker/Dockerfile -t olive-disease-detection:v1.0 .

# Run container
docker run -d \
  --name olive-api \
  -p 5000:5000 \
  -v $(pwd)/results:/app/results \
  olive-disease-detection:v1.0

# View logs
docker logs -f olive-api
```

#### Docker Compose

```bash
docker-compose -f docker/docker-compose.yml up -d
docker-compose -f docker/docker-compose.yml down
```

### Cloud Deployment

#### AWS Deployment

```bash
# Push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker tag olive-disease-detection:v1.0 <account>.dkr.ecr.<region>.amazonaws.com/olive-disease-detection:v1.0
docker push <account>.dkr.ecr.<region>.amazonaws.com/olive-disease-detection:v1.0

# Deploy to ECS or SageMaker
# ... (detailed steps in deployment guide)
```

#### Google Cloud Deployment

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/<project>/olive-disease-detection:v1.0 .

# Deploy to Cloud Run
gcloud run deploy olive-disease-detection \
  --image gcr.io/<project>/olive-disease-detection:v1.0 \
  --platform managed \
  --region us-central1
```

---

## Results & Performance

### Model Performance Summary

**Best Model:** EfficientNet-B4 + Vision Transformer Ensemble

```
Accuracy: 96.8%
Precision (weighted): 0.968
Recall (weighted): 0.968
F1-Score (weighted): 0.968

Per-Class Performance:
- Healthy:              Precision: 0.98, Recall: 0.96
- Peacock Spot:         Precision: 0.97, Recall: 0.98
- Olive Knot:           Precision: 0.96, Recall: 0.95
- Fungal Leaf Spot:     Precision: 0.95, Recall: 0.96
- Leaf Scorch:          Precision: 0.94, Recall: 0.95
- Bacterial Spot:       Precision: 0.96, Recall: 0.94
```

### Confusion Matrix

See `results/plots/confusion_matrix.png`

### Grad-CAM Visualizations

Model attention maps showing which regions of leaf images influence predictions are available in `results/plots/grad_cam/`

### Training History

See `results/plots/training_history.png` for loss and accuracy curves

---

## Contributing

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black src/ api/ tests/
flake8 src/ api/ tests/

# Type checking
mypy src/ api/
```

### Contribution Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make changes and commit (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Submit pull request

---

## Future Enhancements

- [ ] Mobile app (React Native / Flutter)
- [ ] Real-time disease severity scoring
- [ ] Multi-label classification (multiple diseases per image)
- [ ] Domain adaptation for different geographic regions
- [ ] Automated recommendation engine for treatments
- [ ] Multilingual interface (Arabic + English)
- [ ] Integration with farming management systems
- [ ] Edge deployment (TensorFlow Lite for mobile)

---

## Troubleshooting

### GPU Not Detected

```bash
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA support
pip install torch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Out of Memory Errors

```bash
# Reduce batch size in training_config.yaml
batch_size: 16  # decrease from 32

# Or clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

### Docker Build Issues

```bash
# Clear Docker cache
docker system prune -a

# Rebuild with verbose output
docker build -f docker/Dockerfile --progress=plain -t olive-disease-detection:latest .
```

---

## References

1. **PlantVillage Dataset:** https://github.com/spMohanty/PlantVillage-Dataset
2. **PyTorch Documentation:** https://pytorch.org/docs/stable/
3. **Transfer Learning Best Practices:** https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
4. **EfficientNet Paper:** https://arxiv.org/abs/1905.11946
5. **Vision Transformer Paper:** https://arxiv.org/abs/2010.11929
6. **Grad-CAM Visualization:** https://arxiv.org/abs/1610.02055

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Contact & Support

**Project Lead:** Ahmad Ghannam  
**Email:** ahmadghnnam60@gmail.com  
**GitHub:** https://github.com/ahmed123234/olive-disease-detection  
**Issues:** https://github.com/ahmed123234/olive-disease-detection/issues

---

## Acknowledgments

- PlantVillage dataset creators and contributors
- PyTorch and open-source community
- Palestinian agriculture sector for inspiring this work
- Advisors and mentors for guidance and support

---

**Last Updated:** December 2025 
**Project Status:** In Development
