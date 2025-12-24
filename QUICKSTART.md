# Quick Start Guide - Olive Disease Detection

Get up and running in 5 minutes!

## Prerequisites

- Python 3.8+
- Git
- 4GB RAM minimum (8GB+ recommended)
- Optional: GPU with CUDA support for faster training

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/olive-disease-detection.git
cd olive-disease-detection
```

### 2. Create and Activate Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download Dataset

**Option A: Using Kaggle CLI (Recommended)**

```bash
# Install and configure Kaggle
pip install kaggle

# Download from Kaggle (requires free account)
mkdir -p data/raw
kaggle datasets download -d spmohanty/plantvillage-dataset
unzip plantvillage-dataset.zip -d data/raw/
rm plantvillage-dataset.zip
```

**Option B: Manual Download**

- Visit: https://www.kaggle.com/datasets/spmohanty/plantvillage-dataset
- Download and extract to `data/raw/`

**Option C: Using Download Script**

```bash
python scripts/download_data.py
```

### 5. Verify Setup

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
ls -la data/raw/  # Check if data exists
```

## Running EDA (5-10 minutes)

```bash
jupyter notebook notebooks/01_eda.ipynb
```

This will:
- Load sample images
- Show class distribution
- Display statistics
- Identify any data issues

## Training a Model (30-60 minutes on CPU, 5-10 minutes on GPU)

### Quick Training (Default)

```bash
python src/models/train.py --config configs/training_config.yaml
```

### Custom Training

```bash
python src/models/train.py \
    --model resnet50 \
    --epochs 30 \
    --batch-size 16 \
    --learning-rate 0.001
```

Check `results/models/` for saved weights.

## Testing Predictions

```bash
# Single image
python src/models/predict.py \
    --model results/models/best_model.pth \
    --image data/raw/sample_image.jpg
```

## Running API Locally

```bash
python api/app.py --port 5000
```

Visit: http://localhost:5000/health

### Test API

```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@data/raw/sample_image.jpg"
```

## Docker Deployment

```bash
# Build image
docker build -f docker/Dockerfile -t olive-disease:latest .

# Run container
docker run -p 5000:5000 olive-disease:latest

# Visit http://localhost:5000
```

## Project Structure

```
olive-disease-detection/
├── data/                 # Dataset folder
├── notebooks/            # Jupyter notebooks for EDA and training
├── src/                  # Source code (models, data, utils)
├── api/                  # Flask/FastAPI application
├── docker/               # Docker configuration
├── configs/              # Configuration files
├── scripts/              # Helper scripts
└── results/              # Saved models and plots
```

## Common Issues

### Issue: "No module named 'torch'"

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: CUDA/GPU not detected

**Solution:**
```bash
python -c "import torch; print(torch.cuda.is_available())"

# If False, use CPU or reinstall PyTorch with CUDA
pip install torch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Issue: Out of Memory (OOM)

**Solution:**
Reduce batch size in `configs/training_config.yaml`:
```yaml
training:
  batch_size: 16  # Change from 32 to 16
```

### Issue: Data not found

**Solution:**
```bash
# Check data location
ls data/raw/
# Should show: "Olive" or similar disease folders
```

## Next Steps

1. **Explore the notebooks:** Start with `notebooks/01_eda.ipynb`
2. **Read the full README:** See detailed documentation in `README.md`
3. **Train models:** Run `notebooks/02_model_training.ipynb`
4. **Evaluate results:** Check `notebooks/03_model_evaluation.ipynb`
5. **Deploy:** Set up the API and Docker

## Getting Help

- Check `README.md` for detailed documentation
- Review `notebooks/` for example code
- Check GitHub issues: https://github.com/yourusername/olive-disease-detection/issues
- Read PyTorch tutorials: https://pytorch.org/tutorials/

## File Structure for First-Time Users

After setup, your structure should look like:

```
olive-disease-detection/
├── venv/                           # Your virtual environment
├── data/
│   └── raw/
│       ├── Olive - healthy/
│       ├── Olive - peacock_spot/
│       ├── Olive - knot/
│       └── ... (other disease classes)
├── results/
│   ├── models/                    # Will be filled after training
│   └── plots/                     # Will be filled after training
└── ... (other files)
```

## Tips for Success

1. **Start with EDA:** Understand your data first
2. **Use CPU for testing:** Verify code works before GPU training
3. **Monitor training:** Watch loss and accuracy curves
4. **Validate before deploy:** Test on unseen data
5. **Save often:** Checkpoints help with long training jobs
6. **Document changes:** Track what works and what doesn't

Good luck! 🚀