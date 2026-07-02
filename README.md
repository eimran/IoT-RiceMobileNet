# IoT-RiceMobileNet

**A lightweight improved MobileNetV2 model for real-time multi-class rice leaf disease detection using IoT.**

IoT-RiceMobileNet is a compact deep learning model built on a MobileNetV2 backbone for classifying six rice leaf conditions — five diseases (bacterial leaf blight, brown spot, leaf blast, leaf scald, narrow brown spot) and healthy leaves. It is designed for real-time deployment on resource-constrained mobile and edge devices, and is integrated into an end-to-end IoT pipeline (ESP32-CAM → cloud inference → mobile/web app).

## Highlights

- **99.19%** test accuracy on the constructed RLD dataset (7,092 images) and **99.25%** on a heterogeneous multi-source dataset (14,758 images).
- **Lightweight:** 2.418M parameters, 9.501 MB model size, 0.670 GFLOPs.
- **Fast:** 85.17 FPS inference speed, suitable for real-time use.
- **End-to-end IoT system:** ESP32-CAM capture, FastAPI cloud inference on AWS EC2, and a React Native Android app.
- Validated with stratified 5-fold and 10-fold cross-validation and Grad-CAM interpretability.

## Repository structure

```
.
├── API/                                # FastAPI backend
├── Experiment_on_proposed_DATASET/     # Training & evaluation on the constructed RLD dataset
├── Experiment_on_Multi_Source_DATASET/ # Training & evaluation on the multi-source dataset
├── plot_models_comparison.ipynb        # Comparative plots across all evaluated models
├── images        # Comparative plots across all evaluated models
    ├── bacterial_leaf_blight/
    ├── brown_spot/
    ├── healthy/
    ├── leaf_blast/
    ├── leaf_scald/
    └── narrow_brown_spot/
├── .gitignore
└── README.md
```

## Datasets

The model is evaluated on two dataset settings:

1. **Constructed RLD dataset** — 7,092 images combining self-collected field images (Chapulia, Gazipur, Bangladesh) with publicly available Kaggle images.
2. **Multi-source benchmark dataset** — 14,758 images integrated from public Kaggle and Mendeley sources.

Public source datasets:
- https://www.kaggle.com/datasets/dedeikhsandwisaputra/rice-leafs-disease-dataset/data
- https://www.kaggle.com/datasets/anshulm257/rice-disease-dataset
- https://data.mendeley.com/datasets/hx6f852hw4/2

## Getting started

```bash
# Clone the repository
git clone <your-repo-url>
cd IoT-RiceMobileNet

# Install dependencies
pip install -r requirements.txt
```

Run the notebooks in `Experiment_on_proposed_DATASET/` and `Experiment_on_Multi_Source_DATASET/` to reproduce training and evaluation, and `plot_models_comparison.ipynb` for the comparison figures.

### Running the inference API

```bash
cd API
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API exposes `/predict` (POST, `multipart/form-data` JPEG upload) and returns a JSON response with the predicted class label and confidence score.

## Citation

Manuscript under review. Citation details will be added upon publication.

## License

To be added.