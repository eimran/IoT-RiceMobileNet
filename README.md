# IoT-RiceMobileNet

**An Improved Lightweight MobileNetV2 Model for Real-time Multi-Class Rice Disease Detection Using IoT.**

IoT-RiceMobileNet is a deep learning model built on a MobileNetV2 backbone for classifying six rice leaf conditions: five diseases (bacterial leaf blight, brown spot, leaf blast, leaf scald, narrow brown spot) and healthy leaves. It is designed for real-time deployment on resource-constrained mobile and edge devices, and is integrated into an end-to-end IoT pipeline (ESP32-CAM → cloud inference → mobile/web app).

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
├── images                              # Self collected pre-processed images
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

The model was evaluated using two dataset settings:

1. **Constructed RLD dataset** — 7,092 images combining self-collected field images and publicly available Kaggle rice leaf images. This repository redistributes only the 773 self-collected field images in the `images/` folder. The Kaggle images should be obtained from their original source:  
   https://www.kaggle.com/datasets/dedeikhsandwisaputra/rice-leafs-disease-dataset/data

2. **Multi-source benchmark dataset** — 14,758 images integrated from public Kaggle and Mendeley sources. These third-party images are not redistributed in this repository and should be obtained from their original sources:
   - https://www.kaggle.com/datasets/anshulm257/rice-disease-dataset
   - https://data.mendeley.com/datasets/hx6f852hw4/2

## Data availability

The source code and self-collected rice leaf image dataset are archived on Zenodo:

https://doi.org/10.5281/zenodo.21140529

This repository includes the model implementation, requirements file, README documentation, and 773 self-collected rice leaf images organized by class. Third-party datasets used in the study are not redistributed here and should be accessed from their original public sources.

## Getting started

```bash
# Clone the repository
git clone <your-repo-url>
cd IoT-RiceMobileNet

# Install dependencies
pip install -r requirements.txt
```

Run the notebooks in `Experiment_on_proposed_DATASET/` and `Experiment_on_Multi_Source_DATASET/` to reproduce training and evaluation, and `plot_models_comparison.ipynb` for the comparison figures.


## Citation

Manuscript under review. Citation details will be added upon publication.

## License

Code in this repository is released under the MIT License.

The self-collected rice leaf images are released for research and academic use under the Creative Commons Attribution 4.0 International (CC BY 4.0) License. Third-party datasets referenced in this repository are governed by their original licenses and are not redistributed here.