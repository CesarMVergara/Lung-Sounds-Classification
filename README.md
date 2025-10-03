# Lung-Sounds-Classification
Lung Sound Classification using CNNs. This project implements deep learning models to classify lung sounds from Mel spectrograms.


---

### Data Source and Assets

The foundational dataset for this research is the publicly available **ICBHI 2017 Respiratory Sound Classification Challenge**.

| Asset | Description | Reference / Link |
| :--- | :--- | :--- |
| **Source Dataset** | **ICBHI 2017 Challenge** used for baseline audio. | https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge |
| **Citation** | The work is based on the data described in this paper: | Rocha BM et al. (2019) "An open access database for the evaluation of respiratory sound classification algorithms" *Physiological Measurement* 40 035001 |
| **Repository Data** | The `/OfficialSpectrogramsMelAUG` folder contains the **pre-processed and augmented Mel spectrograms** used directly as model input. | N/A (Local asset) |

---

### Environment Specifications

The following key Python libraries and versions were used to ensure the integrity of the preprocessing pipeline and the training process.

To replicate the environment, please use the following dependencies:

| Package | Version |
| :--- | :--- |
| **tensorflow** | 2.10.0 |
| **keras** | 2.10.0 |
| **numpy** | 1.26.2 |
| **librosa** | 0.10.1 |
| **scikit-learn** | 1.3.2 |
| **seaborn** | 0.13.2 |
| **matplotlib** | 3.8.2 |
| **soundfile** | 0.12.1 |

*(Note: The `requirements.txt` file is included in this repository for easy environment setup.)*

---

### Code Execution Pipeline (Jupyter Notebooks)

The following notebooks document the sequential steps of data preparation and model training.

| Notebook | Phase | Functionality |
| :--- | :--- | :--- |
| `1_Cycles.ipynb` | **Preprocessing** | Performs audio downsampling and segments raw audio files into individual respiratory **cycles**, organizing them by class. |
| `2_AudioAugmentation.ipynb` | **Data Augmentation (Audio)** | Applies 4 traditional augmentation techniques directly to the segmented audio cycles. |
| `3_SpectrogramsMel.ipynb` | **Feature Extraction** | Generates Mel spectrogram images from the complete augmented audio dataset. |
| `4_MixUp_ImageAugmentation.ipynb` | **Data Augmentation (Images)** | Applies the **MixUp** technique exclusively on the newly generated Mel spectrogram images, finalizing the dataset. |
| **Training Notebooks** | **Model Training** | Notebooks dedicated to transfer learning and training: `VGG16`, `VGG19`, `MNV3L`, `InceptionV3`, `ResNet152V2`, and `CustomDualNet`. |
| `LoadModel.ipynb` | **Verification** | Used to load a saved model and verify performance metrics. |
| `LoadCustomModel.ipynb` | **Verification (DualNet)** | Used to load the **CustomDualNet** model and verify performance metrics. |

---
