# Multimodal Product Price Prediction using NLP and Computer Vision

This project focuses on predicting product prices using a **multimodal
machine learning approach**, combining textual and visual information.
The solution leverages transformer-based text embeddings and CNN-based
image embeddings, fused together for regression.

The project was **implemented and executed on Kaggle using GPU support**.
The complete workflow is contained in a single Jupyter Notebook.

---

## Problem Statement
Accurately predicting product prices is a challenging task due to the
diverse nature of product descriptions and images. This project aims to
leverage both **textual product descriptions** and **product images** to
improve price prediction performance.

---

## Dataset
- Dataset provided by the **Amazon ML Hackathon team**
- Each sample contains:
  - Product textual description
  - Image URL
  - Product price
- Dataset is **not included** in this repository due to size and access
  restrictions
- Precomputed embeddings were uploaded as Kaggle notebook inputs

---

## Approach
The project follows a multimodal pipeline:

1. Text preprocessing and encoding using **DistilBERT**
2. Image feature extraction using **ResNet50**
3. Precomputation of text and image embeddings
4. Feature engineering and cross-modal feature fusion
5. Training an **XGBoost Regressor** for price prediction

*Note:* Some notebook cells appear out of order due to the workflow of
generating embeddings first and later reusing them as Kaggle inputs.

---

## Technologies Used
- Python
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- PyTorch
- Transformers (DistilBERT)
- Computer Vision (ResNet50)
- Kaggle Notebooks

---

## Model & Evaluation

### Model
- XGBoost Regressor

### Evaluation Metrics
The model was evaluated using the metrics defined by the hackathon:

- **R² Score:** 0.34  
- **RMSE:** 0.77  
- **MAE:** 0.61  
- **SMAPE:** 61.8  

---

### SMAPE (Symmetric Mean Absolute Percentage Error)

The official SMAPE metric used in the hackathon is defined as:

\[
\text{SMAPE} = \frac{1}{n} \sum \frac{|\text{predicted\_price} - \text{actual\_price}|}
{\left(\frac{\text{actual\_price} + \text{predicted\_price}}{2}\right)}
\]

- SMAPE is bounded between **0% and 200%**
- **Lower values indicate better performance**

#### Example
If:
- Actual Price = 100  
- Predicted Price = 120  

\[
\text{SMAPE} = \frac{|100 - 120|}{(100 + 120)/2} \times 100 = 18.18\%
\]

---

## Execution Environment
- Platform: Kaggle Notebooks
- Hardware: Kaggle GPU
- Operating System: Linux (Kaggle Environment)

---

## How to Run the Project

### Option 1: Run on Kaggle (Recommended)
1. Upload the notebook to Kaggle
2. Add the dataset and embedding files as notebook inputs
3. Enable GPU from **Notebook Settings**
4. Run all cells sequentially

### Option 2: Run Locally
1. Install required dependencies:
   ```bash
   pip install -r requirements.txt


## Key Learnings
1.Multimodal feature extraction using NLP and Computer Vision

2.Embedding-based representation learning

3.Cross-modal feature fusion

4.Regression modeling using gradient boosting

5.Working with GPU-based environments on Kaggle

## Future Improvements
1.Fine-tuning text and image encoders end-to-end

2.Experimenting with attention-based multimodal fusion

3.Deploying the model as an inference API
