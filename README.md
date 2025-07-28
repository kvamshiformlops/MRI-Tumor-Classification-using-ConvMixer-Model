
# 🧠 Breast MRI Tumor Classification with ConvMixer
A high-performance deep learning pipeline for classifying breast tumors (Benign vs Malignant) from MRI scans. This project combines robust data augmentation with a custom ConvMixer model, delivering precision-level diagnostics from visual patterns.
---

## 📂 Dataset

Sourced from [Kaggle: Breast MRI Tumor Classification Dataset](https://www.kaggle.com/datasets/abenjelloun/breast-mri-tumor-classification-dataset)

- **Training Set:** 16,826 images  
- **Validation Set:** 3,594 images  
- **Test Set:** 3,604 images  
- **Categories:**  
  - `Benign`  
  - `Malignant`

---

## 🏋️ Model Training Strategy

### 🔧 Preprocessing & Data Augmentation

To simulate diverse clinical conditions and reduce overfitting:

- **Rescaling:** Pixel intensity normalized to [0, 1]  
- **Train-time Augmentations:**
  - Horizontal flipping
  - Random zoom, shift, shear, and rotation
  - `fill_mode='nearest'` to handle spatial transformation artifacts
- **Validation/Test Sets:** Kept pristine for unbiased evaluation

### 📊 Class Balancing
Instead of blindly trusting balance,computed precise class weights using scikit-learn:
```python
class_weights = compute_class_weight(...)
```
Achieved `{0: 1.0, 1: 1.0}` — near-perfect balance across benign and malignant classes.

### 🧠 Model Architecture — ConvMixer

Inspired by emerging research, the ConvMixer structure merges the simplicity of convolutions with Transformer-like patch mixing:

- **Patch Embedding:** Reduces spatial dimensionality early
- **Mixer Blocks:**
  - Depthwise Convolutions for spatial filtering
  - Residual connections for stability
  - Pointwise Convolutions for cross-channel blending
- **Head:** Global Average Pooling + Dense layer with sigmoid activation

### ⚙️ Optimization & Regularization

- **Loss Function:** Binary crossentropy  
- **Optimizer:** Adam with `learning_rate=1e-4`  
- **Callbacks:** Early stopping (`patience=5`) to halt overfitting and retain best weights  
- **Batch Size:** 64  
- **Epochs:** 30 (subject to early stopping)

### 🔍 Evaluation

Post-training predictions were rounded with a threshold of 0.5.Metrics confirmed strong generalization:

| Metric       | Value    |
|--------------|----------|
| Accuracy     | 96.03%   |
| Precision    | 96.16%   |
| Recall       | 96.03%   |
| F1-Score     | 96.03%   |

**Classification Report:**

```
              precision    recall  f1-score   support

      Benign       0.94      0.99      0.96      1802
   Malignant       0.99      0.93      0.96      1802

    Accuracy                           0.96      3604
   Macro Avg       0.96      0.96      0.96      3604
Weighted Avg       0.96      0.96      0.96      3604
```

---

## 🧰 Requirements

To reproduce this environment:

- `TensorFlow >= 2.10`  
- `scikit-learn`  
- `numpy`  
- `matplotlib`  
- `kagglehub`

Install dependencies with:
```bash
pip install tensorflow scikit-learn numpy matplotlib kagglehub
```
---


## 💡 Future Directions
There's room for further innovation:
- Transfer learning with pretrained medical vision backbones  
- Grad-CAM visualization for interpretability  
- Hyperparameter tuning with Ray or Optuna  
- Ensemble models for voting-based predictions  
- Integration into diagnostic dashboards
---

## 🤝 Contributing
Whether you’re refining medical models, passionate about AI ethics in healthcare, or just curious—contributions are welcome! Fork this repo and submit a pull request, or open an issue to discuss new directions.
---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more details.
---
