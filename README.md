
# HinglishMemeX: Multimodal Fake News & Satire Classification in Hinglish Memes

HinglishMemeX is a multimodal deep learning pipeline that classifies Hinglish memes into four categories: **Real**, **Fake**, **Satire**, and **Partially True**.  
It combines **CLIP** for visual feature extraction with **IndicBERT** for text understanding, enabling robust classification of meme-based misinformation.

---

## 📌 Features

- **Multimodal architecture**: Combines image and text encoders for richer contextual understanding.
- **Pretrained backbones**:  
  - Image Encoder: `openai/clip-vit-base-patch32`  
  - Text Encoder: `ai4bharat/indic-bert`
- **4-class classification**:
  - `real`
  - `fake`
  - `satire`
  - `partially true`
- **CLIP-style image preprocessing** and **tokenization for Hinglish**.
- **W&B (Weights & Biases)** integration for experiment tracking.
- **Training, validation, and testing pipelines** with early stopping.
- **Evaluation reports** including classification metrics, confusion matrix, and per-class performance.

---

## 📂 Project Structure

HinglishMemeX/
│
├── HinglishMemeX.ipynb # Main training and evaluation notebook
├── dataset_split_final/ # Dataset splits (train/val/test)
│ ├── dataset_splits/
│ │ ├── train/
│ │ │ ├── annotations/ # JSONL annotation files
│ │ │ ├── images/ # Meme images
│ │ ├── val/
│ │ ├── test/
│
├── best_model.pt # Saved best model
├── wandb/ # W&B experiment logs
└── test_predictions.csv # Model predictions on test set


#🛠️ Installation
1. Clone the repository:

```bash
git clone https://github.com/your-username/HinglishMemeX.git
cd HinglishMemeX
```

2. Install dependencies:
```bash
pip install torch torchvision transformers wandb scikit-learn matplotlib seaborn pandas pillow tqdm
```

# 1️⃣ Data Preparation

# 2️⃣ Training
1. Inside the notebook (HinglishMemeX.ipynb)
2. Load dataset splits
3. Initialize MultimodalClassifier
4. Train with W&B logging
5. Early stopping enabled for better generalization


# 3️⃣ Evaluation
1. After training, the notebook
2. Loads the best saved model
3. Evaluates on the test set
4. Generates classification report & confusion matrix
5. Saves predictions to test_predictions.csv

# 🧠 Model Architecture
1. Image Encoder: CLIP Vision Transformer (clip-vit-base-patch32)
2. Text Encoder: IndicBERT (ai4bharat/indic-bert)
3. Projection Layers: Reduce embeddings to a unified hidden dimension
4. Fusion Layer: Concatenate image & text embeddings
5. Classification Head: Fully connected layers → softmax output





