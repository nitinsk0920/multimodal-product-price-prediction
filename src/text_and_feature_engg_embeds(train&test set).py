# ===============================
# DistilBERT embeddings + Engineered numeric features (NaN-safe) for train/test
# ===============================
import os
import re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder

# ===============================
# Config
# ===============================
TEXT_COL = "catalog_content"  # Text column
BATCH_SIZE = 64
MAX_LEN = 128
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "/kaggle/working/embeddings_features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CSV paths
TRAIN_CSV = "/kaggle/input/mlhack/train.csv"
TEST_CSV = "/kaggle/input/mlhack/test.csv"

# ===============================
# Load datasets
# ===============================
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

# Fill missing text with empty string and clean
def light_clean(text):
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

train_df[TEXT_COL] = train_df[TEXT_COL].fillna("").astype(str).apply(light_clean)
test_df[TEXT_COL] = test_df[TEXT_COL].fillna("").astype(str).apply(light_clean)

# ===============================
# Device and model setup
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

# -------------------------------
# Mean pooling
def mean_pooling(token_embeddings, attention_mask):
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeds = torch.sum(token_embeddings * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    return sum_embeds / sum_mask

# -------------------------------
# Encode text to embeddings
def encode_texts_to_embeddings(df, text_col, out_path, batch_size=BATCH_SIZE, max_length=MAX_LEN):
    n = len(df)
    emb_dim = model.config.hidden_size
    embeddings = np.zeros((n, emb_dim), dtype=np.float32)

    texts = df[text_col].fillna("").astype(str).tolist()
    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc=f"Encoding {os.path.basename(out_path)}"):
            end = min(n, start + batch_size)
            batch_texts = texts[start:end]
            encoded = tokenizer(batch_texts, padding=True, truncation=True,
                                max_length=max_length, return_tensors='pt')
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pooled = mean_pooling(outputs.last_hidden_state, attention_mask).cpu().numpy()
            embeddings[start:end, :] = pooled

    # Replace any NaNs (just in case)
    embeddings = np.nan_to_num(embeddings, nan=0.0)
    np.save(out_path, embeddings)
    print(f"Saved text embeddings: {out_path}")
    return embeddings

# -------------------------------
# Save engineered numeric/categorical features
def save_engineered_features(df, out_path, text_col=TEXT_COL, target_col=None, train_le_dict=None):
    # Exclude text and optionally target
    exclude_cols = [text_col, 'sample_id']
    if target_col is not None and target_col in df.columns:
        exclude_cols.append(target_col)

    engineered_cols = [c for c in df.columns if c not in exclude_cols]
    numeric_df = df[engineered_cols].copy()

    le_dict = {} if train_le_dict is None else train_le_dict

    for col in numeric_df.select_dtypes(include='object').columns:
        if train_le_dict is None:  # train set
            le = LabelEncoder()
            numeric_df[col] = numeric_df[col].fillna("nan_missing")
            numeric_df[col] = le.fit_transform(numeric_df[col].astype(str))
            le_dict[col] = le
        else:  # test set
            le = le_dict[col]
            numeric_df[col] = numeric_df[col].fillna("nan_missing")
            numeric_df[col] = numeric_df[col].apply(lambda x: x if x in le.classes_ else "nan_missing")
            numeric_df[col] = le.transform(numeric_df[col].astype(str))

    # Fill numeric NaNs with 0
    numeric_df = numeric_df.fillna(0)

    np.save(out_path, numeric_df.values.astype(np.float32))
    print(f"Saved engineered numeric features: {out_path}")
    return numeric_df.values, le_dict

# ===============================
# Paths for saving
train_text_emb_path = os.path.join(OUTPUT_DIR, "train_catalog_text_emb.npy")
train_features_path = os.path.join(OUTPUT_DIR, "train_engineered_numeric_features.npy")
test_text_emb_path = os.path.join(OUTPUT_DIR, "test_catalog_text_emb.npy")
test_features_path = os.path.join(OUTPUT_DIR, "test_engineered_numeric_features.npy")

# ===============================
# Process train
train_text_emb = encode_texts_to_embeddings(train_df, TEXT_COL, train_text_emb_path)
train_features, le_dict = save_engineered_features(train_df, train_features_path, text_col=TEXT_COL, target_col='price')

# ===============================
# Process test (no target column in test)
test_text_emb = encode_texts_to_embeddings(test_df, TEXT_COL, test_text_emb_path)
test_features, _ = save_engineered_features(test_df, test_features_path, text_col=TEXT_COL, target_col=None, train_le_dict=le_dict)

# ===============================
print("✅ All embeddings and features saved separately for train and test.")
print("Train text embeddings:", train_text_emb_path)
print("Train engineered numeric features:", train_features_path)
print("Test text embeddings:", test_text_emb_path)
print("Test engineered numeric features:", test_features_path)

##FOR TEST SET EMBEDS:

import os
import re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder

# ===============================
# Config
# ===============================
TEXT_COL = "catalog_content"
BATCH_SIZE = 64
MAX_LEN = 128
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "/kaggle/working/test_embeddings_features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_CSV = "/kaggle/input/mlhack/test.csv"

# ===============================
# Load test dataset
# ===============================
test_df = pd.read_csv(TEST_CSV)

# Clean text
def light_clean(text):
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

test_df[TEXT_COL] = test_df[TEXT_COL].fillna("").astype(str).apply(light_clean)

# ===============================
# Device and model setup
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

# -------------------------------
# Mean pooling
def mean_pooling(token_embeddings, attention_mask):
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeds = torch.sum(token_embeddings * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    return sum_embeds / sum_mask

# -------------------------------
# Encode test texts to embeddings
def encode_texts_to_embeddings(df, text_col, out_path, batch_size=BATCH_SIZE, max_length=MAX_LEN):
    n = len(df)
    emb_dim = model.config.hidden_size
    embeddings = np.zeros((n, emb_dim), dtype=np.float32)

    texts = df[text_col].fillna("").astype(str).tolist()
    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc=f"Encoding {os.path.basename(out_path)}"):
            end = min(n, start + batch_size)
            batch_texts = texts[start:end]
            encoded = tokenizer(batch_texts, padding=True, truncation=True,
                                max_length=max_length, return_tensors='pt')
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pooled = mean_pooling(outputs.last_hidden_state, attention_mask).cpu().numpy()
            embeddings[start:end, :] = pooled

    embeddings = np.nan_to_num(embeddings, nan=0.0)
    np.save(out_path, embeddings)
    print(f"Saved test text embeddings: {out_path}")
    return embeddings

# -------------------------------
# Save engineered numeric/categorical features for test
def save_engineered_features(df, out_path, text_col=TEXT_COL):
    exclude_cols = [text_col, 'sample_id']
    numeric_df = df[[c for c in df.columns if c not in exclude_cols]].copy()

    for col in numeric_df.select_dtypes(include='object').columns:
        numeric_df[col] = numeric_df[col].fillna("nan_missing")
        le = LabelEncoder()
        numeric_df[col] = le.fit_transform(numeric_df[col].astype(str))

    numeric_df = numeric_df.fillna(0)
    np.save(out_path, numeric_df.values.astype(np.float32))
    print(f"Saved test engineered numeric features: {out_path}")
    return numeric_df.values

# ===============================
# Paths for saving test embeddings/features
test_text_emb_path = os.path.join(OUTPUT_DIR, "test_catalog_text_emb.npy")
test_features_path = os.path.join(OUTPUT_DIR, "test_engineered_numeric_features.npy")

# ===============================
# Process test set
test_text_emb = encode_texts_to_embeddings(test_df, TEXT_COL, test_text_emb_path)
test_features = save_engineered_features(test_df, test_features_path, text_col=TEXT_COL)

print("✅ Test embeddings and features saved.")
print("Test text embeddings:", test_text_emb_path)
print("Test engineered numeric features:", test_features_path)


