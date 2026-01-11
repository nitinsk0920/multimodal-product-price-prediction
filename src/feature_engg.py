
train_csv_path= "/kaggle/input/mlhack/train.csv"
test_csv_path = "/kaggle/input/mlhack/test.csv"
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

import re
import pandas as pd

def extract_weight(text):
    match = re.search(r'(\d+(?:\.\d+)?)\s*(g|kg|oz|lb|ml|l)', text, re.IGNORECASE)
    if match:
        value, unit = match.groups()
        value = float(value)
        unit = unit.lower()
        if unit == 'kg': value *= 1000
        elif unit == 'lb': value *= 453.592
        elif unit == 'oz': value *= 28.3495
        elif unit == 'l': value *= 1000  # ml
        return value
    return 0
train_df['weight_g'] = train_df['catalog_content'].apply(extract_weight)

test_df['weight_g'] = test_df['catalog_content'].apply(extract_weight)


def extract_pack_count(text):
    match = re.search(r'(\d+)\s*(?:x|X|pack of)\s*\d*', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 1

train_df['pack_count'] = train_df['catalog_content'].apply(extract_pack_count)
test_df['pack_count'] = test_df['catalog_content'].apply(extract_pack_count)




train_df['num_tokens'] = train_df['catalog_content'].str.split().str.len()
train_df['num_digits'] = train_df['catalog_content'].str.count(r'\d')
train_df['num_uppercase_words'] = train_df['catalog_content'].str.count(r'\b[A-Z]{2,}\b')

test_df['num_tokens'] = test_df['catalog_content'].str.split().str.len()
test_df['num_digits'] = test_df['catalog_content'].str.count(r'\d')
test_df['num_uppercase_words'] = test_df['catalog_content'].str.count(r'\b[A-Z]{2,}\b')



