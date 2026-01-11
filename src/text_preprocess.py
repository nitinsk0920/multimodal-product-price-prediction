# -------------------------
# Utility preprocessing functions
# -------------------------
def light_clean(text):
    """Minimal cleaning for DistilBERT: remove HTML, collapse whitespace, keep punctuation."""
    if text is None:
        return ""
    text = str(text)
    # remove html tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # replace newlines / tabs with space
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    # collapse multi-spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Example regex extraction for weight/quantity (optional but high ROI)
def extract_quantity_and_weight(text):
    """
    Returns tuple (quantity_int_or_nan, weight_grams_or_nan)
    This is a heuristic extractor — tweak regexes for your data.
    """
    q = np.nan
    w = np.nan
    if not text:
        return q, w
    # pack counts like 'pack of 6', '6 pack', '6 x 250g'
    m = re.search(r'(\bpack of\b|\bpack\b|\bx\b)\s*(\d{1,3})', text, flags=re.I)
    if m:
        try:
            q = int(re.search(r'(\d{1,3})', m.group(0)).group(1))
        except:
            q = np.nan
    # weights like '250 g', '250g', '0.5 kg', '12 oz'
    m2 = re.search(r'(\d+(?:\.\d+)?)\s*(kg|g|gram|grams|oz|ounce|ounces|ml|l|litre|liter)\b', text, flags=re.I)
    if m2:
        val = float(m2.group(1))
        unit = m2.group(2).lower()
        # convert to grams/ml where possible
        if unit in ['kg']:
            w = val * 1000.0
        elif unit in ['g','gram','grams']:
            w = val
        elif unit in ['mg']:
            w = val / 1000.0
        elif unit in ['l','litre','liter']:
            w = val * 1000.0  # liters -> ml (approx)
        elif unit in ['ml']:
            w = val
        elif unit in ['oz','ounce','ounces']:
            w = val * 28.3495
    return q, w


# -------------------------
# Load dataset(s)
# -------------------------
print("Loading CSVs...")
train_df = pd.read_csv(TRAIN_CSV)
print("Train rows:", len(train_df))
if os.path.exists(TEST_CSV):
    test_df = pd.read_csv(TEST_CSV)
    print("Test rows:", len(test_df))
else:
    test_df = None
print("Using text column:", TEXT_COL)

# Fill missing and clean text
train_df[TEXT_COL] = train_df[TEXT_COL].fillna("").astype(str).apply(light_clean)
if test_df is not None:
    test_df[TEXT_COL] = test_df[TEXT_COL].fillna("").astype(str).apply(light_clean)

# Optional: extract numeric structured features from the text (quantity, weight)
print("Extracting quantity/weight heuristics (optional)...")
train_qw = train_df[TEXT_COL].apply(extract_quantity_and_weight)
train_df['qty'] = [x[0] for x in train_qw]
train_df['weight_g'] = [x[1] for x in train_qw]

if test_df is not None:
    test_qw = test_df[TEXT_COL].apply(extract_quantity_and_weight)
    test_df['qty'] = [x[0] for x in test_qw]
    test_df['weight_g'] = [x[1] for x in test_qw]
