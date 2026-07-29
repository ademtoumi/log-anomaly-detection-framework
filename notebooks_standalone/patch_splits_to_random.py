import os
import pathlib
import gc

STANDALONE_DIR = pathlib.Path(r'c:\Users\toumi\Desktop\(Anomaly detection)\notebooks_standalone')

def patch_svm_bgl(text):
    old_block1 = """    # Temporal split 60/20/20
    i1, i2 = int(n*0.60), int(n*0.80)
    y_train, y_val, y_test = labels[:i1], labels[i1:i2], labels[i2:]
    print(f"  Split → train={len(y_train):,} | val={len(y_val):,} | test={len(y_test):,}")

    # TF-IDF: fit on TRAIN only — no data leakage
    print("  Building TF-IDF features ...")
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train = vectorizer.fit_transform(templates[:i1]).astype(np.float32)
    X_val   = vectorizer.transform(templates[i1:i2]).astype(np.float32)
    X_test  = vectorizer.transform(templates[i2:]).astype(np.float32)"""

    new_block1 = """    # Stratified random split 70/10/20 (80/20 trainval vs test)
    from sklearn.model_selection import train_test_split
    indices = np.arange(n)
    train_val_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=labels)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.125, random_state=42, stratify=labels[train_val_idx])
    y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]
    print(f"  Split → train={len(y_train):,} | val={len(y_val):,} | test={len(y_test):,}")

    # TF-IDF: fit on TRAIN only — no data leakage
    print("  Building TF-IDF features ...")
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train = vectorizer.fit_transform(templates[train_idx]).astype(np.float32)
    X_val   = vectorizer.transform(templates[val_idx]).astype(np.float32)
    X_test  = vectorizer.transform(templates[test_idx]).astype(np.float32)"""

    old_block2 = """    i1, i2 = int(n*0.60), int(n*0.80)
    y_train, y_val, y_test = labels[:i1], labels[i1:i2], labels[i2:]
    X_train = vectorizer.transform(templates[:i1]).astype(np.float32)
    X_val   = vectorizer.transform(templates[i1:i2]).astype(np.float32)
    X_test  = vectorizer.transform(templates[i2:]).astype(np.float32)"""

    new_block2 = """    from sklearn.model_selection import train_test_split
    indices = np.arange(n)
    train_val_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=labels)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.125, random_state=42, stratify=labels[train_val_idx])
    y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]
    X_train = vectorizer.transform(templates[train_idx]).astype(np.float32)
    X_val   = vectorizer.transform(templates[val_idx]).astype(np.float32)
    X_test  = vectorizer.transform(templates[test_idx]).astype(np.float32)"""

    old_lime = """        # Reload raw templates just to retrieve the text format
        filepath = find_file('BGL_Drain.csv')
        df_log = pd.read_csv(filepath, usecols=['template'], on_bad_lines='skip', low_memory=False)
        templates_all = df_log['template'].fillna('').values
        del df_log; gc.collect()
        test_templates = templates_all[i2:]"""

    new_lime = """        # Reload raw templates just to retrieve the text format
        filepath = find_file('BGL_Drain.csv')
        df_log = pd.read_csv(filepath, usecols=['template', 'label'], on_bad_lines='skip', low_memory=False)
        templates_all = df_log['template'].fillna('').values
        labels_all = (df_log['label'].astype(str).str.strip() != '-').astype(int).values
        del df_log; gc.collect()
        from sklearn.model_selection import train_test_split
        indices_all = np.arange(len(labels_all))
        _, test_idx = train_test_split(indices_all, test_size=0.20, random_state=42, stratify=labels_all)
        test_templates = templates_all[test_idx]"""

    text = text.replace(old_block1, new_block1)
    text = text.replace(old_block2, new_block2)
    text = text.replace(old_lime, new_lime)
    return text

def patch_svm_spirit(text):
    old_block1 = """    # Temporal split 60/20/20
    i1, i2 = int(n*0.60), int(n*0.80)
    y_train, y_val, y_test = labels[:i1], labels[i1:i2], labels[i2:]
    print(f"  Split → train={len(y_train):,} | val={len(y_val):,} | test={len(y_test):,}")

    # Word2Vec training — fit on training templates only to avoid leakage
    print("  Tokenizing and training Word2Vec on train split...")
    tokenized_train = [str(log).lower().split() for log in templates[:i1]]
    w2v_model = Word2Vec(
        sentences=tokenized_train, vector_size=W2V_SIZE,
        window=5, min_count=2, workers=4, epochs=5
    )
    joblib.dump(w2v_model, f'{BASE_OUT}/models/w2v_{DS_KEY}_opt.pkl')
    print("  Word2Vec model trained and saved.")

    print("  Building dense Word2Vec average vector features...")
    tokenized_val  = [str(log).lower().split() for log in templates[i1:i2]]
    tokenized_test = [str(log).lower().split() for log in templates[i2:]]"""

    new_block1 = """    # Stratified random split 70/10/20 (80/20 trainval vs test)
    from sklearn.model_selection import train_test_split
    indices = np.arange(n)
    train_val_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=labels)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.125, random_state=42, stratify=labels[train_val_idx])
    y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]
    print(f"  Split → train={len(y_train):,} | val={len(y_val):,} | test={len(y_test):,}")

    # Word2Vec training — fit on training templates only to avoid leakage
    print("  Tokenizing and training Word2Vec on train split...")
    tokenized_train = [str(log).lower().split() for log in templates[train_idx]]
    w2v_model = Word2Vec(
        sentences=tokenized_train, vector_size=W2V_SIZE,
        window=5, min_count=2, workers=4, epochs=5
    )
    joblib.dump(w2v_model, f'{BASE_OUT}/models/w2v_{DS_KEY}_opt.pkl')
    print("  Word2Vec model trained and saved.")

    print("  Building dense Word2Vec average vector features...")
    tokenized_val  = [str(log).lower().split() for log in templates[val_idx]]
    tokenized_test = [str(log).lower().split() for log in templates[test_idx]]"""

    old_block2 = """    # Reload templates for explainability step
    filepath = find_file('Spirit_Drain.csv')
    all_templates = []
    rows_loaded = 0
    for chunk in pd.read_csv(filepath, usecols=['template'], chunksize=500_000, on_bad_lines='skip', low_memory=False):
        all_templates.extend(chunk['template'].fillna('').tolist())
        rows_loaded += len(chunk)
        if NROWS_LIMIT and rows_loaded >= NROWS_LIMIT: break
    templates = np.array(all_templates); del all_templates; gc.collect()
    n = len(templates)
    i1, i2 = int(n*0.60), int(n*0.80)"""

    new_block2 = """    # Reload templates for explainability step
    filepath = find_file('Spirit_Drain.csv')
    all_templates = []
    all_labels = []
    rows_loaded = 0
    for chunk in pd.read_csv(filepath, usecols=['template', 'label'], chunksize=500_000, on_bad_lines='skip', low_memory=False):
        all_templates.extend(chunk['template'].fillna('').tolist())
        all_labels.extend((chunk['label'].astype(str).str.strip() != '0').astype(int).tolist())
        rows_loaded += len(chunk)
        if NROWS_LIMIT and rows_loaded >= NROWS_LIMIT: break
    templates = np.array(all_templates); del all_templates
    labels = np.array(all_labels, dtype=np.int32); del all_labels; gc.collect()
    n = len(templates)
    from sklearn.model_selection import train_test_split
    indices = np.arange(n)
    train_val_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=labels)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.125, random_state=42, stratify=labels[train_val_idx])"""

    old_lime = """            target_log = str(templates[i2:][idx])"""
    new_lime = """            target_log = str(templates[test_idx][idx])"""

    text = text.replace(old_block1, new_block1)
    text = text.replace(old_block2, new_block2)
    text = text.replace(old_lime, new_lime)
    return text

def patch_rf_dt_if_bgl(text):
    # RF/DT/IF ML models on BGL
    old_block1 = """    # Temporal split 60/20/20
    i1, i2 = int(n*0.60), int(n*0.80)
    y_train, y_val, y_test = labels[:i1], labels[i1:i2], labels[i2:]
    print(f"  Split → train={len(y_train):,} | val={len(y_val):,} | test={len(y_test):,}")

    # TF-IDF: fit on TRAIN only — no data leakage [Bekkouche2024]
    print("  Building TF-IDF features ...")
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train = vectorizer.fit_transform(templates[:i1]).astype(np.float32)
    X_val   = vectorizer.transform(templates[i1:i2]).astype(np.float32)
    X_test  = vectorizer.transform(templates[i2:]).astype(np.float32)"""

    new_block1 = """    # Stratified random split 70/10/20 (80/20 trainval vs test)
    from sklearn.model_selection import train_test_split
    indices = np.arange(n)
    train_val_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=labels)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.125, random_state=42, stratify=labels[train_val_idx])
    y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]
    print(f"  Split → train={len(y_train):,} | val={len(y_val):,} | test={len(y_test):,}")

    # TF-IDF: fit on TRAIN only — no data leakage [Bekkouche2024]
    print("  Building TF-IDF features ...")
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train = vectorizer.fit_transform(templates[train_idx]).astype(np.float32)
    X_val   = vectorizer.transform(templates[val_idx]).astype(np.float32)
    X_test  = vectorizer.transform(templates[test_idx]).astype(np.float32)"""

    old_block2 = """    i1, i2 = int(n*0.60), int(n*0.80)
    y_train, y_val, y_test = labels[:i1], labels[i1:i2], labels[i2:]
    X_train = vectorizer.transform(templates[:i1]).astype(np.float32)
    X_val   = vectorizer.transform(templates[i1:i2]).astype(np.float32)
    X_test  = vectorizer.transform(templates[i2:]).astype(np.float32)"""

    new_block2 = """    from sklearn.model_selection import train_test_split
    indices = np.arange(n)
    train_val_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=labels)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.125, random_state=42, stratify=labels[train_val_idx])
    y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]
    X_train = vectorizer.transform(templates[train_idx]).astype(np.float32)
    X_val   = vectorizer.transform(templates[val_idx]).astype(np.float32)
    X_test  = vectorizer.transform(templates[test_idx]).astype(np.float32)"""

    # For DT BGL & IF BGL split logic in build_data
    old_dt_block = """    # Temporal split 60 / 20 / 20 (preserve time order – no shuffle)
    i60 = int(0.60 * n_total)
    i80 = int(0.80 * n_total)

    X_raw_train = df['template'].iloc[:i60].tolist()
    y_train     = df['label'].iloc[:i60].values.astype(np.int8)

    X_raw_val   = df['template'].iloc[i60:i80].tolist()
    y_val       = df['label'].iloc[i60:i80].values.astype(np.int8)

    X_raw_test  = df['template'].iloc[i80:].tolist()
    y_test      = df['label'].iloc[i80:].values.astype(np.int8)"""

    new_dt_block = """    # Stratified random split 70/10/20 (80/20 trainval vs test)
    from sklearn.model_selection import train_test_split
    indices = np.arange(n_total)
    train_val_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=df['label'].values)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.125, random_state=42, stratify=df['label'].values[train_val_idx])

    X_raw_train = df['template'].iloc[train_idx].tolist()
    y_train     = df['label'].iloc[train_idx].values.astype(np.int8)

    X_raw_val   = df['template'].iloc[val_idx].tolist()
    y_val       = df['label'].iloc[val_idx].values.astype(np.int8)

    X_raw_test  = df['template'].iloc[test_idx].tolist()
    y_test      = df['label'].iloc[test_idx].values.astype(np.int8)"""

    old_if_block = """    i60 = int(0.60 * n_total)
    i80 = int(0.80 * n_total)

    X_raw_train = df['template'].iloc[:i60].tolist()
    y_train     = df['label'].iloc[:i60].values.astype(np.int8)
    X_raw_val   = df['template'].iloc[i60:i80].tolist()
    y_val       = df['label'].iloc[i60:i80].values.astype(np.int8)
    X_raw_test  = df['template'].iloc[i80:].tolist()
    y_test      = df['label'].iloc[i80:].values.astype(np.int8)"""

    new_if_block = """    # Stratified random split 70/10/20 (80/20 trainval vs test)
    from sklearn.model_selection import train_test_split
    indices = np.arange(n_total)
    train_val_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=df['label'].values)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.125, random_state=42, stratify=df['label'].values[train_val_idx])

    X_raw_train = df['template'].iloc[train_idx].tolist()
    y_train     = df['label'].iloc[train_idx].values.astype(np.int8)
    X_raw_val   = df['template'].iloc[val_idx].tolist()
    y_val       = df['label'].iloc[val_idx].values.astype(np.int8)
    X_raw_test  = df['template'].iloc[test_idx].tolist()
    y_test      = df['label'].iloc[test_idx].values.astype(np.int8)"""

    text = text.replace(old_block1, new_block1)
    text = text.replace(old_block2, new_block2)
    text = text.replace(old_dt_block, new_dt_block)
    text = text.replace(old_if_block, new_if_block)
    return text

def patch_rf_dt_if_spirit(text):
    # RF/DT/IF ML models on Spirit
    old_block1 = """    # Temporal split 60/20/20
    i1, i2 = int(n*0.60), int(n*0.80)
    y_train, y_val, y_test = labels[:i1], labels[i1:i2], labels[i2:]
    print(f"  Split → train={len(y_train):,} | val={len(y_val):,} | test={len(y_test):,}")

    # TF-IDF: fit on TRAIN only — no data leakage [Bekkouche2024]
    print("  Building TF-IDF features ...")
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train = vectorizer.fit_transform(templates[:i1]).astype(np.float32)
    X_val   = vectorizer.transform(templates[i1:i2]).astype(np.float32)
    X_test  = vectorizer.transform(templates[i2:]).astype(np.float32)"""

    new_block1 = """    # Stratified random split 70/10/20 (80/20 trainval vs test)
    from sklearn.model_selection import train_test_split
    indices = np.arange(n)
    train_val_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=labels)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.125, random_state=42, stratify=labels[train_val_idx])
    y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]
    print(f"  Split → train={len(y_train):,} | val={len(y_val):,} | test={len(y_test):,}")

    # TF-IDF: fit on TRAIN only — no data leakage [Bekkouche2024]
    print("  Building TF-IDF features ...")
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train = vectorizer.fit_transform(templates[train_idx]).astype(np.float32)
    X_val   = vectorizer.transform(templates[val_idx]).astype(np.float32)
    X_test  = vectorizer.transform(templates[test_idx]).astype(np.float32)"""

    old_block2 = """    n = len(labels); i1, i2 = int(n*0.60), int(n*0.80)
    y_train, y_val, y_test = labels[:i1], labels[i1:i2], labels[i2:]
    X_train = vectorizer.transform(templates[:i1]).astype(np.float32)
    X_val   = vectorizer.transform(templates[i1:i2]).astype(np.float32)
    X_test  = vectorizer.transform(templates[i2:]).astype(np.float32)"""

    new_block2 = """    from sklearn.model_selection import train_test_split
    indices = np.arange(n)
    train_val_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=labels)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.125, random_state=42, stratify=labels[train_val_idx])
    y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]
    X_train = vectorizer.transform(templates[train_idx]).astype(np.float32)
    X_val   = vectorizer.transform(templates[val_idx]).astype(np.float32)
    X_test  = vectorizer.transform(templates[test_idx]).astype(np.float32)"""

    # For DT Spirit & IF Spirit split logic in build_data
    old_dt_block = """    # Temporal split 60 / 20 / 20
    i60 = int(0.60 * n_total)
    i80 = int(0.80 * n_total)

    X_raw_train = df['template'].iloc[:i60].tolist()
    y_train     = df['label'].iloc[:i60].values.astype(np.int8)
    X_raw_val   = df['template'].iloc[i60:i80].tolist()
    y_val       = df['label'].iloc[i60:i80].values.astype(np.int8)
    X_raw_test  = df['template'].iloc[i80:].tolist()
    y_test      = df['label'].iloc[i80:].values.astype(np.int8)"""

    new_dt_block = """    # Stratified random split 70/10/20 (80/20 trainval vs test)
    from sklearn.model_selection import train_test_split
    indices = np.arange(n_total)
    train_val_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=df['label'].values)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.125, random_state=42, stratify=df['label'].values[train_val_idx])

    X_raw_train = df['template'].iloc[train_idx].tolist()
    y_train     = df['label'].iloc[train_idx].values.astype(np.int8)
    X_raw_val   = df['template'].iloc[val_idx].tolist()
    y_val       = df['label'].iloc[val_idx].values.astype(np.int8)
    X_raw_test  = df['template'].iloc[test_idx].tolist()
    y_test      = df['label'].iloc[test_idx].values.astype(np.int8)"""

    old_if_block = """    i60 = int(0.60 * n_total)
    i80 = int(0.80 * n_total)

    X_raw_train = df['template'].iloc[:i60].tolist()
    y_train     = df['label'].iloc[:i60].values.astype(np.int8)
    X_raw_val   = df['template'].iloc[i60:i80].tolist()
    y_val       = df['label'].iloc[i60:i80].values.astype(np.int8)
    X_raw_test  = df['template'].iloc[i80:].tolist()
    y_test      = df['label'].iloc[i80:].values.astype(np.int8)"""

    new_if_block = """    # Stratified random split 70/10/20 (80/20 trainval vs test)
    from sklearn.model_selection import train_test_split
    indices = np.arange(n_total)
    train_val_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=df['label'].values)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.125, random_state=42, stratify=df['label'].values[train_val_idx])

    X_raw_train = df['template'].iloc[train_idx].tolist()
    y_train     = df['label'].iloc[train_idx].values.astype(np.int8)
    X_raw_val   = df['template'].iloc[val_idx].tolist()
    y_val       = df['label'].iloc[val_idx].values.astype(np.int8)
    X_raw_test  = df['template'].iloc[test_idx].tolist()
    y_test      = df['label'].iloc[test_idx].values.astype(np.int8)"""

    text = text.replace(old_block1, new_block1)
    text = text.replace(old_block2, new_block2)
    text = text.replace(old_dt_block, new_dt_block)
    text = text.replace(old_if_block, new_if_block)
    return text

def patch_dense_ae_bgl(text):
    old_block1 = """    # Temporal split 60/20/20
    i1, i2 = int(n * 0.60), int(n * 0.80)
    y_train, y_val, y_test = labels[:i1], labels[i1:i2], labels[i2:]
    print(f"  Split → train={len(y_train):,} | val={len(y_val):,} | test={len(y_test):,}")

    print("  Building TF-IDF features ...")
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train_sp = vectorizer.fit_transform(templates[:i1])
    X_val_sp   = vectorizer.transform(templates[i1:i2])
    X_test_sp  = vectorizer.transform(templates[i2:])"""

    new_block1 = """    # Stratified random split 70/10/20 (80/20 trainval vs test)
    from sklearn.model_selection import train_test_split
    indices = np.arange(n)
    train_val_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=labels)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.125, random_state=42, stratify=labels[train_val_idx])
    y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]
    print(f"  Split → train={len(y_train):,} | val={len(y_val):,} | test={len(y_test):,}")

    print("  Building TF-IDF features ...")
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train_sp = vectorizer.fit_transform(templates[train_idx])
    X_val_sp   = vectorizer.transform(templates[val_idx])
    X_test_sp  = vectorizer.transform(templates[test_idx])"""

    old_block2 = """    i1, i2 = int(n * 0.60), int(n * 0.80)
    y_train, y_val, y_test = labels[:i1], labels[i1:i2], labels[i2:]
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train_sp = vectorizer.fit_transform(templates[:i1])
    X_val_sp   = vectorizer.transform(templates[i1:i2])
    X_test_sp  = vectorizer.transform(templates[i2:])"""

    new_block2 = """    from sklearn.model_selection import train_test_split
    indices = np.arange(n)
    train_val_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=labels)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.125, random_state=42, stratify=labels[train_val_idx])
    y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train_sp = vectorizer.fit_transform(templates[train_idx])
    X_val_sp   = vectorizer.transform(templates[val_idx])
    X_test_sp  = vectorizer.transform(templates[test_idx])"""

    text = text.replace(old_block1, new_block1)
    text = text.replace(old_block2, new_block2)
    return text

def patch_dense_ae_spirit(text):
    old_block1 = """    # Temporal split 60/20/20 — preserving log order
    n = len(labels)
    i1, i2 = int(n * 0.60), int(n * 0.80)
    y_train = labels[:i1].astype(np.int32)
    y_val   = labels[i1:i2].astype(np.int32)
    y_test  = labels[i2:].astype(np.int32)
    print(f"  Split → train={len(y_train):,} | val={len(y_val):,} | test={len(y_test):,}")

    # TF-IDF (fit on TRAIN only)
    print("  Building TF-IDF features ...")
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train_sp = vectorizer.fit_transform(templates[:i1])
    X_val_sp   = vectorizer.transform(templates[i1:i2])
    X_test_sp  = vectorizer.transform(templates[i2:])"""

    new_block1 = """    # Stratified random split 70/10/20 (80/20 trainval vs test)
    from sklearn.model_selection import train_test_split
    n = len(labels)
    indices = np.arange(n)
    train_val_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=labels)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.125, random_state=42, stratify=labels[train_val_idx])
    y_train = labels[train_idx].astype(np.int32)
    y_val   = labels[val_idx].astype(np.int32)
    y_test  = labels[test_idx].astype(np.int32)
    print(f"  Split → train={len(y_train):,} | val={len(y_val):,} | test={len(y_test):,}")

    # TF-IDF (fit on TRAIN only)
    print("  Building TF-IDF features ...")
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train_sp = vectorizer.fit_transform(templates[train_idx])
    X_val_sp   = vectorizer.transform(templates[val_idx])
    X_test_sp  = vectorizer.transform(templates[test_idx])"""

    text = text.replace(old_block1, new_block1)
    return text

def patch_dl_sliding_window(text, model_name, ds_name):
    # For DL models building sessions from sliding windows: BiLSTM, CNN+BiLSTM, LSTM AE, DeepLog
    # First: Vocab build training set range (80% instead of 60%)
    old_vocab1 = """    # Build vocabulary from training portion only (60% of lines) to avoid leakage
    i1_lines = int(n_total * 0.60)
    unique_t = sorted(set(all_templates[:i1_lines]))"""
    
    new_vocab1 = """    # Build vocabulary from training + validation portion (first 80% of lines) to avoid leakage
    i1_lines = int(n_total * 0.80)
    unique_t = sorted(set(all_templates[:i1_lines]))"""

    old_vocab2 = """    # Build vocabulary from training portion only (60% of lines)
    i1_lines = int(n_total * 0.60)
    unique_t = sorted(set(all_templates[:i1_lines]))"""

    new_vocab2 = """    # Build vocabulary from training + validation portion (first 80% of lines) to avoid leakage
    i1_lines = int(n_total * 0.80)
    unique_t = sorted(set(all_templates[:i1_lines]))"""

    # For 09 CNN+BiLSTM:
    old_vocab3 = """    # ── Build vocabulary (fit on TRAIN corpus to avoid data leakage) ──────────
    print("  Building vocabulary from training split only (leak-safe) ...")
    token_freq = defaultdict(int)
    n_train = int(len(templates) * 0.60)
    for t in templates[:n_train]:"""

    new_vocab3 = """    # ── Build vocabulary (fit on TRAIN+VAL corpus to avoid data leakage) ──────────
    print("  Building vocabulary from training+validation split (leak-safe) ...")
    token_freq = defaultdict(int)
    n_train = int(len(templates) * 0.80)
    for t in templates[:n_train]:"""

    # For 08 BiLSTM:
    old_vocab4 = """    # 1b. Build vocabulary (train split only — no leakage)
    print("  Building event vocabulary from training split ...")
    i1_lines = int(len(df_log) * 0.60)
    unique_t = sorted(set(df_log['template'].iloc[:i1_lines].fillna('unknown').astype(str)))"""

    new_vocab4 = """    # 1b. Build vocabulary (train+val split — no leakage)
    print("  Building event vocabulary from training+validation split ...")
    i1_lines = int(len(df_log) * 0.80)
    unique_t = sorted(set(df_log['template'].iloc[:i1_lines].fillna('unknown').astype(str)))"""

    text = text.replace(old_vocab1, new_vocab1)
    text = text.replace(old_vocab2, new_vocab2)
    text = text.replace(old_vocab3, new_vocab3)
    text = text.replace(old_vocab4, new_vocab4)

    # Second: Slices replacement
    # 08 BiLSTM Spirit:
    old_split_08 = """    # ------------------------------------------------------------------
    # 1e. Temporal split 60 / 20 / 20
    # ------------------------------------------------------------------
    n   = len(X_seq)
    t1  = int(n * 0.60)
    t2  = int(n * 0.80)

    X_train, y_train = X_seq[:t1],    y_win[:t1]
    X_val,   y_val   = X_seq[t1:t2],  y_win[t1:t2]
    X_test,  y_test  = X_seq[t2:],    y_win[t2:]"""

    new_split_08 = """    # ------------------------------------------------------------------
    # 1e. Stratified random split 70 / 10 / 20
    # ------------------------------------------------------------------
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(X_seq))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=y_win)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.125, random_state=42, stratify=y_win[train_val_idx])

    X_train, y_train = X_seq[train_idx], y_win[train_idx]
    X_val,   y_val   = X_seq[val_idx],   y_win[val_idx]
    X_test,  y_test  = X_seq[test_idx],  y_win[test_idx]"""

    # 12 LSTM AE Spirit:
    old_split_12 = """    i1 = int(n_windows * 0.60)
    i2 = int(n_windows * 0.80)

    np.savez_compressed(f'{MODEL_DIR}/spirit_sessions_train_lstmae.npz', X=sequences[:i1], y=labels[:i1])
    np.savez_compressed(f'{MODEL_DIR}/spirit_sessions_val_lstmae.npz',   X=sequences[i1:i2], y=labels[i1:i2])
    np.savez_compressed(f'{MODEL_DIR}/spirit_sessions_test_lstmae.npz',  X=sequences[i2:], y=labels[i2:])

    print(f"  Train: {i1:,} | Val: {i2-i1:,} | Test: {n_windows-i2:,}")"""

    new_split_12 = """    from sklearn.model_selection import train_test_split
    indices = np.arange(n_windows)
    train_val_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=labels)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.125, random_state=42, stratify=labels[train_val_idx])

    np.savez_compressed(f'{MODEL_DIR}/spirit_sessions_train_lstmae.npz', X=sequences[train_idx], y=labels[train_idx])
    np.savez_compressed(f'{MODEL_DIR}/spirit_sessions_val_lstmae.npz',   X=sequences[val_idx],   y=labels[val_idx])
    np.savez_compressed(f'{MODEL_DIR}/spirit_sessions_test_lstmae.npz',  X=sequences[test_idx],  y=labels[test_idx])

    print(f"  Train: {len(train_idx):,} | Val: {len(val_idx):,} | Test: {len(test_idx):,}")"""

    # 15 BiLSTM AE BGL:
    old_split_15 = """    i1 = int(n_windows * 0.60)
    i2 = int(n_windows * 0.80)

    np.savez_compressed(f'{MODEL_DIR}/bgl_sessions_train_lstmae.npz', X=sequences[:i1], y=labels[:i1])
    np.savez_compressed(f'{MODEL_DIR}/bgl_sessions_val_lstmae.npz',   X=sequences[i1:i2], y=labels[i1:i2])
    np.savez_compressed(f'{MODEL_DIR}/bgl_sessions_test_lstmae.npz',  X=sequences[i2:], y=labels[i2:])

    print(f"  Train: {i1:,} | Val: {i2-i1:,} | Test: {n_windows-i2:,}")"""

    new_split_15 = """    from sklearn.model_selection import train_test_split
    indices = np.arange(n_windows)
    train_val_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=labels)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.125, random_state=42, stratify=labels[train_val_idx])

    np.savez_compressed(f'{MODEL_DIR}/bgl_sessions_train_lstmae.npz', X=sequences[train_idx], y=labels[train_idx])
    np.savez_compressed(f'{MODEL_DIR}/bgl_sessions_val_lstmae.npz',   X=sequences[val_idx],   y=labels[val_idx])
    np.savez_compressed(f'{MODEL_DIR}/bgl_sessions_test_lstmae.npz',  X=sequences[test_idx],  y=labels[test_idx])

    print(f"  Train: {len(train_idx):,} | Val: {len(val_idx):,} | Test: {len(test_idx):,}")"""

    # 17 DeepLog BGL:
    old_split_17 = """    i1 = int(n_windows * 0.60)
    i2 = int(n_windows * 0.80)

    np.savez_compressed(f'{MODEL_DIR}/bgl_sessions_train_deeplog.npz', X=sequences[:i1], y=labels[:i1])
    np.savez_compressed(f'{MODEL_DIR}/bgl_sessions_val_deeplog.npz',   X=sequences[i1:i2], y=labels[i1:i2])
    np.savez_compressed(f'{MODEL_DIR}/bgl_sessions_test_deeplog.npz',  X=sequences[i2:], y=labels[i2:])

    print(f"  Train: {i1:,} | Val: {i2-i1:,} | Test: {n_windows-i2:,}")"""

    new_split_17 = """    from sklearn.model_selection import train_test_split
    indices = np.arange(n_windows)
    train_val_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=labels)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.125, random_state=42, stratify=labels[train_val_idx])

    np.savez_compressed(f'{MODEL_DIR}/bgl_sessions_train_deeplog.npz', X=sequences[train_idx], y=labels[train_idx])
    np.savez_compressed(f'{MODEL_DIR}/bgl_sessions_val_deeplog.npz',   X=sequences[val_idx],   y=labels[val_idx])
    np.savez_compressed(f'{MODEL_DIR}/bgl_sessions_test_deeplog.npz',  X=sequences[test_idx],  y=labels[test_idx])

    print(f"  Train: {len(train_idx):,} | Val: {len(val_idx):,} | Test: {len(test_idx):,}")"""

    text = text.replace(old_split_08, new_split_08)
    text = text.replace(old_split_12, new_split_12)
    text = text.replace(old_split_15, new_split_15)
    text = text.replace(old_split_17, new_split_17)

    # 09 CNN+BiLSTM Spirit is a special loop
    old_split_loop_09 = """    # ── Temporal split indices (60/20/20) ────────────────────────────────────
    n = len(token_ids)
    i1 = int(n * 0.60)
    i2 = int(n * 0.80)

    splits = {
        'train': (0,  i1),
        'val':   (i1, i2),
        'test':  (i2, n),
    }
    print(f"  Split → train={i1:,} | val={i2-i1:,} | test={n-i2:,} rows")

    # ── Build sliding-window sequences per split ──────────────────────────────
    # [Bekkouche2025_Spirit]: window label = 1 if ANY event in window is anomaly
    PAD_ID = vocab['<PAD>']

    for split_name, (lo, hi) in splits.items():
        X_win, y_win = [], []
        tids_split   = token_ids[lo:hi]
        labs_split   = labels[lo:hi]

        for start in range(0, len(tids_split) - WINDOW_SIZE + 1, STEP_SIZE):
            end = start + WINDOW_SIZE
            window_toks = list(itertools.chain.from_iterable(tids_split[start:end]))
            window_lab  = int(any(labs_split[start:end]))

            # Truncate or pad to fixed length WINDOW_SIZE (one tok per event)
            # We use one representative token per log line (first token)
            seq = [tids_split[i][0] if tids_split[i] else PAD_ID
                   for i in range(start, end)]
            X_win.append(seq)
            y_win.append(window_lab)

        X_arr = np.array(X_win, dtype=np.int32)
        y_arr = np.array(y_win, dtype=np.int32)
        np.savez_compressed(
            f'{MODEL_DIR}/spirit_sessions_{split_name}.npz',
            X=X_arr, y=y_arr)
        print(f"  ✅ {split_name}: {X_arr.shape} | "
              f"anomaly={y_arr.mean()*100:.1f}%")
        del X_win, y_win, X_arr, y_arr; gc.collect()"""

    new_split_loop_09 = """    # ── Build sliding-window sequences on entire log stream first ─────────────
    # [Bekkouche2025_Spirit]: window label = 1 if ANY event in window is anomaly
    PAD_ID = vocab['<PAD>']
    X_win, y_win = [], []
    for start in range(0, len(token_ids) - WINDOW_SIZE + 1, STEP_SIZE):
        end = start + WINDOW_SIZE
        window_lab  = int(any(labels[start:end]))
        seq = [token_ids[i][0] if token_ids[i] else PAD_ID
               for i in range(start, end)]
        X_win.append(seq)
        y_win.append(window_lab)

    X_arr = np.array(X_win, dtype=np.int32)
    y_arr = np.array(y_win, dtype=np.int32)

    # ── Stratified random split 70/10/20 ────────────────────────────────────
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(X_arr))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=y_arr)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.125, random_state=42, stratify=y_arr[train_val_idx])

    # Save splits
    np.savez_compressed(f'{MODEL_DIR}/spirit_sessions_train.npz', X=X_arr[train_idx], y=y_arr[train_idx])
    np.savez_compressed(f'{MODEL_DIR}/spirit_sessions_val.npz',   X=X_arr[val_idx],   y=y_arr[val_idx])
    np.savez_compressed(f'{MODEL_DIR}/spirit_sessions_test.npz',  X=X_arr[test_idx],  y=y_arr[test_idx])

    print(f"  ✅ train: {len(train_idx):,} | anomaly={y_arr[train_idx].mean()*100:.1f}%")
    print(f"  ✅ val: {len(val_idx):,} | anomaly={y_arr[val_idx].mean()*100:.1f}%")
    print(f"  ✅ test: {len(test_idx):,} | anomaly={y_arr[test_idx].mean()*100:.1f}%")
    del X_win, y_win, X_arr, y_arr; gc.collect()"""

    text = text.replace(old_split_loop_09, new_split_loop_09)

    return text

def main():
    targets = [
        "03_svm_bgl_standalone.py",
        "03_svm_spirit_standalone.py",
        "04_rf_bgl_standalone.py",
        "04_rf_spirit_standalone.py",
        "05_dt_bgl_standalone.py",
        "05_dt_spirit_standalone.py",
        "08_bilstm_spirit_standalone.py",
        "09_cnn_bilstm_spirit_standalone.py",
        "10_isolation_forest_bgl_standalone.py",
        "10_isolation_forest_spirit_standalone.py",
        "11_dense_ae_bgl_standalone.py",
        "11_dense_ae_spirit_standalone.py",
        "12_lstm_ae_spirit_standalone.py",
        "15_bilstm_ae_bgl_standalone.py",
        "17_deeplog_bgl_standalone.py"
    ]

    patched = []
    for target in targets:
        fpath = STANDALONE_DIR / target
        if not fpath.exists():
            print(f"[WARN] target file {target} does not exist!")
            continue

        orig_text = fpath.read_text(encoding='utf-8')
        text = orig_text
        
        # Apply correct patching function based on file name/patterns
        if target == "03_svm_bgl_standalone.py":
            text = patch_svm_bgl(text)
        elif target == "03_svm_spirit_standalone.py":
            text = patch_svm_spirit(text)
        elif "bgl" in target and any(x in target for x in ["rf", "dt", "isolation_forest"]):
            text = patch_rf_dt_if_bgl(text)
        elif "spirit" in target and any(x in target for x in ["rf", "dt", "isolation_forest"]):
            text = patch_rf_dt_if_spirit(text)
        elif target == "11_dense_ae_bgl_standalone.py":
            text = patch_dense_ae_bgl(text)
        elif target == "11_dense_ae_spirit_standalone.py":
            text = patch_dense_ae_spirit(text)
        
        # Apply deep learning sliding window patch if applicable
        if any(x in target for x in ["bilstm", "cnn", "lstm_ae", "deeplog"]) and not "dense_ae" in target:
            text = patch_dl_sliding_window(text, target, target)

        if text != orig_text:
            fpath.write_text(text, encoding='utf-8')
            patched.append(target)
            print(f"[OK] Patched {target}")
        else:
            print(f"[FAIL] Failed to patch {target} (no matches found)")

    print(f"\nDone! Patched {len(patched)} / {len(targets)} files.")

if __name__ == "__main__":
    main()
