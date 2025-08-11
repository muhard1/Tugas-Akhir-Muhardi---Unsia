# import dataset
import pandas as pd
import os

file_path = '/kaggle/input/smiles-gpu/DatasetAll.xlsx'
sheet_name = 'dataset'

if os.path.exists(file_path):
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
        print(f"Successfully loaded Excel dataset from: {file_path} (sheet: {sheet_name})")
        print("DataFrame head:")
        print(df.head())
    except Exception as e:
        print(f"An error occurred while reading the Excel file: {e}")
else:
    print(f"Error: File not found at path '{file_path}'. Please check the filename and directory.")


# Preposesing data
# mengecek data Null, kosong, Nan

import pandas as pd
import numpy as np

nan_count = df['hazard_name'].isna().sum()

unique_hazard_count = df['hazard_name'].nunique()

print(f"Jumlah jenis data unik pada kolom 'links_hazard_name': {unique_hazard_count}")
print(f"Jumlah nilai NaN pada kolom 'links_hazard_name': {nan_count}\n")

unique_hazard_list = df['links_hazard_name'].dropna().unique().tolist()

print("List jenis data unik pada kolom 'links_hazard_name':")
print(unique_hazard_list)
print(f"\nTotal jenis unik: {len(unique_hazard_list)}")

hazard_counts = df['links_hazard_name'].value_counts(dropna=False)

count_df = pd.DataFrame({
    'Jenis Data': hazard_counts.index,
    'Jumlah': hazard_counts.values
})

count_df['Jenis Data'] = count_df['Jenis Data'].apply(lambda x: 'NaN' if pd.isna(x) else x)

print("\nJumlah setiap jenis data pada kolom 'links_hazard_name':")
print(count_df)
print(f"\nTotal baris data: {len(df)}")
print(f"Total nilai non-NaN: {len(df) - nan_count}")

# Menganti nilai Nan, kosong, Null pada kolom hazard name menjad No Danger

empty_string_count = (df['hazard_name'] == '').sum()
nan_count = df['hazard_name'].isna().sum()

print(f"Jumlah string kosong pada kolom 'links_hazard_name': {empty_string_count}")
print(f"Jumlah nilai NaN pada kolom 'links_hazard_name': {nan_count}")

df['hazard_name'] = df['hazard_name'].replace('', 'No Danger')
df['hazard_name'] = df['hazard_name'].fillna('No Danger')

print("\nString kosong dan nilai NaN pada kolom 'links_hazard_name' telah diganti dengan 'No Danger'.")

print("\nJumlah setiap jenis data pada kolom 'links_hazard_name' setelah penggantian:")
print(df['hazard_name'].value_counts())

no_danger_count = (df['hazard_name'] == 'No Danger').sum()
print(f"\nTotal entri 'No Danger': {no_danger_count} (termasuk {empty_string_count} string kosong + {nan_count} NaN)")

# melakukan one hot encoding pada dataset

original_columns = [col for col in df.columns if col != 'hazard_name']

hazard_categories = df['hazard_name'].unique().tolist()

df_hazard = pd.get_dummies(df['hazard_name'], dtype=int)

df_combined = pd.concat([df, df_hazard], axis=1)

agg_dict = {col: 'first' for col in original_columns}
agg_dict.update({hazard: 'max' for hazard in hazard_categories})

df_grouped = df_combined.groupby('links_SMILE', as_index=False).agg(agg_dict)

df_grouped[hazard_categories] = df_grouped[hazard_categories].astype(int)

ordered_columns = original_columns + [hazard for hazard in hazard_categories if hazard not in original_columns]
df = df_grouped[ordered_columns]

display(df.head())

# Proses Ekstraksi fitur-fitur yang dibutuhkan dengan RDKIT

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Fragments
import numpy as np

def extract_features(smiles, radius=3, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    num_single = num_double = num_triple = num_aromatic = num_ring = 0
    for bond in mol.GetBonds():
        btype = bond.GetBondType()
        if btype.name == "SINGLE": num_single += 1
        elif btype.name == "DOUBLE": num_double += 1
        elif btype.name == "TRIPLE": num_triple += 1
        elif btype.name == "AROMATIC": num_aromatic += 1
        if bond.IsInRing(): num_ring += 1

    fp = AllChem.GetHashedMorganFingerprint(mol, radius=radius, nBits=nBits)
    fp_array = np.zeros((nBits,), dtype=int)
    for idx, val in fp.GetNonzeroElements().items():
        if idx < nBits:
            fp_array[idx] = val
    fp_list = fp_array.tolist()

    return {
        "MolWt": Descriptors.MolWt(mol),
        "MolLogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "FractionCSP3": Descriptors.FractionCSP3(mol),
        "RingCount": Descriptors.RingCount(mol),
        "Nitro": Fragments.fr_nitro(mol),
        "Halogen": Fragments.fr_halogen(mol),
        "Phenol": Fragments.fr_phenol(mol),
        "PrimaryAmine": Fragments.fr_NH2(mol),
        "SecondaryAmine": Fragments.fr_NH1(mol),
        "TertiaryAmine": Fragments.fr_NH0(mol),
        "NumBonds": mol.GetNumBonds(),
        "NumSingleBonds": num_single,
        "NumDoubleBonds": num_double,
        "NumTripleBonds": num_triple,
        "NumAromaticBonds": num_aromatic,
        "NumRingBonds": num_ring,
        "NumAromaticRings": Descriptors.NumAromaticRings(mol),
        "MorganFP": fp_list
    }

if 'links_SMILE' not in df.columns:
    raise ValueError("Kolom 'links_SMILE' tidak ditemukan dalam DataFrame.")

features = df['links_SMILE'].apply(extract_features)

valid_mask = features.notnull()
df = df[valid_mask].reset_index(drop=True)
features = features[valid_mask].reset_index(drop=True)

df_features = pd.DataFrame(features.tolist())

fp_columns = df_features['MorganFP'].apply(pd.Series)
fp_columns = fp_columns.rename(columns=lambda x: f'morgan_{x}')
df_features = df_features.drop(columns=['MorganFP'])
df_features = pd.concat([df_features, fp_columns], axis=1)

insert_pos = df.columns.get_loc("links_SMILE") + 1
df_before = df.iloc[:, :insert_pos]
df_after = df.iloc[:, insert_pos:]
df_final = pd.concat([df_before, df_features, df_after], axis=1)

# Melakukan Feature Selection dengan menggunakan Mutual Information
from sklearn.feature_selection import mutual_info_classif
import numpy as np
import pandas as pd

label_cols = [ 'No Danger', 'Corrosive', 'Irritant', 'Acute Toxic', 'Health Hazard', 'Environmental Hazard', 'Flammable', 'Compressed Gas', 'Explosive', 'Oxidizer']
feature_cols = [col for col in df_final.columns if col not in label_cols]
X = df_final[feature_cols].values
Y = df_final[label_cols].values

Y_string = pd.Series(['-'.join(row.astype(str)) for row in Y])
y_powerset = pd.factorize(Y_string)[0]

print("Menghitung Mutual Information untuk semua fitur...")
mi_scores = mutual_info_classif(X, y_powerset, discrete_features='auto')
print("Selesai.")

mi_df = pd.DataFrame({'Feature': feature_cols, 'MI_Score': mi_scores})
mi_df = mi_df.sort_values(by='MI_Score', ascending=False).reset_index(drop=True)

percentages = [20, 40, 60, 80]
for p in percentages:
    num_features = int(len(feature_cols) * (p / 100))
    # Buat nama variabel secara dinamis (e.g., selected_features_powerset_20)
    var_name = f"selected_features_powerset_{p}"
    # Ambil list fitur dan simpan ke dalam variabel global
    globals()[var_name] = mi_df['Feature'].tolist()[:num_features]

    print(f"\n--- Top {p}% Fitur Terpilih ({num_features} fitur) ---")
    # Hanya menampilkan beberapa fitur pertama untuk singkatnya
    print(globals()[var_name][:10]) 

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import pandas as pd
import numpy as np

label_cols = [
    'No Danger', 'Corrosive', 'Irritant', 'Acute Toxic', 
    'Health Hazard', 'Environmental Hazard', 'Flammable', 
    'Compressed Gas', 'Explosive', 'Oxidizer'
]

def split_data(selected_features, df_final, label_cols, test_size=0.25, random_state=42):
    X = df_final[selected_features].values
    y = df_final[label_cols].values

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(msss.split(X, y))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"\n=== Varian {len(selected_features)} Fitur ===")
    print(f"Ukuran X_train: {X_train.shape}")
    print(f"Ukuran y_train: {y_train.shape}")
    print(f"Ukuran X_test: {X_test.shape}")
    print(f"Ukuran y_test: {y_test.shape}")

    print("\nDistribusi label di data training:")
    print(pd.DataFrame(y_train, columns=label_cols).sum())

    print("\nDistribusi label di data test:")
    print(pd.DataFrame(y_test, columns=label_cols).sum())

    return X_train, X_test, y_train, y_test

X_train_20, X_test_20, y_train_20, y_test_20 = split_data(selected_features_powerset_20, df_final, label_cols)
X_train_40, X_test_40, y_train_40, y_test_40 = split_data(selected_features_powerset_40, df_final, label_cols)
X_train_60, X_test_60, y_train_60, y_test_60 = split_data(selected_features_powerset_60, df_final, label_cols)
X_train_80, X_test_80, y_train_80, y_test_80 = split_data(selected_features_powerset_80, df_final, label_cols)
X_train_100, X_test_100, y_train_100, y_test_100 = split_data(selected_features_powerset_100, df_final, label_cols)

# Melakukan penyeimbangan data dengan MLSMOTE
import pandas as pd
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors

def get_tail_label(df):
    columns = df.columns
    n = len(columns)
    irpl = np.zeros(n)
    for column in range(n):
        irpl[column] = df[columns[column]].value_counts().get(1, 0)
    irpl = np.divide(max(irpl), irpl, out=np.full_like(irpl, 0), where=irpl != 0)
    mir = np.average(irpl[irpl > 0])
    tail_label = [columns[i] for i in range(n) if irpl[i] > mir]
    return tail_label

def get_minority_instance(X, y):
    tail_labels = get_tail_label(y)
    index = set()
    for tail_label in tail_labels:
        sub_index = set(y[y[tail_label] == 1].index)
        index = index.union(sub_index)
    if not index:
        return pd.DataFrame(columns=X.columns), pd.DataFrame(columns=y.columns)
    return X.loc[list(index)], y.loc[list(index)]

def nearest_neighbour(X):
    nbs = NearestNeighbors(n_neighbors=5, metric='euclidean', algorithm='kd_tree').fit(X)
    _, indices = nbs.kneighbors(X)
    return indices

def MLSMOTE(X, y, n_sample):
    indices2 = nearest_neighbour(X)
    n = len(indices2)
    new_X = np.zeros((n_sample, X.shape[1]))
    target = np.zeros((n_sample, y.shape[1]))

    for i in range(n_sample):
        reference = random.randint(0, n - 1)
        neighbour = random.choice(indices2[reference, 1:])
        all_point = indices2[reference]
        nn_df = y.iloc[all_point]
        ser = nn_df.sum(axis=0, skipna=True)
        target[i] = np.array([1 if val > 2 else 0 for val in ser])
        ratio = random.random()
        gap = X.iloc[reference, :] - X.iloc[neighbour, :]
        new_X[i] = np.array(X.iloc[reference, :] + ratio * gap)

    new_X = pd.DataFrame(new_X, columns=X.columns)
    target = pd.DataFrame(target, columns=y.columns)
    return pd.concat([X, new_X], axis=0), pd.concat([y, target], axis=0)

def run_mlsmote_for_variant(X_train, y_train, selected_features, label_cols, n_new_samples=1500):
    X_train = pd.DataFrame(X_train, columns=selected_features)
    y_train = pd.DataFrame(y_train, columns=label_cols, dtype=int)

    print(f"\n=== Varian {len(selected_features)} fitur ===")
    print("Distribusi label sebelum MLSMOTE:")
    print(y_train.sum())

    X_min, y_min = get_minority_instance(X_train, y_train)
    print(f"\nJumlah sampel minoritas: {X_min.shape[0]}")

    X_min, y_min = MLSMOTE(X_min, y_min, n_new_samples)
    print(f"Berhasil menambahkan {n_new_samples} sampel sintetis.")

    non_minority_idx = list(set(X_train.index) - set(X_min.index))
    X_train = pd.concat([X_train.loc[non_minority_idx], X_min], axis=0)
    y_train = pd.concat([y_train.loc[non_minority_idx], y_min], axis=0)

    print("\nDistribusi label setelah MLSMOTE:")
    print(y_train.sum())

    print("\nBentuk akhir data training:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")

    return X_train, y_train

X_train_20, y_train_20 = run_mlsmote_for_variant(X_train_20, y_train_20, selected_features_powerset_20, label_cols)
X_train_40, y_train_40 = run_mlsmote_for_variant(X_train_40, y_train_40, selected_features_powerset_40, label_cols)
X_train_60, y_train_60 = run_mlsmote_for_variant(X_train_60, y_train_60, selected_features_powerset_60, label_cols)
X_train_80, y_train_80 = run_mlsmote_for_variant(X_train_80, y_train_80, selected_features_powerset_80, label_cols)
X_train_100, y_train_100 = run_mlsmote_for_variant(X_train_100, y_train_100, selected_features_powerset_100, label_cols)

# Binary Relevan + Random Forest (20% Fitur)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, hamming_loss, jaccard_score, classification_report
)
import time
import numpy as np

rf_base_br_20 = RandomForestClassifier(class_weight='balanced', random_state=42)
br_base_20 = BinaryRelevance(rf_base_br_20)

param_dist_br_20 = {
    'classifier__n_estimators': [50, 100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2']
}

# 3. Buat RandomizedSearchCV
random_search_br_20 = RandomizedSearchCV(
    estimator=br_base_20,
    param_distributions=param_dist_br_20,
    n_iter=10,
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

random_search_br_20.fit(X_train_20, y_train_20)

best_br_model_20 = random_search_br_20.best_estimator_

training_time_br_20 = time.time() - start_time_br_20

Y_pred_br_20 = best_br_model_20.predict(X_test_20).toarray()

y_test_array_20 = y_test_20.toarray() if hasattr(y_test_20, 'toarray') else np.array(y_test_20)

accuracy_br_20 = accuracy_score(y_test_array_20, Y_pred_br_20)
precision_weighted_br_20 = precision_score(y_test_array_20, Y_pred_br_20, average='macro', zero_division=0)
recall_weighted_br_20 = recall_score(y_test_array_20, Y_pred_br_20, average='macro', zero_division=0)
f1_weighted_br_20 = f1_score(y_test_array_20, Y_pred_br_20, average='macro', zero_division=0)
hamming_loss_br_20 = hamming_loss(y_test_array_20, Y_pred_br_20)

print("\n=== Model Evaluation (Binary Relevance with MLSMOTE and 20% features) ===")
print(f"Subset Accuracy: {accuracy_br_20:.4f}")
print(f"Weighted Precision: {precision_weighted_br_20:.4f}")
print(f"Weighted Recall: {recall_weighted_br_20:.4f}")
print(f"Weighted F1-Score: {f1_weighted_br_20:.4f}")
print(f"Hamming Loss: {hamming_loss_br_20:.4f}")

n_labels_20 = y_test_array_20.shape[1]
label_names_20 = [f'Label_{i+1}' for i in range(n_labels_20)]
print("\nClassification Report per Label:")
print(classification_report(y_test_array_20, Y_pred_br_20, target_names=label_names_20, zero_division=0, digits=4))
