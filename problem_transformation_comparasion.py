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
    var_name = f"selected_features_powerset_{p}"
    globals()[var_name] = mi_df['Feature'].tolist()[:num_features]
    print(f"\n--- Top {p}% Fitur Terpilih ({num_features} fitur) ---")
    print(globals()[var_name][:10]) 

# Split data menjadi training dan testing set
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

# Binary Relevance + Random Forest Classifier (20% Features)
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, hamming_loss, jaccard_score, classification_report
)
import time
import numpy as np

start_time_br_20 = time.time()

rf_base_br_20 = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42
)
br_base_20 = BinaryRelevance(rf_base_br_20)

br_base_20.fit(X_train_20, y_train_20)

training_time_br_20 = time.time() - start_time_br_20

Y_pred_br_20 = br_base_20.predict(X_test_20).toarray()

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
print(f"Training Time: {training_time_br_20:.2f} seconds")

# Classifier Chain + Random Forest Classifier (20% Features)
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.problem_transform import ClassifierChain
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, hamming_loss, jaccard_score, classification_report
)
import time
import numpy as np

start_time_cc_20 = time.time()

rf_base_cc_20 = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42
)
cc_base_20 = ClassifierChain(rf_base_cc_20)

cc_base_20.fit(X_train_20, y_train_20)

training_time_cc_20 = time.time() - start_time_cc_20

Y_pred_cc_20 = cc_base_20.predict(X_test_20).toarray()

y_test_array_20 = y_test_20.toarray() if hasattr(y_test_20, 'toarray') else np.array(y_test_20)

# 6. Evaluasi metrik
accuracy_cc_20 = accuracy_score(y_test_array_20, Y_pred_cc_20)
precision_weighted_cc_20 = precision_score(y_test_array_20, Y_pred_cc_20, average='macro', zero_division=0)
recall_weighted_cc_20 = recall_score(y_test_array_20, Y_pred_cc_20, average='macro', zero_division=0)
f1_weighted_cc_20 = f1_score(y_test_array_20, Y_pred_cc_20, average='macro', zero_division=0)
hamming_loss_cc_20 = hamming_loss(y_test_array_20, Y_pred_cc_20)

print("\n=== Model Evaluation (Classifier Chain with MLSMOTE and 20% features) ===")
print(f"Subset Accuracy: {accuracy_cc_20:.4f}")
print(f"Weighted Precision: {precision_weighted_cc_20:.4f}")
print(f"Weighted Recall: {recall_weighted_cc_20:.4f}")
print(f"Weighted F1-Score: {f1_weighted_cc_20:.4f}")
print(f"Hamming Loss: {hamming_loss_cc_20:.4f}")
print(f"Training Time: {training_time_cc_20:.2f} seconds")

## Label Powerset + Random Forest Classifier (20% Features)
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.problem_transform import LabelPowerset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, hamming_loss, jaccard_score, classification_report
)
import time
import numpy as np

start_time_lp_20 = time.time()

rf_base_lp_20 = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42
)
lp_base_20 = LabelPowerset(rf_base_lp_20, require_dense=[False, True])

lp_base_20.fit(X_train_20, y_train_20)

training_time_lp_20 = time.time() - start_time_lp_20

Y_pred_lp_20 = lp_base_20.predict(X_test_20).toarray()

y_test_array_20 = y_test_20.toarray() if hasattr(y_test_20, 'toarray') else np.array(y_test_20)

# 6. Evaluasi metrik
accuracy_lp_20 = accuracy_score(y_test_array_20, Y_pred_lp_20)
precision_weighted_lp_20 = precision_score(y_test_array_20, Y_pred_lp_20, average='macro', zero_division=0)
recall_weighted_lp_20 = recall_score(y_test_array_20, Y_pred_lp_20, average='macro', zero_division=0)
f1_weighted_lp_20 = f1_score(y_test_array_20, Y_pred_lp_20, average='macro', zero_division=0)
hamming_loss_lp_20 = hamming_loss(y_test_array_20, Y_pred_lp_20)

# 7. Print hasil
print("\n=== Model Evaluation (Label Powerset with MLSMOTE and 20% features) ===")
print(f"Subset Accuracy: {accuracy_lp_20:.4f}")
print(f"Weighted Precision: {precision_weighted_lp_20:.4f}")
print(f"Weighted Recall: {recall_weighted_lp_20:.4f}")
print(f"Weighted F1-Score: {f1_weighted_lp_20:.4f}")
print(f"Hamming Loss: {hamming_loss_lp_20:.4f}")
print(f"Training Time: {training_time_lp_20:.2f} seconds")

# Binary Relevance + Random Forest Classifier (40% Features)
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, hamming_loss, jaccard_score, classification_report
)
import time
import numpy as np

start_time_br_40 = time.time()

rf_base_br_40 = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42
)
br_base_40 = BinaryRelevance(rf_base_br_40)

br_base_40.fit(X_train_40, y_train_40)

training_time_br_40 = time.time() - start_time_br_40

Y_pred_br_40 = br_base_40.predict(X_test_40).toarray()

y_test_array_40 = y_test_40.toarray() if hasattr(y_test_40, 'toarray') else np.array(y_test_40)

print(f"Subset Accuracy: {accuracy_score(y_test_array_40, Y_pred_br_40):.4f}")
print(f"Precision: {precision_score(y_test_array_40, Y_pred_br_40, average='macro', zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test_array_40, Y_pred_br_40, average='macro', zero_division=0):.4f}")
print(f"F1-Score: {f1_score(y_test_array_40, Y_pred_br_40, average='macro', zero_division=0):.4f}")
print(f"Hamming Loss: {hamming_loss(y_test_array_40, Y_pred_br_40):.4f}")
print(f"Training Time: {training_time_br_40:.2f} seconds")

# Classifier Chain + Random Forest Classifier (40% Features)
from skmultilearn.problem_transform import ClassifierChain
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, hamming_loss, jaccard_score, classification_report
)
import time
import numpy as np

start_time_cc_40 = time.time()

rf_base_cc_40 = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42
)

cc_base_40 = ClassifierChain(rf_base_cc_40)

cc_base_40.fit(X_train_40, y_train_40)

training_time_cc_40 = time.time() - start_time_cc_40

Y_pred_cc_40 = cc_base_40.predict(X_test_40).toarray()

y_test_array_40 = y_test_40.toarray() if hasattr(y_test_40, 'toarray') else np.array(y_test_40)

print(f"Subset Accuracy: {accuracy_score(y_test_array_40, Y_pred_cc_40):.4f}")
print(f"Precision: {precision_score(y_test_array_40, Y_pred_cc_40, average='macro', zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test_array_40, Y_pred_cc_40, average='macro', zero_division=0):.4f}")
print(f"F1-Score: {f1_score(y_test_array_40, Y_pred_cc_40, average='macro', zero_division=0):.4f}")
print(f"Hamming Loss: {hamming_loss(y_test_array_40, Y_pred_cc_40):.4f}")
print(f"Training Time: {training_time_cc_40:.2f} seconds")

# Label Powerset + Random Forest Classifier (40% Features)
from skmultilearn.problem_transform import LabelPowerset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, hamming_loss, jaccard_score, classification_report
)
import time
import numpy as np

start_time_lp_40 = time.time()

rf_base_lp_40 = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42
)

lp_base_40 = LabelPowerset(rf_base_lp_40, require_dense=[False, True])

lp_base_40.fit(X_train_40, y_train_40)

training_time_lp_40 = time.time() - start_time_lp_40

Y_pred_lp_40 = lp_base_40.predict(X_test_40).toarray()

y_test_array_40 = y_test_40.toarray() if hasattr(y_test_40, 'toarray') else np.array(y_test_40)

print(f"Subset Accuracy: {accuracy_score(y_test_array_40, Y_pred_lp_40):.4f}")
print(f"Precision: {precision_score(y_test_array_40, Y_pred_lp_40, average='macro', zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test_array_40, Y_pred_lp_40, average='macro', zero_division=0):.4f}")
print(f"F1-Score: {f1_score(y_test_array_40, Y_pred_lp_40, average='macro', zero_division=0):.4f}")
print(f"Hamming Loss: {hamming_loss(y_test_array_40, Y_pred_lp_40):.4f}")
print(f"Training Time: {training_time_lp_40:.2f} seconds")

# Binary Relevance + Random Forest Classifier (60% Features)
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, hamming_loss, jaccard_score, classification_report
)
import time
import numpy as np

start_time_br_60 = time.time()

rf_base_br_60 = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42
)

br_base_60 = BinaryRelevance(rf_base_br_60)

br_base_60.fit(X_train_60, y_train_60)

training_time_br_60 = time.time() - start_time_br_60

Y_pred_br_60 = br_base_60.predict(X_test_60).toarray()

y_test_array_60 = y_test_60.toarray() if hasattr(y_test_60, 'toarray') else np.array(y_test_60)

print(f"Subset Accuracy: {accuracy_score(y_test_array_60, Y_pred_br_60):.4f}")
print(f"Precision: {precision_score(y_test_array_60, Y_pred_br_60, average='macro', zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test_array_60, Y_pred_br_60, average='macro', zero_division=0):.4f}")
print(f"F1-Score: {f1_score(y_test_array_60, Y_pred_br_60, average='macro', zero_division=0):.4f}")
print(f"Hamming Loss: {hamming_loss(y_test_array_60, Y_pred_br_60):.4f}")
print(f"Training Time: {training_time_br_60:.2f} seconds")

# Classifier Chains + Random Forest Classifier (60% Features)
from skmultilearn.problem_transform import ClassifierChain
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, hamming_loss, jaccard_score, classification_report
)
import time
import numpy as np

start_time_cc_60 = time.time()

rf_base_cc_60 = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42
)

cc_base_60 = ClassifierChain(rf_base_cc_60)

cc_base_60.fit(X_train_60, y_train_60)

training_time_cc_60 = time.time() - start_time_cc_60

Y_pred_cc_60 = cc_base_60.predict(X_test_60).toarray()

y_test_array_60 = y_test_60.toarray() if hasattr(y_test_60, 'toarray') else np.array(y_test_60)

print(f"Subset Accuracy: {accuracy_score(y_test_array_60, Y_pred_cc_60):.4f}")
print(f"Precision: {precision_score(y_test_array_60, Y_pred_cc_60, average='macro', zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test_array_60, Y_pred_cc_60, average='macro', zero_division=0):.4f}")
print(f"F1-Score: {f1_score(y_test_array_60, Y_pred_cc_60, average='macro', zero_division=0):.4f}")
print(f"Hamming Loss: {hamming_loss(y_test_array_60, Y_pred_cc_60):.4f}")
print(f"Training Time: {training_time_cc_60:.2f} seconds")

# Label Powerset + Random Forest Classifier (60% Features)
from skmultilearn.problem_transform import LabelPowerset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, hamming_loss, jaccard_score, classification_report
)
import time
import numpy as np

start_time_lp_60 = time.time()

rf_base_lp_60 = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42
)

lp_base_60 = LabelPowerset(rf_base_lp_60, require_dense=[False, True])

lp_base_60.fit(X_train_60, y_train_60)

training_time_lp_60 = time.time() - start_time_lp_60

Y_pred_lp_60 = lp_base_60.predict(X_test_60).toarray()

y_test_array_60 = y_test_60.toarray() if hasattr(y_test_60, 'toarray') else np.array(y_test_60)

print(f"Subset Accuracy: {accuracy_score(y_test_array_60, Y_pred_lp_60):.4f}")
print(f"Precision: {precision_score(y_test_array_60, Y_pred_lp_60, average='macro', zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test_array_60, Y_pred_lp_60, average='macro', zero_division=0):.4f}")
print(f"F1-Score: {f1_score(y_test_array_60, Y_pred_lp_60, average='macro', zero_division=0):.4f}")
print(f"Hamming Loss: {hamming_loss(y_test_array_60, Y_pred_lp_60):.4f}")
print(f"Training Time: {training_time_lp_60:.2f} seconds")

# Binary Relevance + Random Forest Classifier (80% Features)
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, hamming_loss, jaccard_score, classification_report
)
import time
import numpy as np

start_time_br_80 = time.time()

rf_base_br_80 = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42
)

br_base_80 = BinaryRelevance(rf_base_br_80)

br_base_80.fit(X_train_80, y_train_80)

training_time_br_80 = time.time() - start_time_br_80

Y_pred_br_80 = br_base_80.predict(X_test_80).toarray()

y_test_array_80 = y_test_80.toarray() if hasattr(y_test_80, 'toarray') else np.array(y_test_80)

print(f"Subset Accuracy: {accuracy_score(y_test_array_80, Y_pred_br_80):.4f}")
print(f"Precision: {precision_score(y_test_array_80, Y_pred_br_80, average='macro', zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test_array_80, Y_pred_br_80, average='macro', zero_division=0):.4f}")
print(f"F1-Score: {f1_score(y_test_array_80, Y_pred_br_80, average='macro', zero_division=0):.4f}")
print(f"Hamming Loss: {hamming_loss(y_test_array_80, Y_pred_br_80):.4f}")
print(f"Training Time: {training_time_br_80:.2f} seconds")

# Classifier Chain + Random Forest Classifier (80% Features)
from skmultilearn.problem_transform import ClassifierChain
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, hamming_loss, jaccard_score, classification_report
)
import time
import numpy as np

start_time_cc_80 = time.time()

rf_base_cc_80 = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42
)

cc_base_80 = ClassifierChain(rf_base_cc_80)

cc_base_80.fit(X_train_80, y_train_80)

training_time_cc_80 = time.time() - start_time_cc_80

Y_pred_cc_80 = cc_base_80.predict(X_test_80).toarray()

y_test_array_80 = y_test_80.toarray() if hasattr(y_test_80, 'toarray') else np.array(y_test_80)

print(f"Subset Accuracy: {accuracy_score(y_test_array_80, Y_pred_cc_80):.4f}")
print(f"Precision: {precision_score(y_test_array_80, Y_pred_cc_80, average='macro', zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test_array_80, Y_pred_cc_80, average='macro', zero_division=0):.4f}")
print(f"F1-Score: {f1_score(y_test_array_80, Y_pred_cc_80, average='macro', zero_division=0):.4f}")
print(f"Hamming Loss: {hamming_loss(y_test_array_80, Y_pred_cc_80):.4f}")
print(f"Training Time: {training_time_cc_80:.2f} seconds")

# Label Powerset + Random Forest Classifier (80% Features)
from skmultilearn.problem_transform import LabelPowerset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, hamming_loss, jaccard_score, classification_report
)
import time
import numpy as np

start_time_lp_80 = time.time()

rf_base_lp_80 = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42
)

lp_base_80 = LabelPowerset(classifier=rf_base_lp_80, require_dense=[False, True])

lp_base_80.fit(X_train_80, y_train_80)

training_time_lp_80 = time.time() - start_time_lp_80

Y_pred_lp_80 = lp_base_80.predict(X_test_80).toarray()

y_test_array_80 = y_test_80.toarray() if hasattr(y_test_80, 'toarray') else np.array(y_test_80)

print(f"Subset Accuracy: {accuracy_score(y_test_array_80, Y_pred_lp_80):.4f}")
print(f"Precision: {precision_score(y_test_array_80, Y_pred_lp_80, average='macro', zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test_array_80, Y_pred_lp_80, average='macro', zero_division=0):.4f}")
print(f"F1-Score: {f1_score(y_test_array_80, Y_pred_lp_80, average='macro', zero_division=0):.4f}")
print(f"Hamming Loss: {hamming_loss(y_test_array_80, Y_pred_lp_80):.4f}")
print(f"Training Time: {training_time_lp_80:.2f} seconds")

# Binary Relevance + Random Forest Classifier (100% Features)
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, hamming_loss, jaccard_score, classification_report
)
import time
import numpy as np

start_time_br_100 = time.time()

rf_base_br_100 = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42
)

br_base_100 = BinaryRelevance(rf_base_br_100)

br_base_100.fit(X_train_100, y_train_100)

training_time_br_100 = time.time() - start_time_br_100

Y_pred_br_100 = br_base_100.predict(X_test_100).toarray()

y_test_array_100 = y_test_100.toarray() if hasattr(y_test_100, 'toarray') else np.array(y_test_100)

print(f"Precision: {precision_score(y_test_array_100, Y_pred_br_100, average='macro', zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test_array_100, Y_pred_br_100, average='macro', zero_division=0):.4f}")
print(f"F1-Score: {f1_score(y_test_array_100, Y_pred_br_100, average='macro', zero_division=0):.4f}")
print(f"Hamming Loss: {hamming_loss(y_test_array_100, Y_pred_br_100):.4f}")
print(f"Training Time: {training_time_br_100:.2f} seconds")

# Classifier Chain + Random Forest Classifier (100% Features)
from skmultilearn.problem_transform import ClassifierChain
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, hamming_loss, jaccard_score, classification_report
)
import time
import numpy as np

start_time_cc_100 = time.time()

rf_base_cc_100 = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42
)

cc_base_100 = ClassifierChain(rf_base_cc_100)

cc_base_100.fit(X_train_100, y_train_100)

training_time_cc_100 = time.time() - start_time_cc_100

Y_pred_cc_100 = cc_base_100.predict(X_test_100).toarray()

y_test_array_100 = y_test_100.toarray() if hasattr(y_test_100, 'toarray') else np.array(y_test_100)

print(f"Subset Accuracy: {accuracy_score(y_test_array_100, Y_pred_cc_100):.4f}")
print(f"Precision: {precision_score(y_test_array_100, Y_pred_cc_100, average='macro', zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test_array_100, Y_pred_cc_100, average='macro', zero_division=0):.4f}")
print(f"F1-Score: {f1_score(y_test_array_100, Y_pred_cc_100, average='macro', zero_division=0):.4f}")
print(f"Hamming Loss: {hamming_loss(y_test_array_100, Y_pred_cc_100):.4f}")
print(f"Training Time: {training_time_cc_100:.2f} seconds")


# Label Powerset + Random Forest Classifier (100% Features)
from skmultilearn.problem_transform import LabelPowerset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, hamming_loss, jaccard_score, classification_report
)
import time
import numpy as np

start_time_lp_100 = time.time()

rf_base_lp_100 = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42
)

lp_base_100 = LabelPowerset(rf_base_lp_100, require_dense=[False, True])

lp_base_100.fit(X_train_100, y_train_100)

training_time_lp_100 = time.time() - start_time_lp_100

Y_pred_lp_100 = lp_base_100.predict(X_test_100).toarray()

y_test_array_100 = y_test_100.toarray() if hasattr(y_test_100, 'toarray') else np.array(y_test_100)

print(f"Subset Accuracy: {accuracy_score(y_test_array_100, Y_pred_lp_100):.4f}")
print(f"Precision: {precision_score(y_test_array_100, Y_pred_lp_100, average='macro', zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test_array_100, Y_pred_lp_100, average='macro', zero_division=0):.4f}")
print(f"F1-Score: {f1_score(y_test_array_100, Y_pred_lp_100, average='macro', zero_division=0):.4f}")
print(f"Hamming Loss: {hamming_loss(y_test_array_100, Y_pred_lp_100):.4f}")
print(f"Training Time: {training_time_lp_100:.2f} seconds")