import streamlit as st
import pickle
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Fragments
import io

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

def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

def predict_from_smiles(model, smiles_list, label_cols, selected_features):
    results = []
    feature_rows = []
    valid_smiles = []

    for smi in smiles_list:
        features = extract_features(smi)
        if features is None:
            results.append({"SMILES": smi, "Predicted Labels": "Invalid SMILES"})
        else:
            feature_rows.append(features)
            valid_smiles.append(smi)

    if feature_rows:
        df_features = pd.DataFrame(feature_rows)

        fp_columns = df_features['MorganFP'].apply(pd.Series)
        fp_columns = fp_columns.rename(columns=lambda x: f'morgan_{x}')
        df_features = df_features.drop(columns=['MorganFP'])
        df_features = pd.concat([df_features, fp_columns], axis=1)

        if isinstance(selected_features, dict):
            selected_features = list(selected_features.keys())

        X_input = df_features[selected_features].to_numpy(dtype=float)

        preds = model.predict(X_input)
        if hasattr(preds, "toarray"):
            preds = preds.toarray()

        for smi, pred in zip(valid_smiles, preds):
            pred_labels = [label_cols[i] for i, val in enumerate(pred) if val == 1]
            results.append({
                "SMILES": smi,
                "Predicted Labels": ", ".join(pred_labels) if pred_labels else "No Danger"
            })

    return pd.DataFrame(results)

with open("selected_features.pkl", "rb") as f:
    selected_features_dict = pickle.load(f)
selected_features = selected_features_dict["selected_features_powerset_20"]

with open("best_classifier_chain_model_20.pkl", "rb") as f:
    loaded_model = pickle.load(f)

label_cols = [
    'No Danger', 'Corrosive', 'Irritant', 'Acute Toxic', 
    'Health Hazard', 'Environmental Hazard', 'Flammable', 
    'Compressed Gas', 'Explosive', 'Oxidizer'
]

st.title("Prediksi Bahaya Senyawa Berdasarkan Notasi SMILES")

input_method = st.radio("Pilih metode input:", ["File Upload", "Manual Input"])

if input_method == "File Upload":
    uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx"])
else:
    smiles_input = st.text_area("Masukkan SMILES (pisahkan dengan baris baru)")

if st.button("Predict"):
    if input_method == "File Upload":
        if uploaded_file is not None:
            df_input = pd.read_excel(uploaded_file)

            if "SMILES" not in df_input.columns:
                st.error("File Excel harus memiliki kolom 'SMILES'.")
            else:
                smiles_list = df_input['SMILES'].dropna().astype(str).tolist()
                results = predict_from_smiles(
                    model=loaded_model,
                    smiles_list=smiles_list,
                    label_cols=label_cols,
                    selected_features=selected_features
                )

                df_output = df_input.copy()
                pred_map = dict(zip(results['SMILES'], results['Predicted Labels']))
                insert_index = df_output.columns.get_loc("SMILES") + 1
                df_output.insert(insert_index, "Predicted Labels",
                                 df_output['SMILES'].map(pred_map).fillna("Invalid SMILES"))

                st.dataframe(df_output)
                st.download_button(
                    label="Download Results",
                    data=to_excel(df_output),
                    file_name="Prediksi_Bahaya_Senyawa.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.error("Silakan upload file Excel terlebih dahulu.")

    else:
        smiles_list = [s.strip() for s in smiles_input.split("\n") if s.strip()]
        if smiles_list:
            results = predict_from_smiles(
                model=loaded_model,
                smiles_list=smiles_list,
                label_cols=label_cols,
                selected_features=selected_features
            )
            st.dataframe(results)
            st.download_button(
                label="Download Results",
                data=to_excel(results),
                file_name="Prediksi_Bahaya_Senyawa.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.error("Masukkan minimal satu SMILES.")
