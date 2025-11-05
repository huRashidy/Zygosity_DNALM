import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import pearsonr
import pickle
import seaborn as sns
import xgboost as xgb
from sklearn.utils.class_weight import compute_class_weight
import os
import argparse
import gc
import sys

def map_labels_to_superpopulations(labels, label_to_superpop):
    """
    Maps a list of integer labels to super population codes.

    Args:
        labels (list): List of integer labels (0–4)
        label_to_superpop (dict): Mapping from label to super population code

    Returns:
        list: List of super population codes corresponding to the input labels
    """
    return [label_to_superpop[label] for label in labels if label in label_to_superpop]




def get_label_to_superpop_mapping(sample_ids, samples_df, pop_df):
    """
    Given the same sample list used for label assignment,
    return the mapping from label index (0–4) to super population code (e.g., EUR, AFR).
    """
    filtered_df = samples_df[samples_df["Individual ID"].isin(sample_ids)].copy()

    # Merge with population reference to get super populations
    merged_df = filtered_df.merge(
        pop_df[["Population Code", "Super Population"]],
        left_on="Population",
        right_on="Population Code",
        how="left"
    )

    # Encode Super Populations into integer labels
    unique_superpops = sorted(merged_df["Super Population"].dropna().unique())
    label_to_superpop = {idx: sp for idx, sp in enumerate(unique_superpops)}
    return label_to_superpop

def get_model_by_name(name, labels=None):
    if name == "WLR":
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels),
            y=labels
        )
        class_weight_dict = {int(i): w for i, w in zip(np.unique(labels), class_weights)}
        return LogisticRegression(
            max_iter=10000,
            solver='saga',
            multi_class='multinomial',
            class_weight='balanced',
            penalty='l2',
            n_jobs=-1,
            random_state=42
            )

    elif name == "SVC":
        return SVC(kernel='linear', C=1.0, probability=True, random_state=42)
    elif name == "XGB":
        n_classes = len(np.unique(labels))
        return xgb.XGBClassifier(
            objective="multi:softprob",      # get calibrated probs
            num_class=n_classes,             # len(np.unique(y))
            tree_method="hist",              # or "gpu_hist" if GPU
            eval_metric="mlogloss",          # or "merror"
            n_estimators=1000,               # with early stopping
            learning_rate=0.1,              # small, safer
            max_depth=3,                     # 4–8 usually sweet spot
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model name: {name}")



def cross_validate_model(X, y, model, n_splits=5):
    individual_vectors = X
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_fold_metrics = []

    try:
        onehot = OneHotEncoder(sparse_output=False)
    except TypeError:
        onehot = OneHotEncoder(sparse=False)

    for train_idx, val_idx in skf.split(individual_vectors, y):
        X_train, X_val = individual_vectors[train_idx], individual_vectors[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        y_onehot_train = onehot.fit_transform(y_train.reshape(-1, 1))
        y_onehot_val = onehot.transform(y_val.reshape(-1, 1))

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        y_prob = model.predict_proba(X_val) if hasattr(model, "predict_proba") else np.eye(len(np.unique(y)))[y_pred]

        fold_metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "f1": f1_score(y_val, y_pred, average="macro"),
            "mcc": matthews_corrcoef(y_val, y_pred),
            "auroc": roc_auc_score(y_onehot_val, y_prob, average="macro", multi_class='ovr'),
            "pearson": np.mean([pearsonr(y_onehot_val[:, i], y_prob[:, i])[0] for i in range(y_onehot_val.shape[1])]),
        }
        all_fold_metrics.append(fold_metrics)

    return all_fold_metrics

def save_results_to_csv(results_dict, model_name, chr_name, data_type,
                        results_dir="/data/horse/ws/huel099f-ancestry_task/results"
                        ):
    """
    Save aggregated CV results to a CSV in a consistent directory.
    - results_dir: base directory to write results (created if missing)
    - data_type: one of {"concat","mat","pat"}
    """
    os.makedirs(results_dir, exist_ok=True)

    records = []
    for emb_type, folds in results_dict.items():
        for fold_result in folds:
            rec = {"embedding_type": emb_type, "model": model_name, "chr": chr_name}
            rec.update(fold_result)
            records.append(rec)

    if not records:
        return None  # nothing to save

    fname = f"results_chr{chr_name}_{model_name}_{4}windows_XGBparams.csv"

    out_path = os.path.join(results_dir, fname)
    pd.DataFrame(records).to_csv(out_path, index=False)
    return out_path

def main_cv(X_dict, y, chr_name, method="WLR", filtering=False, var_thresh=0.0001, corr_thresh=0.9, n_splits=5):
    results = {}

    for name, embedding_tuple in X_dict.items():
        print(f"\nProcessing {name} embeddings")
        X, X_mat, X_pat = embedding_tuple

        for data_type, data in zip(["concat", "mat", "pat"], [X, X_mat, X_pat]):
            if data is not None and data.ndim == 2 and data.shape[1] > 0:
                model = get_model_by_name(method, labels=y)
                key = f"{name}_{data_type}" if data_type != "concat" else name
                results[key] = cross_validate_model(data, y, model, n_splits=n_splits)

    return results


def main_refined(chromosome, model_name):
    emb_types = ["last", "mean", "max"]
    all_results = {}

    # Load labels once
    if chromosome == 6:
        with open("/data/horse/ws/huel099f-ancestry_task/data/chr6/chr6_labels_HLA_252.pkl", "rb") as f:
            labels_252 = pickle.load(f)
        with open("/data/horse/ws/huel099f-ancestry_task/data/chr6/chr6_labels_HLA_2296.pkl", "rb") as f:
            labels_2296 = pickle.load(f)
        labels_all = np.concatenate((labels_252, labels_2296), axis=0)

    labels_all = np.squeeze(labels_all, axis=1) if labels_all.ndim > 1 else labels_all

    for emb in emb_types:
        X_dict = {}
        print(f"Processing embedding type: {emb}")
        sys.stdout.flush()

        if chromosome == 6:
            with open(f"/data/horse/ws/huel099f-ancestry_task/data/chr6/HyenaDNA450k_{emb}_252/chr6_{emb}_embeddings.pkl", "rb") as f:
                all_252 = pickle.load(f)
            with open(f"/data/horse/ws/huel099f-ancestry_task/data/chr6/HyenaDNA450k_{emb}_2296_concat/chr6_{emb}_embeddings.pkl", "rb") as f:
                all_2296 = pickle.load(f)
            with open(f"/data/horse/ws/huel099f-ancestry_task/data/chr6/HyenaDNA450k_{emb}_252/chr6_mat_{emb}_embeddings.pkl", "rb") as f:
                ind_mat_252 = pickle.load(f)
            with open(f"/data/horse/ws/huel099f-ancestry_task/data/chr6/HyenaDNA450k_{emb}_2296_concat/chr6_mat_{emb}_embeddings.pkl", "rb") as f:
                ind_mat_2296 = pickle.load(f)
            with open(f"/data/horse/ws/huel099f-ancestry_task/data/chr6/HyenaDNA450k_{emb}_252/chr6_pat_{emb}_embeddings.pkl", "rb") as f:
                ind_pat_252 = pickle.load(f)
            with open(f"/data/horse/ws/huel099f-ancestry_task/data/chr6/HyenaDNA450k_{emb}_2296_concat/chr6_pat_{emb}_embeddings.pkl", "rb") as f:
                ind_pat_2296 = pickle.load(f)

            ind_all = np.concatenate((all_252, all_2296), axis=0)
            ind_mat = np.concatenate((ind_mat_252, ind_mat_2296), axis=0)
            ind_pat = np.concatenate((ind_pat_252, ind_pat_2296), axis=0)

        #shape of ind_all (N,W,D), pick only windows with index 0, 3, 7, and 8
        vec_all = ind_all[:, [0, 3, 7, 8], :].reshape(-1, 4 * ind_all.shape[2])
        vec_mat = ind_mat[:, [0, 3, 7, 8], :].reshape(-1, 4 * ind_mat.shape[2])
        vec_pat = ind_pat[:, [0, 3, 7, 8], :].reshape(-1, 4 * ind_pat.shape[2])
        
        X_dict[emb] = (vec_all, vec_mat, vec_pat)
        del ind_all, ind_mat, ind_pat
        gc.collect()

        results_emb = main_cv(X_dict, labels_all, str(chromosome), method=model_name,
                        n_splits=5)
        all_results.update(results_emb)

        del X_dict, vec_all, vec_mat, vec_pat
        gc.collect()

    # Save all in one CSV
    save_results_to_csv(all_results, model_name, str(chromosome), "all_types")
    print(f"Results saved for chromosome {chromosome} with model {model_name}") 

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run refined main CV")
    parser.add_argument('--chr', type=int, required=True, choices=[6], help='Chromosome number')
    parser.add_argument('--model', type=str, required=True, choices=['SVC', 'WLR', 'XGB'], help='Model type')
    args = parser.parse_args()
    main_refined(args.chr, args.model)
