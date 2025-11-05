import numpy as np
import pandas as pd
import pickle
import gc
import argparse
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
import sys
def get_model_by_name(name, labels=None):
    if name == "WLR":
        class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
        )

        class_weight_dict = {int(labels[i]): weight for i, weight in enumerate(class_weights)}

        return LogisticRegression(
                max_iter=1000,
                solver='lbfgs',
                multi_class='multinomial',
                class_weight=class_weight_dict,
                verbose=1,
                n_jobs=-1,
                random_state=42
            )

    elif name == "SVC":
        return SVC(kernel='linear', C=1.0, probability=True, random_state=42)
    elif name == "XGB":
        n_classes = len(np.unique(labels))
            # compute balanced class weights
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(labels),
            y=labels
        )
        weight_dict = dict(zip(np.unique(labels), class_weights))

        # map weight to each label
        sample_weights = np.array([weight_dict[label] for label in labels])
        model = xgb.XGBClassifier(
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
        return model , sample_weights
    else:
        raise ValueError(f"Unknown model name: {name}")

def train_and_evaluate_model_cv(X, y, method, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    f1_scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        if method == "XGB":
            model, sample_weights = get_model_by_name(method, labels=y_train)
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model = get_model_by_name(method, labels=y_train)
            model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        f1_scores.append(f1_score(y_val, y_pred, average="macro"))
    return f1_scores

def predict_by_window(embeddings_all, y, method="WLR"):
    n_samples, n_windows, embedding_dim = embeddings_all.shape
    f1_scores_all = []
    for w in range(n_windows):
        print(f"Processing window {w+1}/{n_windows}")
        sys.stdout.flush()

        X_window = embeddings_all[:, w, :]
        f1_window = train_and_evaluate_model_cv(X_window, y, method)
        f1_scores_all.append(f1_window)
    return np.array(f1_scores_all)

def main_refined_windows(chromosome, model_name):
    if chromosome == 21:
        with open(f"/data/horse/ws/huel099f-ancestry_task/data/chr21/chr21_labels_All.pkl", "rb") as f:
            labels_all = pickle.load(f)
    elif chromosome == 6:
        with open("/data/horse/ws/huel099f-ancestry_task/data/chr6/chr6_labels_HLA_252.pkl", "rb") as f:
            labels_252 = pickle.load(f)
        with open("/data/horse/ws/huel099f-ancestry_task/data/chr6/chr6_labels_HLA_2296.pkl", "rb") as f:
            labels_2296 = pickle.load(f)
        labels_all = np.concatenate((labels_252, labels_2296), axis=0)
    y = np.squeeze(labels_all, axis=1) if labels_all.ndim > 1 else labels_all

    emb_types = ["max"]
    for emb in emb_types:
        print(f"Loading {emb} embeddings for chromosome {chromosome}")
        if chromosome == 21:
            print("Processing chromosome 21")
            with open(f"/data/horse/ws/huel099f-ancestry_task/data/chr21/HyenaDNA450k_{emb}_chr21_concat/chr21_{emb}_embeddings.pkl", "rb") as f:
                ind_all = pickle.load(f)
            sys.stdout.flush()


        elif chromosome == 6:
            print("Processing chromosome 6")
            sys.stdout.flush()

            with open(f"/data/horse/ws/huel099f-ancestry_task/data/chr6/HyenaDNA450k_{emb}_252/chr6_{emb}_embeddings.pkl", "rb") as f:
                all_252 = pickle.load(f)

            with open(f"/data/horse/ws/huel099f-ancestry_task/data/chr6/HyenaDNA450k_{emb}_2296_concat/chr6_{emb}_embeddings.pkl", "rb") as f:
                all_2296 = pickle.load(f)
            ind_all = np.concatenate((all_252, all_2296), axis=0)



        print(f"Embeddings shape: {ind_all.shape}")
        sys.stdout.flush()

        f1_scores = predict_by_window(ind_all, y, method=model_name)
        np.save(f"/data/horse/ws/huel099f-ancestry_task/results/f1_scores_{model_name}_{emb}_chr{chromosome}_windows.npy", f1_scores)
        del ind_all, f1_scores
        gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run per-window training with low memory usage")
    parser.add_argument('--chr', type=int, required=True, choices=[6, 21], help='Chromosome number')
    parser.add_argument('--model', type=str, required=True, choices=['SVC', 'WLR', 'XGB'], help='Model type')
    args = parser.parse_args()
    main_refined_windows(args.chr, args.model)
