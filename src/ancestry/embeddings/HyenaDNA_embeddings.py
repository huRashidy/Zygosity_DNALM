import sys 
print("python is here ")
sys.stdout.flush()

# Replace these with your local paths or package locations
sys.path.append("./libs/hyena-dna")
sys.path.append("./libs/Orthrus")

import numpy as np
from itertools import islice

import genome_kit as gk
from genome_kit import Genome
from genome_kit import Interval
import pandas as pd
"""
import transformers
from transformers import PreTrainedModel, AutoModelForCausalLM, PretrainedConfig
from transformers import AutoModelForCausalLM
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer, logging
"""
from orthrus.newpkldata import SlidingWindowVariantDataset
import torch
import torch.nn.functional as F
import json
"""
import os
import subprocess
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
"""

import pandas as pd
import pickle
sys.stdout.flush()

print("libraries are imported")
sys.stdout.flush()

def batch_oh_to_seq(tensor_list):  # List[Tensor]
    base_map = ['A', 'C', 'G', 'T']
    sequences = []

    for tensor in tensor_list:
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.squeeze(0).cpu().numpy()  # ensures shape [L, 4]
        assert tensor.ndim == 2 and tensor.shape[1] == 4, f"Expected [L, 4], got {tensor.shape}"
        indices = np.argmax(tensor, axis=1)  # [L]
        seq = ''.join([base_map[int(i)] for i in indices])  # cast to int just in case
        sequences.append(seq)

    return sequences

ref = Genome("hg38")

# Replace with your sequence file path
human_seq = pd.read_csv("./data/human-sequences.tsv" , names = ["CHROM" , "START" , "END", "set"] , sep="\t")
human_seq_chr21 = human_seq[human_seq["CHROM"] == "chr6"]


start = 28510120
end = 33480577

# Replace with your vcf/gz or positions file path
gzip_path = "./data/chr6_MHC.tsv"
#gzip_path = "./data/chr21_positions.tsv"
from orthrus.newpkldata import SlidingWindowVariantDataset

data = SlidingWindowVariantDataset(
        ref_genome = ref,
        # Replace with a local pkl directory for cached positions/windows
        pkl_dir = "./data/chr6pkl_positions",
        vcf_gz_path = gzip_path, 
        region_start = start,
        region_end = end,
        window_size = 450000
)

print("dataset is created")
sys.stdout.flush()

from torch.utils.data import DataLoader
from torch.utils.data import Subset

# Assume dataset is already initialized
total_windows = list(range(len(data)))
train_indices = total_windows

# Use Subset to wrap the original dataset
train_dataset = Subset(data, train_indices)
train_loader = DataLoader(train_dataset, batch_size=1, pin_memory= False )

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = 'LongSafari/hyenadna-medium-450k-seqlen-hf'
"""
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("LongSafari/hyenadna-medium-450k-seqlen-hf", trust_remote_code=True)
model.to(device)

print("Before DataLoader loop")
if torch.cuda.is_available():
    print(f"Allocated GPU: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Max reserved GPU: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")
sys.stdout.flush()
"""
for idx, batch in enumerate((train_loader)):

    m, p, m_lengths, p_lengths, labels = batch
    mat_seq = m
    pat_seq = p
    print("converted to seq")
sys.stdout.flush()
    #convert labels to numpy and save as pkl file
    labels = np.array(labels)
    with open(f"./data/chr21_labels.pkl", "wb") as f:
        pickle.dump(labels, f)
    break
    #for each sample in the batch, get the sequence pass to the model and get the embedding
    for i in range(len(mat_seq)):
        tok_mat = tokenizer(mat_seq[i], return_tensors='pt' , padding=False, truncation=True, max_length=450_000 )
        tok_pat = tokenizer(pat_seq[i], return_tensors='pt' , padding=False, truncation=True, max_length=450_000 )
        
        tok_mat = {k: v.to(device) for k, v in tok_mat.items()}
        tok_pat = {k: v.to(device) for k, v in tok_pat.items()}
        # Inference
        model.eval()
        with torch.inference_mode():
            out_m = model(**tok_mat, output_hidden_states=True, return_dict=True)
            out_p = model(**tok_pat, output_hidden_states=True, return_dict=True)

            final_hidden_m = out_m.hidden_states[-1]  # [B, L, H]
            final_hidden_p = out_p.hidden_states[-1] # [B, L, H]

            # Get max embedding: max over sequence length (dim=1)
            max_emb_m = final_hidden_m.max(dim=1).values
            max_emb_p = final_hidden_p.max(dim=1).values
            # Get mean embedding: average over sequence length (dim=1)
            mean_emb_m = final_hidden_m.mean(dim=1)  # [B, H]
            mean_emb_p = final_hidden_p.mean(dim=1)

            # Get last embedding in the sequence: take last token
            last_emb_m = final_hidden_m[:, -1, :]  # [B, H]
            last_emb_p = final_hidden_p[:, -1, :]

            # Get the max embedding
            full_emb_max = torch.cat([max_emb_m, max_emb_p], dim=-1).squeeze(0).cpu().numpy()

            # Save the max embedding    
            with open(f"./data/HyenaDNA450k_max_chr21/chr21_individual_{i}_window_{idx}.pkl", "wb") as f:
                pickle.dump(full_emb_max, f)
            with open(f"./data/HyenaDNA450k_max_chr21/chr21_mat_individual_{i}_window_{idx}.pkl", "wb") as f:
                pickle.dump(max_emb_m.squeeze(0).cpu().numpy(), f)
            with open(f"./data/HyenaDNA450k_max_chr21/chr21_pat_individual_{i}_window_{idx}.pkl", "wb") as f:
                pickle.dump(max_emb_p.squeeze(0).cpu().numpy(), f)

            # Concatenate maternal and paternal
            full_emb_mean = torch.cat([mean_emb_m, mean_emb_p], dim=-1).squeeze(0).cpu().numpy()  # [2*H]
            #save the mean embedding
            with open(f"./data/HyenaDNA450k_mean_chr21/chr21_individual_{i}_window_{idx}.pkl", "wb") as f:
                pickle.dump(full_emb_mean, f)
            with open(f"./data/HyenaDNA450k_mean_chr21/chr21_mat_individual_{i}_window_{idx}.pkl", "wb") as f:
                pickle.dump(mean_emb_m.squeeze(0).cpu().numpy(), f)
            with open(f"./data/HyenaDNA450k_mean_chr21/chr21_pat_individual_{i}_window_{idx}.pkl", "wb") as f:
                pickle.dump(mean_emb_p.squeeze(0).cpu().numpy(), f)
            # Concatenate maternal and paternal
            full_emb_last= torch.cat([last_emb_m, last_emb_p], dim=-1).squeeze(0).cpu().numpy()  # [2*H]

            #save the last embedding
            with open(f"./data/HyenaDNA450k_last_chr21/chr21_individual_{i}_window_{idx}.pkl", "wb") as f:
                pickle.dump(full_emb_last, f)
            with open(f"./data/HyenaDNA450k_last_chr21/chr21_mat_individual_{i}_window_{idx}.pkl", "wb") as f:
                pickle.dump(last_emb_m.squeeze(0).cpu().numpy(), f)
            with open(f"./data/HyenaDNA450k_last_chr21/chr21_pat_individual_{i}_window_{idx}.pkl", "wb") as f:
                pickle.dump(last_emb_p.squeeze(0).cpu().numpy(), f)
            
            del tok_mat, tok_pat
            del out_m, out_p
            del final_hidden_m, final_hidden_p
            del max_emb_m, max_emb_p, mean_emb_m, mean_emb_p, last_emb_m, last_emb_p
            del full_emb_max, full_emb_mean, full_emb_last
            torch.cuda.empty_cache()
