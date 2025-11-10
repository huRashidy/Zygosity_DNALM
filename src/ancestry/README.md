# src README

Important: placeholder data paths

The Python scripts inside the src/ directory contain placeholder file system paths for data and results. Before running any script, replace these placeholders with the actual paths on your machine or cluster.

Common placeholder examples used across scripts:
- ./data/chr6/...
- ./data/results/
- ./data/figures/

Recommended steps:
1. Replace all occurrences of absolute or placeholder paths in scripts with your real paths (either absolute paths or project-relative paths).
2. Verify file names and extensions (for example, .pkl, .npy) match your local data files.

# Source Code Organization

This directory contains scripts supporting the main experiments and figures.

## Folders

- **train_head**  
  Stratified 5-fold cross-validation code to train the XGBoost classification head using either full MHC region embeddings (all 11 windows), the top 4 performing windows only, or each window seperately for max aggregation embedding. 

- **embeddings**  
  Code for generating embeddings using HyenaDNA for the 11 MHC windows (450 kb) of 2,548 individuals from the 1000 Genomes Project. Haplotypes are assigned by applying mutations from VCF files using the Genome kit library.

- **Fig_gen**  
  Scripts for figure generation. Each figure from the paper can be reproduced using CSV outputs from the `./results` folder.

- In addition to windows_to_genes.py, that we used to look for the genes existing in the top performing windows.
