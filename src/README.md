# src README

Important: placeholder data paths

The Python scripts inside the src/ directory contain placeholder file system paths for data and results. Before running any script, replace these placeholders with the actual paths on your machine or cluster.

Common placeholder examples used across scripts:
- ./data/chr6/...
- ./data/results/

Recommended steps:
1. Replace all occurrences of absolute or placeholder paths in scripts with your real paths (either absolute paths or project-relative paths).
2. Prefer using a single configurable variable (e.g., BASE_DATA_DIR or DATA_DIR) or argparse flags so paths are easy to update in one place.
3. Verify file names and extensions (for example, .pkl, .npy) match your local data files.