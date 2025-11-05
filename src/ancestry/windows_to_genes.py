#!/usr/bin/env python3
"""
Map genes (from a GTF file) to user-defined windows (BED-like).
- Works with .gtf or .gtf.gz
- No external genome library required.
- Handles chromosome naming differences automatically.
- Modes: overlap (default) or contained.

Usage:
  python windows_to_genes.py       --gtf /path/to/Homo_sapiens.GRCh38.112.gtf.gz       --windows-file /path/to/windows.csv       --output /path/to/genes_by_window.csv       --mode overlap

windows.csv format (header required):
  window,chromosome,start,end
  W1,6,28510120,28960120

Author: ChatGPT
"""
import argparse
import gzip
import os
import re
import csv

def parse_args():
    p = argparse.ArgumentParser(description="Assign genes from GTF to windows.")
    p.add_argument("--gtf", required=True, help="Path to Ensembl GTF (.gtf or .gtf.gz)")
    p.add_argument("--gtf", required=True, help="Path to GTF file (.gtf or .gtf.gz)")
    p.add_argument("--windows-file", required=True, help="CSV with columns: window,chromosome,start,end")
    p.add_argument("--output", required=True, help="Output CSV path")
    p.add_argument("--mode", choices=["overlap", "contained"], default="overlap",
                   help="overlap: any overlap counts; contained: gene fully inside window")
    return p.parse_args()
def open_maybe_gzip(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")

def norm_chromosome(c):
    c = str(c).strip()
    if c.lower().startswith("chr"):
        c = c[3:]
    return c
def parse_gtf_attributes(attr_str):
    out = {}
    for key in ("gene_id", "gene_name", "gene_biotype", "gene_type"):
        m = re.search(rf'{key}\s+"([^"]+)"', attr_str)
        if m:
            out[key] = m.group(1)
    if "gene_biotype" not in out and "gene_type" in out:
        out["gene_biotype"] = out["gene_type"]
    return out

def load_windows_csv(path):
    wins = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"window", "chromosome", "start", "end"}
        if set(reader.fieldnames) & required != required:
            raise ValueError(f"windows-file must have columns: {sorted(required)}; got {reader.fieldnames}")
        for row in reader:
            try:
                wins.append({
                    "window": row["window"],
                    "chromosome": norm_chromosome(row["chromosome"]),
                    "start": int(row["start"].replace(',', '')),
                    "end": int(row["end"].replace(',', '')),
                })
            except Exception as e:
                raise ValueError(f"Bad row in windows-file: {row} ({e})")
    return wins
def gene_matches_window(gene_start, gene_end, win_start, win_end, mode="overlap"):
    if mode == "overlap":
        return (gene_start <= win_end) and (gene_end >= win_start)
    else:
        return (gene_start >= win_start) and (gene_end <= win_end)

def iterate_gtf_genes(gtf_path):
    with open_maybe_gzip(gtf_path) as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chromosome, _, feature, start, end, _, strand, _, attrs = parts
            if feature != "gene":
                continue
            yield {
                "chromosome": norm_chromosome(chromosome),
                "start": int(start),
                "end": int(end),
                "strand": strand,
                **parse_gtf_attributes(attrs),
            }
def main():
    args = parse_args()
    wins = load_windows_csv(args.windows_file)
    mode = args.mode

    wins_by_chr = {}
    for w in wins:
        wins_by_chr.setdefault(w["chromosome"], []).append(w)

    out_fields = ["window", "chromosome", "gene_start", "gene_end", "strand",
                  "gene_id", "gene_name", "gene_biotype"]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=out_fields)
        writer.writeheader()

        count = 0
        matched = 0
        for g in iterate_gtf_genes(args.gtf):
            count += 1
            chr_wins = wins_by_chr.get(g["chromosome"], [])
            if not chr_wins:
                continue
            for w in chr_wins:
                if gene_matches_window(g["start"], g["end"], w["start"], w["end"], mode):
                    writer.writerow({
                        "window": w["window"],
                        "chromosome": g["chromosome"],
                        "gene_start": g["start"],
                        "gene_end": g["end"],
                        "strand": g["strand"],
                        "gene_id": g.get("gene_id", ""),
                        "gene_name": g.get("gene_name", ""),
                        "gene_biotype": g.get("gene_biotype", ""),
                    })
                    matched += 1
    print(f"Processed {count} genes; wrote {matched} matches to {args.output}")
if __name__ == "__main__":
    main()
