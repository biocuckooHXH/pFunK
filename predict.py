#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pFunK-Kbhb predictor (clean version)
- 支持两种输入：
  A) --sites_tsv  直接提供 [id, fasta_K] 的 61-mer 列表
  B) --fasta + --sites  提供整条FASTA与位点表[id, pos(1-based, 必须是K)]，脚本会自动切 61-mer
- 先解压 models/pFunKbHb 下的 *.zip，再把 --model_path 指向解压得到的 .h5
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.models import load_model

# 21 tokens: 20 standard AAs + X (padding/unknown)
AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY") + ["X"]
amino_acid_dict = {aa: i for i, aa in enumerate(AA_ORDER)}

WINDOW_SIZE = 61
HALF = WINDOW_SIZE // 2  # 30

def clamp(seq, left, right):
    """安全切片"""
    left = max(left, 0)
    right = min(right, len(seq))
    return left, right

def pad_61mer(seq, center_idx_0based):
    """
    给定整条序列和中心位点（0-based），生成以该位点为中心的 61-mer。
    中心字符必须为 'K'。两端不足用 'X' 补齐到 61。
    """
    assert 0 <= center_idx_0based < len(seq)
    assert seq[center_idx_0based].upper() == 'K', f"Center residue at {center_idx_0based+1} is not K."
    seq = seq.upper()
    l = center_idx_0based - HALF
    r = center_idx_0based + HALF + 1  # exclusive
    l2, r2 = clamp(seq, l, r)
    core = seq[l2:r2]
    left_pad = "X" * (l2 - l)
    right_pad = "X" * (r - r2)
    win = left_pad + core + right_pad
    assert len(win) == WINDOW_SIZE
    return win

def encode_seq61(seq61):
    """把 61-mer 映射为整数索引数组"""
    seq61 = seq61.upper()
    ints = []
    for ch in seq61:
        ints.append(amino_acid_dict.get(ch, amino_acid_dict["X"]))
    return np.asarray(ints, dtype=np.int32)

def read_fasta(fp):
    """简单读取FASTA，返回 dict[id] = sequence"""
    records = {}
    sid = None
    seq_chunks = []
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if sid is not None:
                    records[sid] = "".join(seq_chunks)
                sid = line[1:].split()[0]
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if sid is not None:
            records[sid] = "".join(seq_chunks)
    return records

def build_sites_from_fasta_and_pos(fasta_path, sites_pos_tsv):
    """
    sites_pos_tsv: 两列 [id, pos]，pos 为 1-based 且对应字符必须是 K
    返回 DataFrame: [id, fasta_K]
    """
    db = read_fasta(fasta_path)
    df = pd.read_csv(sites_pos_tsv, sep="\t")
    out_rows = []
    for _, row in df.iterrows():
        pid = str(row["id"])
        pos1 = int(row["pos"])
        assert pid in db, f"{pid} not found in FASTA."
        seq = db[pid]
        idx0 = pos1 - 1
        assert seq[idx0].upper() == "K", f"{pid}:{pos1} is {seq[idx0]}, not K."
        win = pad_61mer(seq, idx0)
        out_rows.append({"id": f"{pid}_{pos1}", "fasta_K": win})
    return pd.DataFrame(out_rows)

def load_sites_tsv(sites_tsv):
    """
    读取已经准备好的 [id, fasta_K]；校验长度与字符集
    """
    df = pd.read_csv(sites_tsv, sep="\t")
    assert {"id", "fasta_K"}.issubset(df.columns), "sites.tsv must have columns: id, fasta_K"
    allowed = set(AA_ORDER)
    bad = []
    for i, s in enumerate(df["fasta_K"].astype(str)):
        if len(s) != WINDOW_SIZE:
            bad.append((i, "length", len(s)))
            continue
        if any(ch.upper() not in allowed for ch in s):
            bad.append((i, "alphabet", s))
    if bad:
        raise ValueError(f"Invalid fasta_K rows found (index, reason, val) e.g. {bad[:3]}")
    return df[["id", "fasta_K"]].copy()

def main():
    ap = argparse.ArgumentParser()
    # 方式 A
    ap.add_argument("--sites_tsv", type=str, default=None,
                    help="TSV with columns: id, fasta_K (61 aa centered at K)")
    # 方式 B
    ap.add_argument("--fasta", type=str, default=None, help="Protein FASTA file")
    ap.add_argument("--sites", type=str, default=None,
                    help="TSV with columns: id, pos (1-based K position)")

    ap.add_argument("--model_path", type=str, required=True,
                    help="Path to the unzipped Keras .h5 model, e.g., models/pFunKbHb/pFunK_Transformer.model.h5")
    ap.add_argument("--output", type=str, required=True, help="Path to write results.tsv")
    args = ap.parse_args()

    # 准备输入
    if args.sites_tsv:
        df = load_sites_tsv(args.sites_tsv)
    else:
        assert args.fasta and args.sites, "If --sites_tsv is not provided, you must give --fasta and --sites."
        df = build_sites_from_fasta_and_pos(args.fasta, args.sites)

    # 编码为整数索引
    X = np.stack([encode_seq61(s) for s in df["fasta_K"].astype(str).tolist()], axis=0)

    # 加载模型并预测
    model = load_model(args.model_path)
    scores = model.predict(X, verbose=0).reshape(-1)

    # 写出
    out = pd.DataFrame({"id": df["id"].tolist(), "score": scores})
    out.to_csv(args.output, sep="\t", index=False)
    print(f"Done. Saved {len(out)} predictions to {args.output}")

if __name__ == "__main__":
    main()
