"""
按“正癸烷规模”对其它物质数据做随机下采样，并合并为一个新训练集。

用法示例：
python build_balanced_dataset.py \
  --base-csv decane.csv \
  --other-csv butadiene.csv vinylacetylene.csv cyclopentadiene.csv naphthalene.csv \
  --output merged_balanced.csv \
  --seed 42 \
  --chunksize 500000
"""

import argparse
import os
from typing import List

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Build balanced dataset by downsampling non-base CSVs.")
    p.add_argument("--base-csv", required=True, help="基准数据集（例如正癸烷）")
    p.add_argument("--other-csv", nargs="+", required=True, help="需要下采样的其它数据集")
    p.add_argument("--output", default="merged_balanced.csv", help="合并输出路径")
    p.add_argument("--target-rows", type=int, default=0,
                   help="每个 other-csv 目标行数；0 表示自动使用 base-csv 行数")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--chunksize", type=int, default=500_000, help="分块读取大小")
    p.add_argument("--has-header", action="store_true", help="输入 CSV 含表头")
    return p.parse_args()


def count_rows(csv_path: str, chunksize: int, header_opt):
    n = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, header=header_opt):
        n += len(chunk)
    return n


def sample_csv_bernoulli(csv_path: str, out_path: str, p_keep: float, chunksize: int, header_opt, seed: int):
    rng = np.random.default_rng(seed)
    write_header = header_opt == 0
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, header=header_opt):
        if len(chunk) == 0:
            continue
        keep = rng.random(len(chunk)) < p_keep
        selected = chunk.loc[keep]
        if len(selected) > 0:
            selected.to_csv(out_path, mode="a", index=False, header=write_header)
            write_header = False


def append_csv(csv_path: str, out_path: str, write_header: bool, chunksize: int, header_opt):
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, header=header_opt):
        if len(chunk) == 0:
            continue
        chunk.to_csv(out_path, mode="a", index=False, header=write_header)
        write_header = False
    return write_header


def main():
    args = parse_args()
    header_opt = 0 if args.has_header else None

    base_rows = count_rows(args.base_csv, args.chunksize, header_opt)
    target_rows = args.target_rows if args.target_rows > 0 else base_rows
    print(f"[INFO] base rows = {base_rows}, target rows per other-csv = {target_rows}")

    if os.path.exists(args.output):
        os.remove(args.output)

    # 1) 先写入 base-csv 全量
    write_header = args.has_header
    write_header = append_csv(args.base_csv, args.output, write_header, args.chunksize, header_opt)

    # 2) 对每个 other-csv 按比例下采样后追加
    for idx, path in enumerate(args.other_csv):
        total = count_rows(path, args.chunksize, header_opt)
        if total == 0:
            print(f"[WARN] skip empty file: {path}")
            continue
        p_keep = min(1.0, target_rows / total)
        print(f"[INFO] {path}: total={total}, p_keep={p_keep:.6f}")
        tmp_out = args.output + f".tmp_{idx}.csv"
        if os.path.exists(tmp_out):
            os.remove(tmp_out)
        sample_csv_bernoulli(
            csv_path=path,
            out_path=tmp_out,
            p_keep=p_keep,
            chunksize=args.chunksize,
            header_opt=header_opt,
            seed=args.seed + idx * 100003,
        )
        write_header = append_csv(tmp_out, args.output, write_header, args.chunksize, header_opt)
        os.remove(tmp_out)

    print(f"[DONE] merged dataset written to: {args.output}")


if __name__ == "__main__":
    main()

