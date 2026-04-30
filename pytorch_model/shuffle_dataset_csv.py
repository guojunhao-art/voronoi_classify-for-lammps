"""
对超大 dataset.csv 做可复现的外部打乱（不一次性读入内存）。

思路：
1) 第一阶段：按伪随机 key 将原始 CSV 分桶写入临时文件；
2) 第二阶段：逐桶读取、桶内随机打乱后写入输出 CSV。

说明：
- 这是“工程上足够随机”的外部打乱方案，适合超大文件。
- 通过 seed 固定可复现。
"""

import argparse
import os
import shutil
import tempfile

import numpy as np
import pandas as pd


def build_args():
    parser = argparse.ArgumentParser(description="External shuffle for huge CSV dataset.")
    parser.add_argument("--input", default="dataset.csv", help="输入 CSV 路径")
    parser.add_argument("--output", default="dataset.shuffled.csv", help="输出 CSV 路径")
    parser.add_argument("--chunksize", type=int, default=500_000, help="分块读取行数")
    parser.add_argument("--buckets", type=int, default=256, help="分桶数量（越大越随机，文件越多）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（可复现）")
    parser.add_argument("--tempdir", default=None, help="临时目录（默认系统临时目录）")
    parser.add_argument("--has-header", action="store_true", help="输入 CSV 含表头")
    return parser.parse_args()


def main():
    args = build_args()
    if args.buckets <= 1:
        raise ValueError("--buckets must be > 1")
    if args.chunksize <= 0:
        raise ValueError("--chunksize must be > 0")

    tmp_root = tempfile.mkdtemp(prefix="csv_shuffle_", dir=args.tempdir)
    bucket_dir = os.path.join(tmp_root, "buckets")
    os.makedirs(bucket_dir, exist_ok=True)

    header_opt = 0 if args.has_header else None
    names = None

    try:
        # 第一阶段：分桶
        row_offset = 0
        for chunk in pd.read_csv(args.input, chunksize=args.chunksize, header=header_opt):
            if args.has_header and names is None:
                names = list(chunk.columns)
            if args.has_header:
                chunk.columns = names

            n = len(chunk)
            if n == 0:
                continue

            row_ids = row_offset + np.arange(n, dtype=np.int64)
            rng = np.random.default_rng(args.seed + row_offset)
            bucket_ids = rng.integers(0, args.buckets, size=n)

            for b in range(args.buckets):
                sel = chunk.loc[bucket_ids == b]
                if len(sel) == 0:
                    continue
                bucket_file = os.path.join(bucket_dir, f"bucket_{b:04d}.csv")
                sel.to_csv(bucket_file, mode="a", index=False, header=False)

            row_offset += n

        # 第二阶段：逐桶打乱后写出
        if os.path.exists(args.output):
            os.remove(args.output)

        out_header = args.has_header and names is not None
        for b in range(args.buckets):
            bucket_file = os.path.join(bucket_dir, f"bucket_{b:04d}.csv")
            if not os.path.exists(bucket_file):
                continue

            bucket_df = pd.read_csv(bucket_file, header=None if not args.has_header else 0)
            bucket_df = bucket_df.sample(frac=1.0, random_state=args.seed + b).reset_index(drop=True)
            bucket_df.to_csv(args.output, mode="a", index=False, header=out_header)
            out_header = False

        print(f"Shuffled CSV written to: {args.output}")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()

