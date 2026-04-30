"""
对 40 维特征中的“原子类型”做映射替换（流式，适合大 CSV）。

映射规则由命令行指定，例如：
--map 1:3 --map 2:5
表示原子类型 1->3, 2->5，其它保持不变。

40 维特征按 10 组 * 4 维组织，每组第 4 列为类型：
索引 3, 7, 11, ..., 39（0-based）。
"""

import argparse
import os

import pandas as pd


TYPE_COLS = [3 + 4 * i for i in range(10)]


def parse_args():
    p = argparse.ArgumentParser(description="Remap atom types in 40-dim feature CSV.")
    p.add_argument("--input", required=True, help="输入 CSV 路径")
    p.add_argument("--output", required=True, help="输出 CSV 路径")
    p.add_argument("--chunksize", type=int, default=500_000, help="分块读取大小")
    p.add_argument("--has-header", action="store_true", help="输入 CSV 含表头")
    p.add_argument("--map", dest="maps", action="append", required=True,
                   help="类型映射，格式 old:new，可重复传入，如 --map 1:3 --map 2:5")
    return p.parse_args()


def parse_type_mapping(maps):
    mapping = {}
    for item in maps:
        if ":" not in item:
            raise ValueError(f"Illegal map '{item}', expected format old:new")
        old_s, new_s = item.split(":", 1)
        old_t = int(old_s.strip())
        new_t = int(new_s.strip())
        mapping[old_t] = new_t
    return mapping


def main():
    args = parse_args()
    header_opt = 0 if args.has_header else None
    mapping = parse_type_mapping(args.maps)

    if os.path.exists(args.output):
        os.remove(args.output)

    write_header = args.has_header
    for chunk in pd.read_csv(args.input, chunksize=args.chunksize, header=header_opt):
        if len(chunk) == 0:
            continue

        # 仅修改 40 维特征中的类型列；标签列（第 41 列）不变
        for c in TYPE_COLS:
            col = chunk.iloc[:, c]
            chunk.iloc[:, c] = col.replace(mapping)

        chunk.to_csv(args.output, mode="a", index=False, header=write_header)
        write_header = False

    print(f"[DONE] remapped CSV written to: {args.output}")
    print(f"[INFO] remapped feature type columns (0-based): {TYPE_COLS}")
    print(f"[INFO] applied mapping: {mapping}")


if __name__ == "__main__":
    main()
