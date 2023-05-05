"""

"""
import os
import argparse
import time
import shutil

import dask.dataframe as dd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_omop_parquet_source", type=str)
    parser.add_argument("--path_omop_csv_destination", type=str)
    parser.add_argument("--overwrite", default=False, action="store_true")
    parser.add_argument("--repartition_size", type=str, default=None)
    parser.add_argument("--compression", type=str, default=None)
    args = parser.parse_args()

    if args.overwrite:
        shutil.rmtree(args.path_omop_csv_destination, ignore_errors=True)

    os.makedirs(args.path_omop_csv_destination, exist_ok=True)
    tables = os.listdir(args.path_omop_parquet_source)

    t_start = time.time()

    for table in tables:
        print(f"Converting {table}")
        t0 = time.time()

        df = dd.read_parquet(os.path.join(args.path_omop_parquet_source, table))

        if args.repartition_size is not None:
            df = df.repartition(partition_size=args.repartition_size)

        df.to_csv(
            os.path.join(args.path_omop_csv_destination, table, f"{table}-*.csv"),
            compression=args.compression,
            sep="\t",
        )

        t1 = (time.time() - t0) / 60
        print(f"{table} conversion completed in {t1} minutes")

    t_end = (time.time() - t_start) / 60
    print(f"Conversion of all tables completed in {t_end} minutes")


if __name__ == "__main__":
    main()
