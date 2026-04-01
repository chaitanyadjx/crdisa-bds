"""
CRDISA: Expert Subsampling Variant
Reduces O(n^2) expert-validator pairwise evaluation to O(n^1.5)
by randomly sampling sqrt(n) representative experts per partition.

Key idea: Instead of evaluating ALL n/2 experts against ALL n/2 validators,
each partition samples only sqrt(n_train) experts. The vote aggregation
is then scaled by the sampling ratio, preserving the statistical threshold.
This reduces total KDTree queries from n^2/4 to ~n^1.5/4 without changing
the algorithm's logic or output semantics.
"""
import argparse
import sys
import time
import math
import numpy as np

try:
    from pyspark.sql import SparkSession
    import pyspark.sql.functions as F
    from pyspark.sql.types import StructType, StructField, LongType, ArrayType, DoubleType, IntegerType
    from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
    from pyspark.ml import Pipeline
    from scipy.spatial import cKDTree
except ImportError:
    print("PySpark and SciPy are required.", file=sys.stderr)
    sys.exit(1)


def get_args():
    parser = argparse.ArgumentParser(description="CRDISA Sampled Spark (O(n^1.5))")
    parser.add_argument("--input", type=str, default="hdfs:///project/weatherHistory.csv")
    parser.add_argument("--output", type=str, default="hdfs:///project/weather_crdisa_sampled.csv")
    parser.add_argument("--target", type=str, default="Temperature (C)")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--partitions", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample_factor", type=float, default=1.0,
                        help="Experts sampled = sample_factor * sqrt(n_train). Default 1.0.")
    return parser.parse_args()


def preprocess_and_partition(spark, df, target_col, num_partitions):
    text_cols = ['Formatted Date', 'Summary', 'Daily Summary', 'URL', 'TITLE', 'HOSTNAME', 'STORY']
    cols_present = [c for c in text_cols if c in df.columns]
    if cols_present:
        df = df.drop(*cols_present)

    dtypes = dict(df.dtypes)
    categorical_cols = [c for c, t in dtypes.items() if t == 'string' and c != target_col]
    numerical_cols = [c for c, t in dtypes.items() if t in ['int', 'double', 'float', 'bigint', 'long'] and c != target_col]

    stages = []
    encoded_cols = []
    for c in categorical_cols:
        indexer = StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        stages.append(indexer)
        encoded_cols.append(f"{c}_idx")

    assembler = VectorAssembler(inputCols=numerical_cols + encoded_cols, outputCol="raw_features", handleInvalid="skip")
    stages.append(assembler)
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)
    stages.append(scaler)

    model = Pipeline(stages=stages).fit(df)
    df_transformed = model.transform(df)

    df_final = df_transformed.select("features", F.col(target_col).cast("double").alias("target")).dropna()

    rdd_indexed = df_final.rdd.zipWithIndex().map(
        lambda row: (row[1], row[0].features.toArray().tolist(), float(row[0].target))
    )

    schema = StructType([
        StructField("id", LongType(), False),
        StructField("features", ArrayType(DoubleType()), False),
        StructField("target", DoubleType(), False)
    ])

    df_indexed = spark.createDataFrame(rdd_indexed, schema)
    df1, df2 = df_indexed.randomSplit([0.5, 0.5], seed=42)
    return df1.repartition(num_partitions), df2.repartition(num_partitions)


def collect_as_numpy(df):
    pdf = df.toPandas()
    if pdf.empty:
        return np.array([]), np.array([]).reshape(0, 0), np.array([])
    return pdf['id'].values, np.stack(pdf['features'].values), pdf['target'].values


def perform_iteration_sampled(spark, train_df, val_df, k, alpha, sample_factor):
    """
    Corrected O(n^1.5) CRDISA.
    Evaluates ALL experts against a sampled subset of sqrt(n) benchmarks.
    Total Complexity: O(n_train * sqrt(n_val)) = O(n^1.5).
    """
    t0 = time.time()
    
    # 1. Collect full training set as the 'search space' for KDTrees
    t_ids, X_train, y_train = collect_as_numpy(train_df)
    # 2. Collect full validation set to sample from
    v_ids_all, X_val_all, y_val_all = collect_as_numpy(val_df)

    if len(t_ids) == 0 or len(v_ids_all) == 0:
        return []

    n_train = len(t_ids)
    n_val = len(v_ids_all)
    
    # --- The sqrt(n) Sampling Logic ---
    n_sample_benchmarks = max(k + 1, int(sample_factor * math.sqrt(n_val)))
    
    # Randomly pick indices for our 'Reference Benchmarks'
    rng = np.random.default_rng(seed=42)
    sample_indices = rng.choice(n_val, size=n_sample_benchmarks, replace=False)
    
    X_val_sample = X_val_all[sample_indices]
    y_val_sample = y_val_all[sample_indices]
    v_ids_sample = v_ids_all[sample_indices]

    print(f"  Total Experts: {n_train}")
    print(f"  Reference Benchmarks: {n_sample_benchmarks} (sampled from {n_val})")

    # Pre-calculate soft boundaries for the sampled benchmarks only
    # O(sqrt(n) * log n)
    tree_full = cKDTree(X_train)
    _, neighbor_idxs = tree_full.query(X_val_sample, k=k)
    # Boundaries based on local variance of sampled benchmarks
    boundaries = alpha * np.std(y_train[neighbor_idxs], axis=1)

    # 3. Broadcast only the necessary reference data
    sc = spark.sparkContext
    bc_X_train = sc.broadcast(X_train)
    bc_y_train = sc.broadcast(y_train)
    bc_X_val_ref = sc.broadcast(X_val_sample)
    bc_y_val_ref = sc.broadcast(y_val_sample)
    bc_bounds = sc.broadcast(boundaries)

    def evaluate_all_experts(iterator):
        X_t = bc_X_train.value
        y_t = bc_y_train.value
        X_v_ref = bc_X_val_ref.value
        y_v_ref = bc_y_val_ref.value
        bounds = bc_bounds.value
        
        tree = cKDTree(X_t)
        
        results = []
        for row in iterator:
            expert_id = row.id
            x_i = np.array(row.features)
            y_i = row.target
            
            # Logic: Can this expert predict our sqrt(n) reference benchmarks?
            # Complexity per expert: O(sqrt(n) * log n)
            query_points = 2.0 * x_i - X_v_ref
            _, idxs = tree.query(query_points, k=k)
            
            # Predict targets for the benchmarks
            y_hat = 2.0 * y_i - np.mean(y_t[idxs], axis=1)
            errors = np.abs(y_hat - y_v_ref)
            
            # An expert is 'good' if it passes a threshold of the sampled benchmarks
            passed_count = np.sum(errors <= bounds)
            
            # Threshold: Must correctly predict > 50% of the SAMPLE
            if passed_count > (n_sample_benchmarks * 0.5):
                results.append(expert_id)
        return results

    # 4. Process ALL training partitions
    selected = train_df.rdd.mapPartitions(evaluate_all_experts).collect()

    elapsed = time.time() - t0
    print(f"  Iteration done: {len(selected)}/{n_train} selected ({elapsed:.1f}s)")

    # Clean up
    for bc in [bc_X_train, bc_y_train, bc_X_val_ref, bc_y_val_ref, bc_bounds]:
        bc.unpersist()

    return selected

def main():
    args = get_args()

    spark = SparkSession.builder \
        .appName("CRDISA Sampled (O(n^1.5))") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    t_start = time.time()
    print("=" * 60)
    print("CRDISA SAMPLED: O(n^1.5) Expert Subsampling")
    print(f"  sample_factor = {args.sample_factor}  (experts = {args.sample_factor} * sqrt(n))")
    print("=" * 60)

    try:
        df = spark.read.csv(args.input, header=True, inferSchema=True)
        if args.limit > 0:
            df = df.limit(args.limit)
    except Exception as e:
        print(f"Failed to load dataset: {e}", file=sys.stderr)
        sys.exit(1)

    total_rows = df.count()
    print(f"Dataset rows: {total_rows}")

    df1, df2 = preprocess_and_partition(spark, df, args.target, args.partitions)
    df1.cache()
    df2.cache()

    print("\n--- Cross-Experimentation (Algorithm Start) ---")
    t_algo_start = time.time()
    print(">> Iteration 1: D1=Experts, D2=Benchmarks")
    sel1 = perform_iteration_sampled(spark, df1, df2, args.k, args.alpha, args.sample_factor)

    print("\n>> Iteration 2: D2=Experts, D1=Benchmarks")
    sel2 = perform_iteration_sampled(spark, df2, df1, args.k, args.alpha, args.sample_factor)
    algo_time = time.time() - t_algo_start
    print(f"--- Algorithm done in {algo_time:.2f} seconds ---")

    final_ids = set(sel1).union(set(sel2))
    selected_count = len(final_ids)
    compression = (1.0 - selected_count / total_rows) * 100 if total_rows > 0 else 0
    total_time = time.time() - t_start

    print("\n" + "=" * 60)
    print("RESULTS:")
    print(f"  Total Ext. Time:     {total_time:.2f} seconds")
    print(f"  Original Rows:       {total_rows}")
    print(f"  Selected Rows:       {selected_count}")
    print(f"  Compression Rate:    {compression:.2f}%")
    print("=" * 60)

    if not final_ids:
        print("Empty selection. Exiting.")
        sys.exit(0)

    df_with_ids = spark.createDataFrame(
        df.rdd.zipWithIndex().map(lambda row: (row[1],) + tuple(row[0])),
        StructType([StructField("_rid", LongType(), False)] + df.schema.fields)
    )

    bc_ids = spark.sparkContext.broadcast(final_ids)
    selected_rdd = df_with_ids.rdd.filter(lambda row: row._rid in bc_ids.value)
    final_df = spark.createDataFrame(selected_rdd, df_with_ids.schema).drop("_rid")

    print(f"Saving to: {args.output}")
    final_df.coalesce(1).write.csv(args.output, header=True, mode="overwrite")
    print("Done.")


if __name__ == "__main__":
    main()
