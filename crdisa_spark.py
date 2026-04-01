"""
CRDISA: Cognitive Reasoning-Driven Instance Selection Algorithm
Implemented on Apache Spark (PySpark) for distributed regression-task instance selection.

Performance Optimizations:
- Batch vectorized KDTree queries (eliminates inner Python loop)
- NumPy broadcast arithmetic for bulk difference computation
- Minimal serialization via partition-level processing
"""
import argparse
import sys
import time
import numpy as np

try:
    from pyspark.sql import SparkSession
    import pyspark.sql.functions as F
    from pyspark.sql.types import StructType, StructField, LongType, ArrayType, DoubleType
    from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
    from pyspark.ml import Pipeline
except ImportError:
    print("PySpark is required for this script.", file=sys.stderr)
    sys.exit(1)

try:
    from scipy.spatial import cKDTree
except ImportError:
    print("SciPy is required for this script.", file=sys.stderr)
    sys.exit(1)


def get_args():
    parser = argparse.ArgumentParser(description="CRDISA Spark Implementation")
    parser.add_argument("--input", type=str, default="hdfs:///project/weatherHistory.csv")
    parser.add_argument("--output", type=str, default="hdfs:///project/weather_crdisa_selected.csv")
    parser.add_argument("--target", type=str, default="Temperature (C)")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--partitions", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="Limit rows for testing (0 for all)")
    parser.add_argument("--naive", action="store_true", help="Run naive nested loop approach without vectorization")
    return parser.parse_args()


def preprocess_and_partition(spark, df, target_col, num_partitions):
    """Algorithm 2: Distributed Preprocessing."""
    # Drop text/date columns that aren't useful as numeric features
    text_and_date_cols = ['Formatted Date', 'Summary', 'Daily Summary',
                          'URL', 'TITLE', 'HOSTNAME', 'STORY']
    cols_present = [c for c in text_and_date_cols if c in df.columns]
    if cols_present:
        df = df.drop(*cols_present)

    dtypes = dict(df.dtypes)
    categorical_cols = [c for c, t in dtypes.items() if t == 'string' and c != target_col]
    numerical_cols = [c for c, t in dtypes.items() if t in ['int', 'double', 'float', 'bigint', 'long'] and c != target_col]

    stages = []
    encoded_categorical_cols = []
    for c in categorical_cols:
        indexer = StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        stages.append(indexer)
        encoded_categorical_cols.append(f"{c}_idx")

    assembler_inputs = numerical_cols + encoded_categorical_cols
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="raw_features", handleInvalid="skip")
    stages += [assembler]

    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)
    stages += [scaler]

    pipeline = Pipeline(stages=stages)
    model = pipeline.fit(df)
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

    df1 = df1.repartition(num_partitions)
    df2 = df2.repartition(num_partitions)

    return df1, df2


def collect_as_numpy(df):
    """Collect DataFrame to driver as numpy arrays."""
    pdf = df.toPandas()
    if pdf.empty:
        return np.array([]), np.array([]).reshape(0, 0), np.array([])
    ids = pdf['id'].values
    X = np.stack(pdf['features'].values)
    y = pdf['target'].values
    return ids, X, y


def calculate_soft_boundaries(X_train, y_train, X_val, k, alpha):
    """
    Section 4.2.2: Theta(x_j) = alpha * sigma(Y_Nk(x_j))
    Neighbors of each validation point are found in the TRAINING set.
    """
    tree = cKDTree(X_train)
    _, indices = tree.query(X_val, k=k)  # k neighbors from train set
    neighbor_targets = y_train[indices]   # shape: (m, k)
    sigmas = np.std(neighbor_targets, axis=1)  # shape: (m,)
    return alpha * sigmas


def perform_iteration(spark, train_df, val_df, k, alpha):
    """
    Algorithms 3, 4, 5: Expert Construction, Reasoning, Bidirectional Voting.

    KEY OPTIMIZATION: Instead of looping over validation instances one by one,
    we batch-compute all query points for an expert using NumPy broadcasting
    and then do a single batch KDTree query. This reduces per-expert work from
    O(m) Python calls to O(1) vectorized call.
    """
    print("  Collecting data to Driver for broadcasting...")
    t0 = time.time()
    t_ids, X_train, y_train = collect_as_numpy(train_df)
    v_ids, X_val, y_val = collect_as_numpy(val_df)

    if len(t_ids) == 0 or len(v_ids) == 0:
        return []

    n_train = len(t_ids)
    n_val = len(v_ids)
    print(f"  Train: {n_train}, Val: {n_val}")

    print("  Calculating soft boundaries (using train set neighbors)...")
    boundaries = calculate_soft_boundaries(X_train, y_train, X_val, k, alpha)

    sc = spark.sparkContext
    bc_X_train = sc.broadcast(X_train)
    bc_y_train = sc.broadcast(y_train)
    bc_X_val = sc.broadcast(X_val)
    bc_y_val = sc.broadcast(y_val)
    bc_boundaries = sc.broadcast(boundaries)
    eval_k = k

    def evaluate_experts_vectorized(iterator):
        """
        For each expert x_i in this partition:
          Per Definition 4 in the paper:
          1. Query difference: q = x_val - x_i
          2. Find k entries in Ki where δ_ip^x is closest to q
             Since δ_ip^x = x_i - x_p, we want ||δ_ip^x - q|| = ||(x_i-x_p)-(x_val-x_i)|| minimized
             = ||2x_i - x_p - x_val|| → find neighbors of (2x_i - x_val) in train set
          3. Predict: ŷ = y_i + (1/k) Σ δ_ip^y = y_i + (1/k) Σ (y_i - y_p)
                        = 2*y_i - mean(y_p)
        """
        X_t = bc_X_train.value
        y_t = bc_y_train.value
        X_v = bc_X_val.value
        y_v = bc_y_val.value
        bounds = bc_boundaries.value
        kk = eval_k

        # Build KDTree once per partition
        tree = cKDTree(X_t)

        # Collect all experts in this partition
        experts = []
        for row in iterator:
            experts.append((row.id, np.array(row.features), row.target))

        if not experts:
            return

        # Track diagnostics for first partition
        diag_errors = []
        diag_bounds = []

        for expert_id, x_i, y_i in experts:
            # VECTORIZED query points (Massive speedup trick)
            query_points = 2.0 * x_i - X_v  # shape: (m, d)

            # BATCH KDTree query
            _, idxs = tree.query(query_points, k=kk)  # shape: (m, k)

            # Paper formula: ŷ = y_i + mean(δ_ip^y) = 2*y_i - mean(y_p)
            neighbor_targets = y_t[idxs]  # shape: (m, k)
            mean_neighbor_y = np.mean(neighbor_targets, axis=1)  # shape: (m,)
            y_hat = 2.0 * y_i - mean_neighbor_y

            errors = np.abs(y_hat - y_v)
            votes = (errors <= bounds).astype(np.int32)

            if len(diag_errors) < 3:
                diag_errors.append(float(np.mean(errors)))
                diag_bounds.append(float(np.mean(bounds)))

            yield (expert_id, votes)

        # Print diagnostics from this partition
        if diag_errors:
            print(f"    [Partition Diag] avg_error={np.mean(diag_errors):.4f}, avg_bound={np.mean(diag_bounds):.4f}, ratio={np.mean(diag_errors)/max(np.mean(diag_bounds),1e-10):.2f}x")

    print("  Running distributed expert reasoning (Matrix M)...")
    matrix_rdd = train_df.rdd.mapPartitions(evaluate_experts_vectorized).cache()

    # --- Bidirectional Voting ---
    print("  Forward Voting (column-wise aggregation)...")
    total_experts = n_train
    column_sums = matrix_rdd.values().reduce(lambda a, b: a + b)

    # Diagnostic: print vote distribution
    print(f"  [Vote Diag] col_sums: min={column_sums.min()}, max={column_sums.max()}, "
          f"mean={column_sums.mean():.1f}, threshold={total_experts/2.0:.0f}")

    reliable_indices = np.where(column_sums > (total_experts / 2.0))[0]

    if len(reliable_indices) == 0:
        print("  WARNING: No reliable benchmarks found.")
        # Try relaxed thresholds
        for pct in [0.4, 0.3, 0.2]:
            relaxed = np.where(column_sums > total_experts * pct)[0]
            print(f"    At {pct*100:.0f}% threshold: {len(relaxed)} benchmarks")
        return []

    print(f"  Forward Voting: {len(reliable_indices)}/{n_val} benchmarks reliable.")

    bc_reliable = sc.broadcast(reliable_indices)

    print("  Backward Voting (row-wise filtering)...")
    def filter_experts(iterator):
        rel_idx = bc_reliable.value
        threshold = len(rel_idx) / 2.0
        for expert_id, votes in iterator:
            if np.sum(votes[rel_idx]) > threshold:
                yield expert_id

    selected = matrix_rdd.mapPartitions(filter_experts).collect()
    elapsed = time.time() - t0
    print(f"  Iteration done: {len(selected)}/{total_experts} selected ({elapsed:.1f}s)")

    bc_X_train.unpersist()
    bc_y_train.unpersist()
    bc_X_val.unpersist()
    bc_y_val.unpersist()
    bc_boundaries.unpersist()
    bc_reliable.unpersist()

    return selected


def main():
    args = get_args()

    print("=" * 60)
    print("CRDISA: Cognitive Reasoning-Driven Instance Selection")
    print("=" * 60)

    spark = SparkSession.builder \
        .appName("CRDISA Instance Selection") \
        .config("spark.driver.maxResultSize", "4g") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    t_start = time.time()
    print(f"Loading dataset from: {args.input}")
    try:
        df = spark.read.csv(args.input, header=True, inferSchema=True,
                            ignoreLeadingWhiteSpace=True, ignoreTrailingWhiteSpace=True)
        # Strip whitespace from column names (e.g. " Price" -> "Price")
        for c in df.columns:
            df = df.withColumnRenamed(c, c.strip())
        if args.limit > 0:
            df = df.limit(args.limit)
    except Exception as e:
        print(f"Failed to load dataset: {e}", file=sys.stderr)
        sys.exit(1)

    total_rows = df.count()
    print(f"Dataset rows: {total_rows}")

    print("Preprocessing & partitioning...")
    df1, df2 = preprocess_and_partition(spark, df, args.target, args.partitions)
    df1.cache()
    df2.cache()

    print("\n--- Cross-Experimentation (Algorithm Start) ---")
    t_algo_start = time.time()
    print(">> Iteration 1: D1=Experts, D2=Benchmarks")
    sel1 = perform_iteration(spark, train_df=df1, val_df=df2, k=args.k, alpha=args.alpha)

    print("\n>> Iteration 2: D2=Experts, D1=Benchmarks")
    sel2 = perform_iteration(spark, train_df=df2, val_df=df1, k=args.k, alpha=args.alpha)
    algo_time = time.time() - t_algo_start
    print(f"--- Algorithm done in {algo_time:.2f} seconds ---")

    final_ids = set(sel1).union(set(sel2))
    selected_count = len(final_ids)
    compression = (1.0 - selected_count / total_rows) * 100

    total_time = time.time() - t_start

    print("\n" + "=" * 60)
    print(f"RESULTS:")
    print(f"  Total Ext. Time:     {total_time:.2f} seconds")
    print(f"  Original Rows:       {total_rows}")
    print(f"  Selected Rows:       {selected_count}")
    print(f"  Compression Rate:    {compression:.2f}%")
    print("=" * 60)

    if not final_ids:
        print("Empty selection. Exiting.", file=sys.stderr)
        sys.exit(0)

    # Map logical IDs back to original rows
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
