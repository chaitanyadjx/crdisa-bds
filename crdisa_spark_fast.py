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
    from scipy.spatial import cKDTree
except ImportError:
    print("PySpark and SciPy are required for this script.", file=sys.stderr)
    sys.exit(1)


def get_args():
    parser = argparse.ArgumentParser(description="Fast Exact CRDISA Spark Implementation")
    parser.add_argument("--input", type=str, default="hdfs:///project/weatherHistory.csv")
    parser.add_argument("--output", type=str, default="hdfs:///project/weather_crdisa_fast.csv")
    parser.add_argument("--target", type=str, default="Temperature (C)")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--partitions", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="Limit rows for testing (0 for all)")
    return parser.parse_args()


def preprocess_and_partition(spark, df, target_col, num_partitions):
    """Preprocess data: encode categoricals, scale features, and split 50/50."""
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

    assembler_inputs = numerical_cols + encoded_cols
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="raw_features", handleInvalid="skip")
    stages.append(assembler)

    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)
    stages.append(scaler)

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
    pdf = df.toPandas()
    if pdf.empty:
        return np.array([]), np.array([]).reshape(0, 0), np.array([])
    ids = pdf['id'].values
    X = np.stack(pdf['features'].values)
    y = pdf['target'].values
    return ids, X, y


def calculate_soft_boundaries(X_train, y_train, X_val, k, alpha):
    tree = cKDTree(X_train)
    _, indices = tree.query(X_val, k=k)
    neighbor_targets = y_train[indices]
    sigmas = np.std(neighbor_targets, axis=1)
    return alpha * sigmas


def perform_iteration_fast(spark, train_df, val_df, k, alpha):
    """
    Executes CRDISA reasoning using broadcasted variables but utilizes a memory-efficient 
    sparse representation for candidate matrices, replacing the dense O(n^2) RDD storage.
    """
    t0 = time.time()
    t_ids, X_train, y_train = collect_as_numpy(train_df)
    v_ids, X_val, y_val = collect_as_numpy(val_df)

    if len(t_ids) == 0 or len(v_ids) == 0:
        return []

    n_train, n_val = len(t_ids), len(v_ids)
    print(f"  Train: {n_train}, Val: {n_val}")

    boundaries = calculate_soft_boundaries(X_train, y_train, X_val, k, alpha)

    sc = spark.sparkContext
    bc_X_train = sc.broadcast(X_train)
    bc_y_train = sc.broadcast(y_train)
    bc_X_val = sc.broadcast(X_val)
    bc_y_val = sc.broadcast(y_val)
    bc_boundaries = sc.broadcast(boundaries)
    eval_k = k

    def evaluate_experts_sparse(iterator):
        X_t = bc_X_train.value
        y_t = bc_y_train.value
        X_v = bc_X_val.value
        y_v = bc_y_val.value
        bounds = bc_boundaries.value

        tree = cKDTree(X_t)
        experts = [(row.id, np.array(row.features), row.target) for row in iterator]

        if not experts:
            return

        for expert_id, x_i, y_i in experts:
            query_points = 2.0 * x_i - X_v
            _, idxs = tree.query(query_points, k=eval_k)

            neighbor_targets = y_t[idxs]
            mean_neighbor_y = np.mean(neighbor_targets, axis=1)
            y_hat = 2.0 * y_i - mean_neighbor_y

            errors = np.abs(y_hat - y_v)
            # Find indices where vote is positive. This creates a sparse representation instead of dense matrix.
            voted_indices = np.where(errors <= bounds)[0].tolist()
            
            # Map index to actual validator id
            voted_val_ids = [v_ids[i] for i in voted_indices]
            
            yield (expert_id, voted_val_ids)

    print("  Running Sparse Distributed Reasoning...")
    expert_votes_rdd = train_df.rdd.mapPartitions(evaluate_experts_sparse).cache()

    print("  Forward Voting (aggregating sparse votes)...")
    # Instead of column reducing a dense N x M matrix, we accurately compute combinations 
    # of sparse elements using flatMap and reduceByKey.
    val_vote_counts = expert_votes_rdd.flatMap(
        lambda row: [(vid, 1) for vid in row[1]]
    ).reduceByKey(lambda a, b: a + b)

    reliable_val_rdd = val_vote_counts.filter(lambda x: x[1] > (n_train / 2.0))
    reliable_val_ids = set(reliable_val_rdd.map(lambda x: x[0]).collect())

    print(f"  Forward Voting: {len(reliable_val_ids)}/{n_val} benchmarks reliable.")

    if len(reliable_val_ids) == 0:
        print("  WARNING: No reliable benchmarks found.")
        return []

    print("  Backward Voting (sparse set filtering)...")
    bc_reliable = sc.broadcast(reliable_val_ids)

    def select_experts(expert_row):
        expert_id, voted_val_ids = expert_row
        rel_set = bc_reliable.value
        
        # Calculate intersection with reliable set
        correct_count = sum(1 for vid in voted_val_ids if vid in rel_set)
        return correct_count > (len(rel_set) / 2.0)

    selected_experts_rdd = expert_votes_rdd.filter(select_experts).map(lambda x: x[0])
    selected = selected_experts_rdd.collect()

    elapsed = time.time() - t0
    print(f"  Iteration done: {len(selected)}/{n_train} selected ({elapsed:.1f}s)")

    bc_X_train.unpersist()
    bc_y_train.unpersist()
    bc_X_val.unpersist()
    bc_y_val.unpersist()
    bc_boundaries.unpersist()
    bc_reliable.unpersist()
    expert_votes_rdd.unpersist()

    return selected


def main():
    args = get_args()

    spark = SparkSession.builder \
        .appName("CRDISA Fast Exact (Sparse Memory Optimization)") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    t_start = time.time()

    print("============================================================")
    print("CRDISA FAST EXACT: Sparse Matrix Memory Optimization")
    print("============================================================")

    print(f"Loading dataset from: {args.input}")
    try:
        df = spark.read.csv(args.input, header=True, inferSchema=True)
        if args.limit > 0:
            df = df.limit(args.limit)
    except Exception as e:
        print(f"Failed to load dataset: {e}", file=sys.stderr)
        sys.exit(1)

    total_rows = df.count()
    print(f"Dataset rows: {total_rows}")

    print("Preprocessing...")
    df1, df2 = preprocess_and_partition(spark, df, args.target, args.partitions)
    df1.cache()
    df2.cache()

    print("\n--- Cross-Experimentation (Algorithm Start) ---")
    t_algo_start = time.time()
    print(">> Iteration 1: D1=Experts, D2=Benchmarks")
    sel1 = perform_iteration_fast(spark, train_df=df1, val_df=df2, k=args.k, alpha=args.alpha)

    print("\n>> Iteration 2: D2=Experts, D1=Benchmarks")
    sel2 = perform_iteration_fast(spark, train_df=df2, val_df=df1, k=args.k, alpha=args.alpha)
    algo_time = time.time() - t_algo_start
    print(f"--- Algorithm done in {algo_time:.2f} seconds ---")

    final_ids = set(sel1).union(set(sel2))
    selected_count = len(final_ids)
    compression = (1.0 - selected_count / total_rows) * 100 if total_rows > 0 else 0

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
