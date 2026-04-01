import argparse
import sys
import time
import numpy as np

try:
    from pyspark.sql import SparkSession
    import pyspark.sql.functions as F
    from pyspark.sql.types import StructType, StructField, LongType, ArrayType, DoubleType
    from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, PCA, BucketedRandomProjectionLSH
    from pyspark.ml import Pipeline
    from pyspark.ml.linalg import Vectors, VectorUDT
    from scipy.spatial import cKDTree
except ImportError:
    print("PySpark and SciPy are required for this script.", file=sys.stderr)
    sys.exit(1)


def get_args():
    parser = argparse.ArgumentParser(description="Optimized CRDISA Spark Implementation (PCA + LSH + Co-Partition)")
    parser.add_argument("--input", type=str, default="hdfs:///project/weatherHistory.csv")
    parser.add_argument("--output", type=str, default="hdfs:///project/weather_crdisa_optimized.csv")
    parser.add_argument("--target", type=str, default="Temperature (C)")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--partitions", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="Limit rows (0 for all)")
    parser.add_argument("--pca_k", type=int, default=3, help="Number of Principal Components")
    parser.add_argument("--bucket_length", type=float, default=5.0, help="LSH bucket length (hyperparameter)")
    parser.add_argument("--num_hash_tables", type=int, default=3, help="LSH Hash tables")
    return parser.parse_args()


# UDF to convert Spark Vector to Python List (for easier RDD processing)
@F.udf(returnType=ArrayType(DoubleType()))
def vector_to_list(v):
    return v.toArray().tolist()


def preprocess_and_partition(spark, df, target_col, num_partitions, pca_k, bucket_length, num_hash_tables):
    """
    Algorithm 2 Optimized:
    1. Clean and encode data
    2. PCA Dimension Reduction
    3. LSH Hashing
    """
    text_and_date_cols = ['Formatted Date', 'Summary', 'Daily Summary', 'URL', 'TITLE', 'HOSTNAME', 'STORY']
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
    stages.append(assembler)

    # Scale before PCA
    scaler = StandardScaler(inputCol="raw_features", outputCol="scaled_features", withStd=True, withMean=True)
    stages.append(scaler)

    # 1. OPTIMIZATION: Dimensionality Reduction Preprocessing
    pca = PCA(k=pca_k, inputCol="scaled_features", outputCol="pca_features")
    stages.append(pca)

    pipeline = Pipeline(stages=stages)
    model = pipeline.fit(df)
    df_transformed = model.transform(df)

    # 2. OPTIMIZATION: Locality-Sensitive Hashing (LSH)
    lsh = BucketedRandomProjectionLSH(inputCol="pca_features", outputCol="hashes",
                                      bucketLength=bucket_length, numHashTables=num_hash_tables)
    lsh_model = lsh.fit(df_transformed)
    df_hashed = lsh_model.transform(df_transformed)

    # Extract the first hash bucket as the partitioning key
    # BucketedRandomProjectionLSH output 'hashes' is an Array of Vectors.
    # We want the first value of the first vector.
    @F.udf(returnType=DoubleType())
    def extract_bucket_id(hashes_arr):
        if hashes_arr and len(hashes_arr) > 0:
            return float(hashes_arr[0][0])
        return 0.0

    df_final = df_hashed.withColumn("bucket_id", extract_bucket_id(F.col("hashes"))).select(
        "bucket_id",
        vector_to_list("pca_features").alias("features"),
        F.col(target_col).cast("double").alias("target")
    ).dropna()

    # Index rows and convert to DF
    rdd_indexed = df_final.rdd.zipWithIndex().map(
        lambda row: (row[1], float(row[0].bucket_id), row[0].features, float(row[0].target))
    )
    schema = StructType([
        StructField("id", LongType(), False),
        StructField("bucket_id", DoubleType(), False),
        StructField("features", ArrayType(DoubleType()), False),
        StructField("target", DoubleType(), False)
    ])
    df_indexed = spark.createDataFrame(rdd_indexed, schema)

    df1, df2 = df_indexed.randomSplit([0.5, 0.5], seed=42)

    return df1, df2


def evaluate_experts_copartitioned(bucket_id, expert_iter, val_iter, k, alpha):
    """
    3. OPTIMIZATION: Distributed Join via Co-partitioning (cogroup)
    No broadcasts! Each partition processes only its local LSH bucket.
    """
    experts = list(expert_iter)
    validators = list(val_iter)

    if not experts or not validators:
        return []

    # Emit bucket sizes early so we have them to compute local thresholds
    n_exp = len(experts)
    n_val = len(validators)

    # Format arrays
    e_ids = [e[0] for e in experts]
    X_train = np.array([e[1] for e in experts])
    y_train = np.array([e[2] for e in experts])

    v_ids = [v[0] for v in validators]
    X_val = np.array([v[1] for v in validators])
    y_val = np.array([v[2] for v in validators])

    # If there are fewer training samples in this bucket than k+1, skip
    if len(X_train) <= k:
        return []

    # A. Local Soft Boundary Calculation
    tree_train = cKDTree(X_train)
    _, indices_bound = tree_train.query(X_val, k=k)
    neighbor_targets = y_train[indices_bound]
    sigmas = np.std(neighbor_targets, axis=1)
    bounds = alpha * sigmas

    results = []

    # B. Local Expert Reasoning
    for expert_id, x_i, y_i in zip(e_ids, X_train, y_train):
        # Query difference space: find neighbors of (2xi - xval) in train set
        query_points = 2.0 * x_i - X_val
        _, idxs_pred = tree_train.query(query_points, k=k)

        n_targets = y_train[idxs_pred]
        y_hat = 2.0 * y_i - np.mean(n_targets, axis=1)

        errors = np.abs(y_hat - y_val)
        votes = (errors <= bounds).astype(np.int32)
        
        # Yield (expert_id, bucket_size, [ (val_id, vote), ... ])
        # We need bucket_size downstream to know what "> 50% of experts" means for THIS benchmark
        voted_val_ids = [v_id for v_id, vote in zip(v_ids, votes) if vote == 1]
        results.append((expert_id, n_exp, n_val, voted_val_ids))

    return results


def perform_iteration_optimized(spark, train_df, val_df, k, alpha):
    """Optimized Iteration using RDD cogroup instead of Broadcast."""
    t0 = time.time()

    # Form Key-Value RDDs based on LSH bucket_id
    # Format: (bucket_id, (id, features, target))
    rdd_train = train_df.rdd.map(lambda r: (r.bucket_id, (r.id, r.features, r.target)))
    rdd_val = val_df.rdd.map(lambda r: (r.bucket_id, (r.id, r.features, r.target)))

    n_train = train_df.count()
    n_val = val_df.count()

    print(f"  Train: {n_train}, Val: {n_val}")
    print("  Mapping and Co-partitioning by LSH bucket...")

    # distributed JOIN via cogroup
    # RDD of (bucket_id, ( [expert1, expert2...], [val1, val2...] ))
    cogrouped = rdd_train.cogroup(rdd_val)

    print("  Running Sparse Distributed Reasoning (in-bucket only)...")
    # Evaluate locally in each bucket
    # Output: flattened RDD of (expert_id, n_experts_in_bucket, n_val_in_bucket, [val_id_1,...])
    expert_votes_rdd = cogrouped.flatMap(
        lambda kv: evaluate_experts_copartitioned(kv[0], kv[1][0], kv[1][1], k, alpha)
    ).cache()

    print("  Forward Voting (aggregating votes for validators)...")
    # Map to (val_id, (1, n_experts_in_bucket))   --> we need to know local bucket size to compute 50%
    val_vote_counts = expert_votes_rdd.flatMap(
        lambda row: [(vid, (1, row[1])) for vid in row[3]]
    ).reduceByKey(lambda a, b: (a[0] + b[0], a[1])) # sum votes, keep n_experts_in_bucket (it's the same for all votes to this vid)

    # A val instance is reliable if > 50% of the experts IN ITS BUCKET voted for it.
    reliable_val_rdd = val_vote_counts.filter(lambda x: x[1][0] > (x[1][1] / 2.0))
    reliable_val_ids = set(reliable_val_rdd.map(lambda x: x[0]).collect())

    print(f"  Forward Voting: {len(reliable_val_ids)}/{n_val} benchmarks reliable.")

    if not reliable_val_ids:
         print("  WARNING: No reliable benchmarks found.")
         return []

    print("  Backward Voting (expert selection)...")
    bc_reliable = spark.sparkContext.broadcast(reliable_val_ids)

    def select_experts(expert_row):
        expert_id, n_exp, n_val, voted_val_ids = expert_row
        rel_set = bc_reliable.value
        # Intersection between what this expert got right and the reliable benchmarks
        correct_reliable_count = sum(1 for vid in voted_val_ids if vid in rel_set)
        
        # A safe proxy for backward threshold in LSH is requiring the expert to predict 
        # a meaningful fraction of reliable benchmarks in its bucket.
        # local threshold = 15% of available reliable benchmarks, minimum 1
        return correct_reliable_count >= max(1, len(rel_set) * 0.15 * (n_val / float(n_val + n_exp)))

    selected_experts_rdd = expert_votes_rdd.filter(select_experts).map(lambda x: x[0])
    selected = selected_experts_rdd.collect()

    elapsed = time.time() - t0
    print(f"  Iteration done: {len(selected)}/{n_train} selected ({elapsed:.1f}s)")

    bc_reliable.unpersist()
    expert_votes_rdd.unpersist()

    return selected


def main():
    args = get_args()

    spark = SparkSession.builder \
        .appName("CRDISA Optimized (LSH + PCA + Co-partition)") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    t_start = time.time()

    print("============================================================")
    print("CRDISA OPTIMIZED: PCA + LSH + Co-Partitioning Join")
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

    print("Preprocessing & PCA Dimensionality Reduction...")
    df1, df2 = preprocess_and_partition(spark, df, args.target, args.partitions, 
                                        args.pca_k, args.bucket_length, args.num_hash_tables)
    df1.cache()
    df2.cache()

    print("\n--- Cross-Experimentation (Algorithm Start) ---")
    t_algo_start = time.time()
    print(">> Iteration 1: D1=Experts, D2=Benchmarks")
    sel1 = perform_iteration_optimized(spark, train_df=df1, val_df=df2, k=args.k, alpha=args.alpha)

    print("\n>> Iteration 2: D2=Experts, D1=Benchmarks")
    sel2 = perform_iteration_optimized(spark, train_df=df2, val_df=df1, k=args.k, alpha=args.alpha)
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

    # Reconstruct exact original dataset properly using subset
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
