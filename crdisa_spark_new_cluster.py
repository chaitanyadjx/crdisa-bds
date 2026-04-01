"""
CRDISA: Expert Subsampling Variant with R²-Weighted Refinement Gate
Reduces O(n^2) expert-validator pairwise evaluation to O(n^1.5)
by randomly sampling sqrt(n) representative experts per partition.

After each binary pass/fail iteration, a greedy refinement stage re-evaluates
rejected points using a weighted objective:

    score(point) = ΔR² - λ · Δ(1 - compression)
                 = ΔR² - λ · (1 / n_train)

A rejected point is added back to the selected set only if its score > 0,
i.e. the R² gain it brings outweighs the compression cost of including it.

λ (--lambda_r2) controls the tradeoff:
  - λ → 0  : pure accuracy, compression barely penalised (more points kept)
  - λ → ∞  : pure compression, R² gain must be enormous to justify inclusion

R² is evaluated via KNN (same k) on the FULL held-out val set for exactness.
"""
import argparse
import sys
import time
import math
import numpy as np
import os

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
    parser = argparse.ArgumentParser(description="CRDISA Sampled Spark (O(n^1.5)) + R² Gate")
    parser.add_argument("--input",  type=str, default="hdfs:///project/weatherHistory.csv")
    parser.add_argument("--output", type=str, default="hdfs:///project/weather_crdisa_sampled.csv")
    parser.add_argument("--target", type=str, default="Temperature (C)")
    parser.add_argument("--k",      type=int,   default=5)
    parser.add_argument("--alpha",  type=float, default=1.8)
    parser.add_argument("--partitions", type=int, default=10)
    parser.add_argument("--limit",  type=int,   default=0)
    parser.add_argument("--sample_factor", type=float, default=1.0,
                        help="Benchmarks sampled = sample_factor * sqrt(n_val). Default 1.0.")
    parser.add_argument("--lambda_r2", type=float, default=0.5,
                        help=(
                            "Compression penalty weight in the R² gate. "
                            "score = ΔR² - lambda_r2 * (1/n_train). "
                            "Default 0.5."
                        ))
    parser.add_argument("--probe_size", type=int, default=500,
                        help="Validation points sampled once for greedy R² scoring. Default 500.")
    parser.add_argument("--max_benchmarks", type=int, default=5000,
                        help="Maximum benchmarks per iteration")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="Fraction of benchmarks an expert must predict correctly to be selected. Decrease for noisy data.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Preprocessing (unchanged)
# ---------------------------------------------------------------------------

def preprocess_and_partition(spark, df, target_col, num_partitions):
    text_cols = ['Formatted Date', 'Summary', 'Daily Summary',
                 'URL', 'TITLE', 'HOSTNAME', 'STORY']
    cols_present = [c for c in text_cols if c in df.columns]
    if cols_present:
        df = df.drop(*cols_present)

    dtypes = dict(df.dtypes)
    categorical_cols = [c for c, t in dtypes.items()
                        if t == 'string' and c != target_col]
    numerical_cols   = [c for c, t in dtypes.items()
                        if t in ('int', 'double', 'float', 'bigint', 'long')
                        and c != target_col]

    stages, encoded_cols = [], []
    for c in categorical_cols:
        indexer = StringIndexer(inputCol=c, outputCol=f"{c}_idx",
                                handleInvalid="keep")
        stages.append(indexer)
        encoded_cols.append(f"{c}_idx")

    assembler = VectorAssembler(
        inputCols=numerical_cols + encoded_cols,
        outputCol="raw_features", handleInvalid="skip")
    stages.append(assembler)
    scaler = StandardScaler(inputCol="raw_features", outputCol="features",
                            withStd=True, withMean=True)
    stages.append(scaler)

    model = Pipeline(stages=stages).fit(df)
    df_transformed = model.transform(df)

    df_final = (df_transformed
                .select("features",
                        F.col(target_col).cast("double").alias("target"))
                .dropna())

    rdd_indexed = df_final.rdd.zipWithIndex().map(
        lambda row: (row[1],
                     row[0].features.toArray().tolist(),
                     float(row[0].target))
    )

    schema = StructType([
        StructField("id",       LongType(),            False),
        StructField("features", ArrayType(DoubleType()), False),
        StructField("target",   DoubleType(),           False),
    ])

    df_indexed = spark.createDataFrame(rdd_indexed, schema)
    df1, df2 = df_indexed.randomSplit([0.5, 0.5], seed=42)
    return df1.repartition(num_partitions), df2.repartition(num_partitions)


def collect_as_numpy(df):
    pdf = df.toPandas()
    if pdf.empty:
        return np.array([]), np.array([]).reshape(0, 0), np.array([])
    return (pdf['id'].values,
            np.stack(pdf['features'].values),
            pdf['target'].values)


# ---------------------------------------------------------------------------
# R² helpers
# ---------------------------------------------------------------------------

def knn_r2(X_train: np.ndarray, y_train: np.ndarray,
           X_val: np.ndarray,   y_val: np.ndarray,
           k: int) -> float:
    """
    Compute R² of KNN regression trained on (X_train, y_train)
    evaluated on (X_val, y_val).

    Returns -inf when the selected set is empty or too small for k neighbours.
    """
    n = len(X_train)
    if n == 0:
        return 0.0
    k_eff = min(k, n)
    tree = cKDTree(X_train)
    _, idxs = tree.query(X_val, k=k_eff)
    if k_eff == 1:
        idxs = idxs[:, np.newaxis]
    y_hat = np.mean(y_train[idxs], axis=1)

    ss_res = np.sum((y_val - y_hat) ** 2)
    ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
    if ss_tot < 1e-10:
        return 0.0 if ss_res > 1e-10 else 1.0
    return float(1.0 - ss_res / ss_tot)


def greedy_r2_refinement(
    selected_ids:  list,
    rejected_mask: np.ndarray,     # bool mask into t_ids/X_train/y_train
    t_ids:         np.ndarray,
    X_train:       np.ndarray,
    y_train:       np.ndarray,
    X_probe:       np.ndarray,   # Optimization: use small probe set instead of full val
    y_probe:       np.ndarray,
    k:             int,
    lambda_r2:     float,
) -> list:
    """
    Greedy R²-gated refinement stage.

    For every point rejected by the binary CRDISA pass, compute:

        score = ΔR² - λ / n_train

    where ΔR² is the R² improvement from adding that single point to the
    current selected corpus, and 1/n_train is the compression cost of one
    extra inclusion.

    Points are evaluated in descending order of their individual |error| on
    the val set (those the current model is worst at get priority), so the
    sweep is as greedy-optimal as possible without full sorting.

    Complexity: O(n_rejected · k · log n_selected)  —  a single linear pass.
    """
    if not np.any(rejected_mask):
        return selected_ids

    n_train = len(t_ids)
    compression_cost = lambda_r2 / n_train   # fixed per-point penalty

    # --- indices of currently selected points ---
    sel_id_set = set(selected_ids)
    sel_mask = np.isin(t_ids, list(sel_id_set))

    X_sel = X_train[sel_mask]
    y_sel = y_train[sel_mask]

    # Baseline R² with the current selected set (on probe set)
    r2_current = knn_r2(X_sel, y_sel, X_probe, y_probe, k)

    # Candidate rejected points
    rej_idxs = np.where(rejected_mask)[0]

    # Prioritise candidates where the current model struggles most:
    # predict each val point and sort rejects by proximity to worst-predicted.
    # Fast heuristic: use per-reject-point nearest-val-neighbour error.
    if len(X_sel) > 0:
        tree_sel = cKDTree(X_sel)
        # For each rejected point, find its k-NN val neighbours' errors
        k_probe = min(k, len(X_sel))
        _, probe_idxs = tree_sel.query(X_train[rej_idxs], k=k_probe)
        if k_probe == 1:
            probe_idxs = probe_idxs[:, np.newaxis]
        y_pred_rej = np.mean(y_sel[probe_idxs], axis=1)
        # Use absolute error as proxy priority (descending = worst first)
        proxy_errors = np.abs(y_train[rej_idxs] - y_pred_rej)
        priority_order = np.argsort(-proxy_errors)   # worst-first
    else:
        priority_order = np.arange(len(rej_idxs))

    added = []

    for local_idx in priority_order:
        global_idx = rej_idxs[local_idx]

        # Tentatively add this point
        X_candidate = np.vstack([X_sel, X_train[global_idx]])
        y_candidate = np.append(y_sel, y_train[global_idx])

        r2_candidate = knn_r2(X_candidate, y_candidate, X_probe, y_probe, k)
        delta_r2 = r2_candidate - r2_current

        if delta_r2 - compression_cost > 0:
            # Accept: update running corpus and baseline R²
            X_sel = X_candidate
            y_sel = y_candidate
            r2_current = r2_candidate
            added.append(int(t_ids[global_idx]))

    return selected_ids + added


# ---------------------------------------------------------------------------
# Core CRDISA iteration (binary pass/fail, unchanged logic)
# ---------------------------------------------------------------------------

def perform_iteration_sampled(
    spark, train_df, val_df,
    k, alpha, sample_factor, lambda_r2, probe_size, max_benchmarks, beta,
):
    """
    O(n^1.5) binary expert selection, followed by a greedy R² refinement gate.

    Binary stage  — O(n_train · sqrt(n_val) · log n_train)
    Refinement    — O(n_rejected · k · log n_selected)   ≈ O(n · k · log n)
    Overall       — O(n^1.5) dominated by binary stage.
    """
    t0 = time.time()

    # 1. Collect full sets on the driver
    t_ids,     X_train, y_train = collect_as_numpy(train_df)
    v_ids_all, X_val,   y_val   = collect_as_numpy(val_df)

    if len(t_ids) == 0 or len(v_ids_all) == 0:
        return []

    n_train = len(t_ids)
    n_val   = len(v_ids_all)

    # ---- O(n log n) benchmark sampling (Capped) ----------------------------
    n_sample_benchmarks = min(max_benchmarks, max(k + 1, int(sample_factor * math.sqrt(n_val))))
    rng = np.random.default_rng(seed=42)
    sample_indices = rng.choice(n_val, size=n_sample_benchmarks, replace=False)

    X_val_sample = X_val[sample_indices]
    y_val_sample = y_val[sample_indices]

    print(f"  Total Experts        : {n_train}")
    print(f"  Reference Benchmarks : {n_sample_benchmarks} (from {n_val})")

    # Soft boundaries on sampled benchmarks  O(sqrt(n) · log n)
    tree_full = cKDTree(X_train)
    _, neighbor_idxs = tree_full.query(X_val_sample, k=k)
    boundaries = alpha * np.std(y_train[neighbor_idxs], axis=1)

    # ---- Broadcast reference data ------------------------------------------
    sc = spark.sparkContext
    bc_X_train    = sc.broadcast(X_train)
    bc_y_train    = sc.broadcast(y_train)
    bc_X_val_ref  = sc.broadcast(X_val_sample)
    bc_y_val_ref  = sc.broadcast(y_val_sample)
    bc_bounds     = sc.broadcast(boundaries)
    bc_n_sample   = sc.broadcast(n_sample_benchmarks)
    bc_beta       = sc.broadcast(beta)

    def evaluate_all_experts(iterator):
        X_t      = bc_X_train.value
        y_t      = bc_y_train.value
        X_v_ref  = bc_X_val_ref.value
        y_v_ref  = bc_y_val_ref.value
        bounds   = bc_bounds.value
        n_samp   = bc_n_sample.value
        thr_beta = bc_beta.value

        tree = cKDTree(X_t)
        results = []
        for row in iterator:
            x_i = np.array(row.features)
            y_i = row.target

            # Mirror-point query across each benchmark
            query_points = 2.0 * x_i - X_v_ref
            _, idxs = tree.query(query_points, k=k)
            y_hat = 2.0 * y_i - np.mean(y_t[idxs], axis=1)
            errors = np.abs(y_hat - y_v_ref)

            passed_count = np.sum(errors <= bounds)
            if passed_count > n_samp * thr_beta:
                results.append(row.id)
        return results

    # ---- Distributed binary pass -------------------------------------------
    t_binary_start = time.time()
    selected_ids = train_df.rdd.mapPartitions(evaluate_all_experts).collect()
    binary_time  = time.time() - t_binary_start

    for bc in [bc_X_train, bc_y_train, bc_X_val_ref, bc_y_val_ref,
               bc_bounds, bc_n_sample, bc_beta]:
        bc.unpersist()

    n_selected_binary = len(selected_ids)
    print(f"  Binary pass          : {n_selected_binary}/{n_train} selected "
          f"({binary_time:.1f}s)")

    # ---- R²-weighted greedy refinement gate --------------------------------
    t_refine_start = time.time()

    selected_id_set = set(selected_ids)
    rejected_mask   = ~np.isin(t_ids, list(selected_id_set))

    # Baseline R² on the FULL val set using the binary-selected corpus
    sel_mask = np.isin(t_ids, list(selected_id_set))
    r2_before = knn_r2(X_train[sel_mask], y_train[sel_mask], X_val, y_val, k)

    # --- PROBE SET SAMPLING (Optimization) ---
    n_probe = min(probe_size, n_val)
    probe_indices = rng.choice(n_val, size=n_probe, replace=False)
    X_probe = X_val[probe_indices]
    y_probe = y_val[probe_indices]

    refined_ids = greedy_r2_refinement(
        selected_ids, rejected_mask,
        t_ids, X_train, y_train,
        X_probe, y_probe,
        k, lambda_r2,
    )

    # Recompute R² after refinement
    refined_mask = np.isin(t_ids, list(set(refined_ids)))
    r2_after = knn_r2(X_train[refined_mask], y_train[refined_mask],
                      X_val, y_val, k)

    refine_time = time.time() - t_refine_start
    n_added     = len(refined_ids) - n_selected_binary
    compression = (1.0 - len(refined_ids) / n_train) * 100

    print(f"  Refinement gate      : +{n_added} points added "
          f"({refine_time:.1f}s)")
    print(f"  R² before / after    : {r2_before:.4f} → {r2_after:.4f}")
    print(f"  Compression (iter)   : {compression:.1f}%  "
          f"({len(refined_ids)}/{n_train} kept)")

    elapsed = time.time() - t0
    print(f"  Iteration total      : {elapsed:.1f}s")

    return refined_ids, r2_before, r2_after


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    spark = (SparkSession.builder
             .appName("CRDISA Sampled R²-Gate (O(n^1.5))")
             .config("spark.driver.memory",   "8g")
             .config("spark.executor.memory", "8g")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    t_start = time.time()
    print("=" * 60)
    print("CRDISA SAMPLED: O(n^1.5) + R²-Weighted Refinement Gate")
    print(f"  sample_factor = {args.sample_factor}  "
          f"(benchmarks = {args.sample_factor} * sqrt(n))")
    print(f"  lambda_r2     = {args.lambda_r2}  "
          f"(compression penalty weight)")
    print("=" * 60)

    try:
        df = spark.read.csv(args.input, header=True, inferSchema=True)
        # Trim whitespace from column names for robustness
        for col in df.columns:
            df = df.withColumnRenamed(col, col.strip())
        
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

    print(">> Iteration 1: D1=Experts, D2=Val")
    sel1, r2_1_before, r2_1_after = perform_iteration_sampled(
        spark, df1, df2, args.k, args.alpha, args.sample_factor, args.lambda_r2, args.probe_size, args.max_benchmarks, args.beta)

    print("\n>> Iteration 2: D2=Experts, D1=Val")
    sel2, r2_2_before, r2_2_after = perform_iteration_sampled(
        spark, df2, df1, args.k, args.alpha, args.sample_factor, args.lambda_r2, args.probe_size, args.max_benchmarks, args.beta)

    algo_time = time.time() - t_algo_start
    print(f"--- Algorithm done in {algo_time:.2f} seconds ---")

    final_ids      = set(sel1).union(set(sel2))
    selected_count = len(final_ids)
    compression    = (1.0 - selected_count / total_rows) * 100 if total_rows > 0 else 0
    total_time     = time.time() - t_start

    # Summary R² = average of both iteration post-refinement values
    avg_r2_before = (r2_1_before + r2_2_before) / 2.0
    avg_r2_after  = (r2_1_after  + r2_2_after)  / 2.0

    print("\n" + "=" * 60)
    print("RESULTS:")
    print(f"  Total Ext. Time     : {total_time:.2f}s")
    print(f"  Original Rows       : {total_rows}")
    print(f"  Selected Rows       : {selected_count}")
    print(f"  Compression Rate    : {compression:.2f}%")
    print(f"  Avg R² (binary)     : {avg_r2_before:.4f}")
    print(f"  Avg R² (refined)    : {avg_r2_after:.4f}")
    print(f"  Avg ΔR² from gate   : {avg_r2_after - avg_r2_before:+.4f}")
    print("=" * 60)

    if not final_ids:
        print("Empty selection. Exiting.")
        sys.exit(0)

    df_with_ids = spark.createDataFrame(
        df.rdd.zipWithIndex().map(
            lambda row: (row[1],) + tuple(row[0])
        ),
        StructType(
            [StructField("_rid", LongType(), False)] + df.schema.fields
        )
    )

    bc_ids = spark.sparkContext.broadcast(final_ids)
    selected_rdd = df_with_ids.rdd.filter(lambda row: row._rid in bc_ids.value)
    final_df = (spark
                .createDataFrame(selected_rdd, df_with_ids.schema)
                .drop("_rid"))

    print(f"Saving to: {args.output}")
    final_df.coalesce(1).write.csv(args.output, header=True, mode="overwrite")
    bc_ids.unpersist()
    print("Done.")


if __name__ == "__main__":
    main()