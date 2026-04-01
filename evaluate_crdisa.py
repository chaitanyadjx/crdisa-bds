"""
CRDISA Evaluation Script (Fixed)
Ensures BOTH the baseline model and the CRDISA model are evaluated on
the IDENTICAL held-out test set, and neither leaks test rows into training.

Fix: The selected CRDISA rows are intersected with the 80% train split
     before training, so no test rows contaminate the CRDISA model.
"""
import sys
import argparse

try:
    from pyspark.sql import SparkSession
    import pyspark.sql.functions as F
    from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
    from pyspark.ml.regression import LinearRegression, RandomForestRegressor
    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.ml import Pipeline
except ImportError:
    print("PySpark is required.", file=sys.stderr)
    sys.exit(1)


def build_pipeline(df, target_col, model_type="linear"):
    text_cols = ['Formatted Date', 'Summary', 'Daily Summary',
                 'URL', 'TITLE', 'HOSTNAME', 'STORY']
    df = df.drop(*[c for c in text_cols if c in df.columns])

    dtypes = dict(df.dtypes)
    cat_cols = [c for c, t in dtypes.items() if t == 'string' and c != target_col]
    num_cols = [c for c, t in dtypes.items()
                if t in ['int', 'double', 'float', 'bigint', 'long'] and c != target_col]

    stages, enc_cols = [], []
    for c in cat_cols:
        stages.append(StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep"))
        enc_cols.append(f"{c}_idx")
    
    stages.append(VectorAssembler(inputCols=num_cols + enc_cols, outputCol="raw_features", handleInvalid="skip"))
    stages.append(StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True))
    
    if model_type == "rf":
        stages.append(RandomForestRegressor(featuresCol="features", labelCol=target_col, numTrees=20))
    else:
        stages.append(LinearRegression(featuresCol="features", labelCol=target_col))
        
    return Pipeline(stages=stages), df


def get_args():
    parser = argparse.ArgumentParser(description="CRDISA Evaluation Script (Fixed)")
    parser.add_argument("--orig",     type=str, default="hdfs:///project/weatherHistory.csv")
    parser.add_argument("--selected", type=str, default="hdfs:///project/weather_crdisa_selected.csv")
    parser.add_argument("--target",   type=str, default="Temperature (C)")
    parser.add_argument("--limit",    type=int, default=0)
    parser.add_argument("--model",    type=str, default="linear", choices=["linear", "rf"])
    return parser.parse_args()


def main():
    args = get_args()
    args.target = args.target.strip()

    spark = SparkSession.builder \
        .appName("CRDISA Evaluation (Fixed Split)") \
        .config("spark.sql.shuffle.partitions", "32") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    print("=" * 60)
    print("CRDISA EVALUATION  (train-test safe, same validation set)")
    print("=" * 60)

    # ── Load original dataset ────────────────────────────────────────
    print(f"\nLoading original dataset... [limit={args.limit if args.limit > 0 else 'ALL'}]")
    df_orig = spark.read.csv(
        args.orig, header=True, inferSchema=True, 
        ignoreLeadingWhiteSpace=True, ignoreTrailingWhiteSpace=True
    ).dropna()
    
    # Strip whitespace from columns to avoid mismatch errors
    for c in df_orig.columns:
        df_orig = df_orig.withColumnRenamed(c, c.strip())
        
    if args.limit > 0:
        df_orig = df_orig.limit(args.limit)

    orig_count  = df_orig.count()

    # ── Load CRDISA selected dataset ─────────────────────────────────
    print(f"\nLoading CRDISA selected dataset from:\n  {args.selected}")
    df_selected_raw = spark.read.csv(
        args.selected, header=True, inferSchema=True,
        ignoreLeadingWhiteSpace=True, ignoreTrailingWhiteSpace=True
    ).dropna()
    
    # Strip whitespace from columns to avoid mismatch errors
    for c in df_selected_raw.columns:
        df_selected_raw = df_selected_raw.withColumnRenamed(c, c.strip())
        
    raw_sel_count = df_selected_raw.count()

    # ── ALIGN SCHEMAS ────────────────────────────────────────────────
    # If df_orig inferred a column as string (due to hidden artifacts) but 
    # df_selected_raw inferred it correctly as double, cast df_orig to match.
    import pyspark.sql.functions as F
    sel_dtypes = dict(df_selected_raw.dtypes)
    for c in df_orig.columns:
        if c in sel_dtypes and dict(df_orig.dtypes)[c] != sel_dtypes[c]:
            df_orig = df_orig.withColumn(c, F.col(c).cast(sel_dtypes[c]))

    # ── KEY FIX: intersect selected set with training split only ─────
    # We join selected rows with train_orig on ALL shared columns so that
    # any test-set rows that happened to be selected are excluded.
    # This prevents data leakage into the CRDISA model.
    # IMPORTANT: We must perform the split AFTER schema alignment.
    train_orig, test_set = df_orig.randomSplit([0.8, 0.2], seed=42)
    test_set.cache()

    train_count = train_orig.count()
    test_count  = test_set.count()
    print(f"  Original rows:  {orig_count:,}")
    print(f"  Train portion:  {train_count:,}  (80%)")
    print(f"  Test  portion:  {test_count:,}  (20%)  ← shared by both models")

    shared_cols = [c for c in df_selected_raw.columns if c in train_orig.columns]
    # Use leftsemi: keeps only selected rows that appear in train, WITHOUT duplicating rows
    # (inner join would multiply rows when train has duplicate feature vectors)
    df_crdisa_train = df_selected_raw.join(
        train_orig.select(shared_cols),
        on=shared_cols,
        how="leftsemi"
    )
    crdisa_train_count = df_crdisa_train.count()
    leaked = raw_sel_count - crdisa_train_count
    compression = (1.0 - crdisa_train_count / orig_count) * 100

    print(f"\n--- Selection Summary ---")
    print(f"  Raw selected rows:          {raw_sel_count:,}")
    print(f"  After test-set exclusion:   {crdisa_train_count:,}  (removed {leaked} test-set overlaps)")
    print(f"  Effective compression rate: {compression:.2f}%")

    # ── Train baseline on full train split ───────────────────────────
    print(f"\nTraining baseline model ({args.model}) on full 80% train split...")
    pipe_orig, train_orig_clean = build_pipeline(train_orig, args.target, args.model)
    model_orig  = pipe_orig.fit(train_orig_clean)
    preds_orig  = model_orig.transform(test_set)

    # ── Train CRDISA model on leak-free selected rows ────────────────
    print(f"Training CRDISA model ({args.model}) on clean selected rows (no test overlap)...")
    pipe_sel, df_sel_clean = build_pipeline(df_crdisa_train, args.target, args.model)
    model_sel  = pipe_sel.fit(df_sel_clean)
    preds_sel  = model_sel.transform(test_set)

    # ── Evaluate on same held-out test set ───────────────────────────
    r2_eval   = RegressionEvaluator(labelCol=args.target, predictionCol="prediction", metricName="r2")
    rmse_eval = RegressionEvaluator(labelCol=args.target, predictionCol="prediction", metricName="rmse")

    r2_orig   = r2_eval.evaluate(preds_orig)
    rmse_orig = rmse_eval.evaluate(preds_orig)
    r2_sel    = r2_eval.evaluate(preds_sel)
    rmse_sel  = rmse_eval.evaluate(preds_sel)

    pct_diff      = ((r2_sel - r2_orig) / max(abs(r2_orig), 1e-10)) * 100
    rmse_pct_diff = ((rmse_sel - rmse_orig) / max(abs(rmse_orig), 1e-10)) * 100

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY  (both models evaluated on same test set)")
    print("=" * 60)
    print(f"\n  {'Metric':<25} {'Baseline':>12} {'CRDISA':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    print(f"  {'R² Score':<25} {r2_orig:>12.4f} {r2_sel:>12.4f}")
    print(f"  {'RMSE':<25} {rmse_orig:>12.4f} {rmse_sel:>12.4f}")
    print(f"  {'Training Rows':<25} {train_count:>12,} {crdisa_train_count:>12,}")
    print(f"  {'Compression Rate':<25} {'N/A':>12} {compression:>11.2f}%")
    
    print(f"\n  R² Change:   {pct_diff:+.2f}%")
    print(f"  RMSE Change: {rmse_pct_diff:+.2f}%")
    
    if pct_diff >= 0:
        print("  ✓ CRDISA maintains or improves R² score")
    else:
        print(f"  ✗ CRDISA shows {abs(pct_diff):.2f}% R² drop")

    if rmse_pct_diff <= 0:
        print("  ✓ CRDISA maintains or improves RMSE (Error decreased)")
    else:
        print(f"  ✗ CRDISA shows {rmse_pct_diff:.2f}% RMSE increase")
    print("\n" + "=" * 60)

    # ── Auto-save results to CSV ─────────────────────────────────────
    import os
    import csv
    log_file = "evaluation_results.csv"
    mode = 'a' if os.path.exists(log_file) else 'w'
    try:
        with open(log_file, mode, newline='') as f:
            writer = csv.writer(f)
            if mode == 'w':
                writer.writerow(['Dataset', 'Target', 'Model', 'Original_Rows', 'Selected_Rows', 'Compression_%', 'Baseline_R2', 'CRDISA_R2', 'Baseline_RMSE', 'CRDISA_RMSE', 'R2_Change_%', 'RMSE_Change_%'])
            
            # Clean dataset path for naming
            dataset_name = os.path.basename(args.orig) if args.orig else "Unknown"
            
            writer.writerow([
                dataset_name, 
                args.target, 
                args.model,
                orig_count, 
                crdisa_train_count, 
                round(compression, 2), 
                round(r2_orig, 4), 
                round(r2_sel, 4), 
                round(rmse_orig, 4), 
                round(rmse_sel, 4), 
                round(pct_diff, 2), 
                round(rmse_pct_diff, 2)
            ])
        print(f"  [✔] Metrics successfully logged to: {log_file}")
    except Exception as e:
        print(f"  [!] Failed to save metrics to CSV: {e}")

    spark.stop()


if __name__ == "__main__":
    main()
