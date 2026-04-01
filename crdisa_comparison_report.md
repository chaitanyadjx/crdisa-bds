# CRDISA Scripts Comparison Report

We ran both the **Normal** (`crdisa_spark.py`) and **Optimized** (`crdisa_spark_optimized.py`) versions of the CRDISA algorithm on a subset of the `weatherHistory.csv` dataset. The subset size was limited to **5,000 rows**.

## Results

| Metric | Normal Script (`crdisa_spark.py`) | Optimized Script (`crdisa_spark_optimized.py`) |
| :--- | :--- | :--- |
| **Total Execution Time** | 41.72 seconds | 43.72 seconds |
| **Original Rows** | 5,000 | 5,000 |
| **Selected Rows** | 3,148 | 4,372 |
| **Compression Rate** | 37.04% | 12.56% |

## Observations & Analysis

1. **Why Optimized Took More Time**: For a small dataset of 5,000 rows, the Optimized version took slightly longer (43.72s vs 41.72s). This is because the optimizations introduced—**PCA (Principal Component Analysis)** for dimensionality reduction and **LSH (Locality-Sensitive Hashing)** for bucketing—require upfront computational overhead. In PySpark, these pipeline stages have a fixed overhead cost. For 5,000 rows, a naive KDTree easily fits in memory, so the overhead of LSH dominates. However, as dataset size scales to millions of rows, the `cogroup` distributed join avoids out-of-memory broadcasting failures, making the Optimized script infinitely more scalable.
2. **Compression Rate**: The Normal script achieved a higher compression rate (37.04%) compared to the Optimized script (12.56%). 
   * The normal script evaluates points across the entire dataset globally using broadcasting.
   * The optimized script restricts comparisons only to local LSH buckets. This restriction makes the algorithm much more scalable but changes the local neighborhood context, causing it to retain more points overall in this specific small subset.
3. **Scalability vs Exactness**: The Optimized version correctly avoided broadcasting the full dataset by utilizing Spark's `cogroup` feature over LSH buckets. This prevents the primary bottleneck of standard KDTree approaches in PySpark.

## Predictive Evaluation Results
We evaluated the predictive performance of models trained on the CRDISA-selected instances versus a baseline model trained on the original dataset (using an 80/20 train/test split).

| Metric | Baseline | Normal CRDISA | Optimized CRDISA |
| :--- | :--- | :--- | :--- |
| **R² Score** | 0.9925 | 0.9901 | 0.9921 |
| **R² Change** | N/A | -0.25% | -0.04% |
| **RMSE** | 0.9584 | 1.1065 | 0.9860 |

## Overall Composite Score
To provide a single metric combining the three key factors (**Execution Time**, **Compression Rate**, and **R² Score**), we apply a weighted heuristic score out of 100:
* **R² Weight (50%)**: `R² * 50` (Predictive performance is most critical)
* **Compression Weight (30%)**: `(Compression % / 50%) * 30` (Assuming 50% is a "perfect" compression rate constraint limit)
* **Time Efficiency Weight (20%)**: `max(0, 100 - Time(s)) / 100 * 20` (Rewarding faster execution under 100s)

**Composite Score Formula**:
`Score = (R² * 50) + (Compression Rate * 0.6) + ((100 - Execution Time) * 0.2)`

| Implementation | R² Component (max 50) | Compression Component (max 30) | Time Component (max 20) | **Final Score / 100** |
| :--- | :--- | :--- | :--- | :--- |
| **Normal CRDISA** | 0.9901 * 50 = 49.50 | 37.04 * 0.6 = 22.22 | (100 - 41.72) * 0.2 = 11.65 | **83.37** |
| **Optimized CRDISA** | 0.9921 * 50 = 49.60 | 12.56 * 0.6 = 7.53 | (100 - 43.72) * 0.2 = 11.25 | **68.38** |

*Conclusion*: For a 5,000-row sample, the **Normal CRDISA** script is formally the "winner" with a higher composite score, strictly driven by its substantially better compression rate on small datasets. However, the Optimized version remains structurally superior for Big Data scenarios where the Normal script's broadcasting would cause `OutOfMemoryError` crashes.

## Outputs
- Normal output saved at: `dataset/normal_output.csv` (3,148 rows)
- Optimized output saved at: `dataset/optimized_output.csv` (4,372 rows)
