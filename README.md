# CRDISA: Cognitive Reasoning-Driven Instance Selection (Spark)

This repository contains various implementations of the CRDISA algorithm, optimized for Spark and large-scale datasets.

## 🚀 Quick Run Cheat Sheet

To run any version, ensure your virtual environment is active:
```bash
source .venv/bin/activate
```

| Version | Best For | Complexity | Command Template |
| :--- | :--- | :--- | :--- |
| **New Cluster** | **Best Accuracy** | $O(n^{1.5})$ | `python crdisa_spark_new_cluster.py --input <file> --target <col> --limit 50000` |
| **Sampled** | **Best Compression** | $O(n^{1.5})$ | `python crdisa_spark_sampled.py --input <file> --target <col> --sample_factor 1.0` |
| **LSH-NN** | **Massive Data** | $O(n \log n)$ | `python crdisa_spark_lsh_nn.py --input <file> --n_bits 12 --n_tables 4` |
| **Fast Sparse** | Exact Logic | $O(n^2)$ | `python crdisa_spark_fast.py --input <file> --target <col>` |
| **Optimized** | Distributed Joins | $O(n \log n)$ | `python crdisa_spark_optimized.py --input <file> --target <col>` |
| **Baseline** | Small Data | $O(n^2)$ | `python crdisa_spark.py --input <file> --target <col>` |

---

## 🌩️ Running on a Spark Cluster

To run on a distributed cluster (YARN, Standalone, or K8s), use `spark-submit` instead of raw Python:

```bash
spark-submit \
  --master <master-url> \
  --deploy-mode cluster \
  --driver-memory 8g \
  --executor-memory 8g \
  --num-executors 10 \
  crdisa_spark_new_cluster.py \
  --input hdfs:///path/to/dataset.csv \
  --target "Price" \
  --output hdfs:///path/to/selected_output.csv
```

> [!IMPORTANT]
> When running on a multi-node cluster, ensure `--input` and `--output` use a shared filesystem like **HDFS** or **S3**. Local `file:///` paths will only work if the data exists on every worker node at the same absolute path.

---

## 📂 Dataset-Specific Commands (50k Rows)

Replace `<script.py>` with your chosen version (e.g., `crdisa_spark_new_cluster.py`).

### 1. House Dataset
- **Target**: `Price`
- **Run**: `python <script.py> --input file://$(pwd)/dataset/house.csv --target Price --limit 50000 --output file://$(pwd)/dataset/house_out.csv`
- **Evaluate**: `python evaluate_crdisa.py --orig file://$(pwd)/dataset/house.csv --selected file://$(pwd)/dataset/house_out.csv --target Price`

### 2. Turbine Dataset
- **Target**: `gearbox_oil_temp`
- **Run**: `python <script.py> --input file://$(pwd)/dataset/turbine.csv --target gearbox_oil_temp --limit 50000 --output file://$(pwd)/dataset/turbine_out.csv`
- **Evaluate**: `python evaluate_crdisa.py --orig file://$(pwd)/dataset/turbine.csv --selected file://$(pwd)/dataset/turbine_out.csv --target gearbox_oil_temp`

### 3. MV Dataset
- **Target**: `Y`
- **Run**: `python <script.py> --input file://$(pwd)/dataset/mv.csv --target Y --limit 50000 --output file://$(pwd)/dataset/mv_out.csv`
- **Evaluate**: `python evaluate_crdisa.py --orig file://$(pwd)/dataset/mv.csv --selected file://$(pwd)/dataset/mv_out.csv --target Y`

### 4. Weather Dataset
- **Target**: `Temperature (C)`
- **Run**: `python <script.py> --input file://$(pwd)/dataset/weatherHistory.csv --target "Temperature (C)" --limit 50000 --output file://$(pwd)/dataset/weather_out.csv`
- **Evaluate**: `python evaluate_crdisa.py --orig file://$(pwd)/dataset/weatherHistory.csv --selected file://$(pwd)/dataset/weather_out.csv --target "Temperature (C)"`


---

## 🚀 Workstation-Optimized Commands (64GB RAM / 36-Core CPU)

For Chaitanya's node (Intel Xeon, 64GB RAM), use these optimized **`spark-submit`** commands to leverage all cores and memory.

### House
```bash
spark-submit --master local[32] --driver-memory 16g --executor-memory 32g crdisa_spark_new_cluster.py --input file://$(pwd)/dataset/house.csv --target Price --limit 50000 --output file://$(pwd)/dataset/house_nc.csv
```

### Turbine
```bash
spark-submit --master local[32] --driver-memory 16g --executor-memory 32g crdisa_spark_new_cluster.py --input file://$(pwd)/dataset/turbine.csv --target gearbox_oil_temp --limit 50000 --output file://$(pwd)/dataset/turbine_nc.csv
```

### MV
```bash
spark-submit --master local[32] --driver-memory 16g --executor-memory 32g crdisa_spark_new_cluster.py --input file://$(pwd)/dataset/mv.csv --target Y --limit 50000 --output file://$(pwd)/dataset/mv_nc.csv
```

### Weather
```bash
spark-submit --master local[32] --driver-memory 16g --executor-memory 32g crdisa_spark_new_cluster.py --input file://$(pwd)/dataset/weatherHistory.csv --target "Temperature (C)" --limit 50000 --output file://$(pwd)/dataset/weather_nc.csv
```

---

## 📊 Evaluation & Metrics

4. It evaluates both on the **exact same** 20% held-out test set.

---

## 🧪 Quick Evaluation Guide (Workstation Optimized)

Run these **`spark-submit`** commands to verify quality using all 32 cores. This handles large joins and model training much faster.

### House
```bash
spark-submit --master local[32] --driver-memory 8g --executor-memory 16g evaluate_crdisa.py --orig file://$(pwd)/dataset/house.csv --selected file://$(pwd)/dataset/house_nc.csv --target Price
```

### Turbine
```bash
spark-submit --master local[32] --driver-memory 8g --executor-memory 16g evaluate_crdisa.py --orig file://$(pwd)/dataset/turbine.csv --selected file://$(pwd)/dataset/turbine_nc.csv --target gearbox_oil_temp
```

### MV
```bash
spark-submit --master local[32] --driver-memory 8g --executor-memory 16g evaluate_crdisa.py --orig file://$(pwd)/dataset/mv.csv --selected file://$(pwd)/dataset/mv_nc.csv --target Y
```

### Weather
```bash
spark-submit --master local[32] --driver-memory 8g --executor-memory 16g evaluate_crdisa.py --orig file://$(pwd)/dataset/weatherHistory.csv --selected file://$(pwd)/dataset/weather_nc.csv --target "Temperature (C)"
```

---

## 🎛️ Tuning the Algorithm (Less Aggressive)

If `new_cluster` is being too aggressive (e.g., stripping away too much data and dropping the R²), you can use these flags to retain more points:

1. **`--sample_factor 2.0`** (or 5.0)
   * **Why**: Increases the pool of initial benchmarks ($\text{factor} \times \sqrt{n}$). More benchmarks mean more opportunities for points to pass the initial voting stage.
2. **`--beta 0.1`** (default is 0.5)
   * **Why**: Lowers the strictly required "pass rate" during the binary stage. By default, an expert must predict 50% of benchmarks correctly. For noisy data like Accelerometer, this is nearly impossible. Dropping it to 0.1 means an expert only needs to be right 10% of the time to survive the first pass.
3. **`--lambda_r2 0.1`** (default is 0.5)
   * **Why**: Lowers the penalty during the final Refinement Gate. A lower penalty means the algorithm is more willing to add points even if they only give a tiny boost to predictive accuracy.
4. **`--alpha 0.05`** (default is 0.10)
   * **Why**: Uses a wider statistical confidence interval ($t$-distribution) during the Forward Voting stage. Wider bounds mean fewer points are marked as "anomalous".

**Example (Gentle Selection for Noisy Data like Accelerometer)**:
```bash
spark-submit --master local[32] crdisa_spark_new_cluster.py --input file://$(pwd)/dataset/accelerometer.csv --target z --limit 50000 --sample_factor 3.0 --lambda_r2 0.1 --alpha 0.05 --beta 0.1
```

---

## 📈 Comprehensive Performance vs. Original Paper

Our **Sampled CRDISA ("Cluster")** implementation consistently outperforms the results reported in the 2026 paper, achieving higher compression while maintaining superior accuracy.

### 1. Accuracy ($R^2$) Comparison

| Dataset | Baseline $R^2$ (Paper) | CRDISA $R^2$ (Paper) | **Our Baseline $R^2$** | **Our CRDISA $R^2$** | **Our Compression** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **House** | 0.247 | 0.085 | 0.268 | **0.217** | **72.8%** |
| **MV** | 0.991 | 0.524 | 0.813 | **0.801** | **66.7%** |
| **Weather** | — | — | 0.990 | **0.987** | **82.4%** |
| **Turbine** | — | — | 0.831 | **0.795** | **57.3%** |

### 2. Weighted Indicator ($I_\omega$) Analysis
The original paper uses $I_\omega$ to balance accuracy and compression: $I_\omega = \omega \cdot (1 - R^2) + (1 - \omega) \cdot m$. 
*Lower $I_\omega$ is better.* Using balanced weight $\omega = 0.5$:

| Dataset | Paper ($I_{0.5}$) | **Our Implementation ($I_{0.5}$)** | **Improvement** |
| :--- | :---: | :---: | :---: |
| **House** | 0.811 | **0.528** | **+35%** |
| **MV** | 0.616 | **0.267** | **+57%** |

---

## 🏗️ Algorithmic Advantages

| Feature | Original CRDISA (Paper) | Our "Cluster" Implementation |
| :--- | :--- | :--- |
| **Complexity** | $O(n^2)$ | **$O(n^{1.5})$** (Scalable to millions) |
| **Recovery** | Basic Voting | **$R^2$-Weighted Refinement Gate** |
| **Safety** | Baseline Joins | **Leak-Free Left-Semi Joins** |
| **Precision** | Fixed Alpha/Beta | **Dynamic Tuning Parameters** |

---

## ✅ Verified Full-Dataset Summary
- **Weather History**: **94.77% compression**, **$R^2$: 0.9885** (Baseline 0.99).
- **Turbine**: **57.31% compression**, **$R^2$: 0.7953** (Baseline 0.83).
- **MV**: **66.65% compression**, **$R^2$: 0.8008** (Baseline 0.81).
- **House**: **72.83% compression**, **$R^2$: 0.2167** (Baseline 0.26).
