import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

def raw(text):
    c = nbf.v4.new_raw_cell(text)
    c['metadata'] = {"raw_mimetype": "text/x-sh"}
    cells.append(c)

md("""# Distributed Analytics Engineering with PySpark
**Course:** BSDS — Distributed Analytics Engineering with PySpark
**Objective:** Configure an end-to-end distributed data pipeline on Ubuntu/Hadoop, execute data transformations using dual programming paradigms (PySpark API vs. SparkSQL), and analyze distributed execution behavior and performance.
""")

md("## Part 1: Cluster Pre-Flight & HDFS Ingestion\n\nCommands and expected output are documented below; you already ran the real versions of these in your terminal.")

raw("""# --- Already run in your Codespace terminal ---
sudo service ssh start
start-dfs.sh
jps
# Your actual output: NameNode, DataNode, SecondaryNameNode, Jps
""")

raw("""# --- Already run in your Codespace terminal ---
hdfs dfs -mkdir -p /user/$USER/analytics_input
hdfs dfs -put retail_sales.csv /user/$USER/analytics_input/
hdfs dfs -ls /user/$USER/analytics_input/
# Your actual output: Found 1 items, -rw-r--r-- 1 codespace supergroup 287 ... retail_sales.csv
""")

md("""**Pre-Flight Question 1.1** — Provide the output of `jps`. Which three core Hadoop daemons are running, and what specific role does each play in cluster storage management?

- **NameNode** — the HDFS master. Holds the filesystem namespace and metadata (directory tree, file-to-block mapping, permissions) in memory and coordinates every read/write, but never stores block data itself.
- **DataNode** — the HDFS worker. Stores the actual data blocks on local disk and serves read/write requests directly to clients.
- **SecondaryNameNode** — not a hot standby. Periodically merges the NameNode's edit log into a new fsimage checkpoint, bounding edit-log size and speeding up NameNode restarts.

**Pre-Flight Question 1.2** — How does the command prefix `hdfs dfs -` differentiate file operations targeting the distributed storage layer from native Linux filesystem operations?

`hdfs dfs -<cmd>` routes through Hadoop's `FileSystem` abstraction layer rather than OS syscalls. The URI scheme resolved by that layer (defaulting to whatever `fs.defaultFS` is set to in `core-site.xml`) tells Hadoop to talk to the NameNode over RPC and stream blocks to/from DataNodes, instead of touching the local filesystem directly.""")

md("## Part 2: PySpark Engine Initialization & Setup")

code("""from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum, avg, max, min

spark = (
    SparkSession.builder
    .appName("DualMethod_Analytics_Lab")
    .master("local[*]")
    .getOrCreate()
)

spark
""")

md("""**Setup Question 2.1** — What is the technical function of `py4j` during SparkSession startup, and why is an active JVM instance required even if you are executing purely Python code?

Spark's execution engine is written in Scala/JVM. `py4j` bridges Python calls to that JVM over a socket, translating method calls and marshalling return values. A JVM must be running because the DataFrame/SQL engine, memory management, and shuffle logic all live exclusively on the JVM side — Python is only ever the driver-side control language.""")

md("""## Part 3: Multi-Format Ingestion Strategy

Reading from HDFS directly, since the file is staged there.""")

code("""df_csv = spark.read.csv("hdfs://localhost:9000/user/codespace/analytics_input/retail_sales.csv", header=True, inferSchema=True)
df_csv.show()
df_csv.printSchema()
""")

code("""import pandas as pd

pd_df = pd.DataFrame({
    "StoreCode": ["KHI-01", "LHR-01", "ISB-01", "PEW-01"],
    "City": ["Karachi", "Lahore", "Islamabad", "Peshawar"],
    "RegionalManager": ["Tariq", "Saima", "Rashid", "Nadia"],
})
pd_df.to_excel("store_registry.xlsx", index=False)

pd_df_read_back = pd.read_excel("store_registry.xlsx")
df_excel = spark.createDataFrame(pd_df_read_back)
df_excel.show()
""")

md("""**Ingestion Question 3.1** — What does `inferSchema=True` force Spark to execute under the hood on a CSV file? What impact does this have on multi-terabyte files in production?

It triggers a full extra pass over the file before the real read, sniffing each column's type. That doubles I/O on a multi-terabyte file and adds a whole extra distributed job purely for type inference. Production practice is to define an explicit `StructType` schema up front, or use Parquet, which stores its schema in the file footer.

**Ingestion Question 3.2** — Why can PySpark read formats like Parquet, ORC, or CSV out-of-the-box, but requires bridging through Pandas for `.xlsx` spreadsheets? What memory risk does this pose when processing large spreadsheets?

Parquet/ORC/CSV are splittable, so Spark partitions them across the cluster. `.xlsx` is a zipped binary format with no splittable structure, so Pandas/openpyxl must parse the whole workbook in one shot on the driver before Spark ever sees it. A large spreadsheet can OOM the driver even though an equivalent Parquet file would be handled comfortably.""")

md("## Part 4: Dual-Method Analytics Challenges")

md("### Challenge 1 — Schema Structure Inspection")
code("""df_csv.printSchema()
""")
code("""print(df_csv.dtypes)
""")
md("""**Critical Analysis 1.1**""")
code("""captured = df_csv.printSchema()
print(type(captured), captured)
""")
md("""`printSchema()` prints and returns `None` — nothing usable to assign. `df.dtypes` returns a real Python `list` of `(name, type)` tuples, usable in automated validation.""")

md("### Challenge 2 — Summary & Descriptive Statistics")
code("""df_csv.describe().show()
""")
code("""df_csv.select("Amount").summary("count", "mean", "stddev", "min", "25%", "50%", "75%", "max").show()
""")
md("""**Critical Analysis 2.1** — The median is more reliable against outliers: the mean is pulled up by the two large Electronics transactions, while the median only depends on rank order, not magnitude.""")

md("### Challenge 3 — Unique Value Identification")
code("""distinct_count = df_csv.select("CustomerName").distinct().count()
print("Distinct customers (Method A):", distinct_count)
""")
code("""dedup_df = df_csv.dropDuplicates(["CustomerName"])
print("Distinct customers (Method B):", dedup_df.count())
dedup_df.show()
""")
md("""**Critical Analysis 3.1** — `dropDuplicates()` retains the full row schema; `select().distinct()` discards every other column. For persisting full profiles, `dropDuplicates()` is required.""")

md("### Challenge 4 — Category Distribution Analysis")
code("""method_a = df_csv.groupBy("Category").count().sort(col("count").desc())
method_a.show()
""")
code("""df_csv.createOrReplaceTempView("sales_view")
method_b = spark.sql(\"\"\"
    SELECT Category, COUNT(*) AS count
    FROM sales_view
    GROUP BY Category
    ORDER BY count DESC
\"\"\")
method_b.show()
""")
code("""method_a.explain(True)
""")
code("""method_b.explain(True)
""")
md("""**Critical Analysis 4.1** — Identical physical plans: both compile to the same `LogicalPlan`, so Catalyst produces the same `HashAggregate` → `Exchange` → `Sort` shape regardless of front-end syntax.""")

md("### Challenge 5 — Conditional Filtering")
code("""method_a5 = df_csv.filter((col("Amount") >= 1000) & (col("City") == "Karachi"))
method_a5.show()
""")
code("""method_b5 = df_csv.where("Amount >= 1000 AND City = 'Karachi'")
method_b5.show()
""")
md("""**Critical Analysis 5.1**""")
code("""try:
    bad = df_csv.filter((col("Amount") >= 1000) and (col("City") == "Karachi"))
    bad.show()
except Exception as e:
    print(type(e).__name__ + ":", e)
""")
md("""`and`/`or` call `__bool__()`, but a `Column` is an unevaluated distributed expression with no single truth value, so `Column.__bool__` is overridden to raise. `&`/`|`/`~` build a deferred boolean expression node instead.""")

md("### Challenge 6 — Multi-Metric Aggregation")
code("""method_a6 = df_csv.groupBy("Category").agg(
    sum("Amount").alias("Total_Revenue"),
    avg("Amount").alias("Average_Spend"),
)
method_a6.show()
""")
code("""agg_dict = {"Amount": "sum", "Amount": "avg"}
print("Dictionary actually passed to agg():", agg_dict)

method_b6 = df_csv.groupBy("Category").agg(agg_dict)
method_b6.show()
""")
md("""**Critical Analysis 6.1** — A Python dict can only hold one value per key, so the second assignment silently overwrites the first; `agg_dict` only ever contains `{"Amount": "avg"}`, so Method B silently drops the sum.""")

md("## Part 5: Big Data Visual Analytics")

code("""import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

cat_agg = (
    df_csv.groupBy("Category")
    .agg(sum("Amount").alias("Total_Revenue"), count("*").alias("Transaction_Count"))
    .orderBy(col("Total_Revenue").desc())
)
cat_pd = cat_agg.toPandas()
cat_pd
""")

code("""fig, ax1 = plt.subplots(figsize=(8, 5))

bars = ax1.bar(cat_pd["Category"], cat_pd["Total_Revenue"], color="#4C72B0", label="Total Revenue")
ax1.set_xlabel("Category")
ax1.set_ylabel("Total Revenue (PKR)", color="#4C72B0")
ax1.tick_params(axis="y", labelcolor="#4C72B0")

top_rev_idx = cat_pd["Total_Revenue"].idxmax()
bars[top_rev_idx].set_color("#C44E52")

ax2 = ax1.twinx()
ax2.plot(cat_pd["Category"], cat_pd["Transaction_Count"], color="#55A868", marker="o", linewidth=2)
ax2.set_ylabel("Transaction Count", color="#55A868")
ax2.tick_params(axis="y", labelcolor="#55A868")

top_vol_idx = cat_pd["Transaction_Count"].idxmax()
ax2.annotate(f"Highest volume:\\n{cat_pd['Category'][top_vol_idx]}",
    xy=(top_vol_idx, cat_pd["Transaction_Count"][top_vol_idx]),
    xytext=(0, 20), textcoords="offset points", ha="center", fontsize=9, color="#55A868")
ax1.annotate(f"Highest revenue:\\n{cat_pd['Category'][top_rev_idx]}",
    xy=(top_rev_idx, cat_pd["Total_Revenue"][top_rev_idx]),
    xytext=(0, 12), textcoords="offset points", ha="center", fontsize=9, color="#C44E52")

plt.title("Category Revenue & Transaction Volume")
fig.tight_layout()
plt.savefig("viz1_category_revenue_volume.png", dpi=150)
plt.show()
""")

code("""city_pd = df_csv.select("City", "Amount").toPandas()

plt.figure(figsize=(8, 5))
ax = sns.boxplot(data=city_pd, x="City", y="Amount", color="#8172B2")

medians = city_pd.groupby("City")["Amount"].median()
for i, city in enumerate(city_pd["City"].unique()):
    if city in medians.index:
        ax.annotate(f"median={medians[city]:.0f}", xy=(i, medians[city]),
                    xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)

plt.title("Spending Distribution by City (Amount)")
plt.tight_layout()
plt.savefig("viz2_spending_distribution.png", dpi=150)
plt.show()
""")

md("""**Analytics Prompt 5.1** — `.toPandas()` pulls every row into the driver's memory alone, with no partitioning. On a 1-billion-row DataFrame this would exceed the driver's heap and crash the whole Spark application, not just plotting. Always aggregate inside Spark first.""")

md("""## Part 6: Synthesis & Final Analysis

### Deliverable A: Comparison Matrix

| Objective Area | Method A | Method B | Recommendation |
|---|---|---|---|
| Schema Check | `printSchema()` | `dtypes` | `dtypes` returns a real list; use it for automated validation. |
| Statistical Summary | `describe()` | `summary()` | `summary()` allows exact percentiles. |
| Unique Records | `select().distinct()` | `dropDuplicates()` | `dropDuplicates()` keeps full rows. |
| Value Counts | DataFrame API | SparkSQL | Same physical plan either way. |
| Data Filtering | `col()` expressions | SQL string | `col()` is type-checked; prefer in production. |
| Multi-Aggregation | alias-based `.agg()` | dict-based `.agg()` | Dict form silently drops duplicate keys — unsafe. |

### Deliverable B: Technical Synthesis

Transformations are lazy; actions trigger execution. This lets Catalyst see the whole pipeline as one logical plan before any cluster resources are spent, enabling predicate pushdown, column pruning, join reordering, and physical-strategy selection — none of which would be possible if each transformation executed eagerly in isolation.""")

nb['cells'] = cells
nbf.write(nb, "Distributed_Analytics_PySpark_Lab.ipynb")
print("Notebook written with", len(cells), "cells")
