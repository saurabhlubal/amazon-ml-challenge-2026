"""
Micro-benchmark comparing feature extraction implementations:
1. NumPy/Python current implementation (extract_features_vectorized)
2. Polars DataFrame vectorized calculation
3. DuckDB SQL vectorized calculation

Tests across 10,000 candidate pairs.
"""

import time
import duckdb
import polars as pl
import numpy as np

# Create synthetic sample pairs representative of BER dataset
n_pairs = 10000

s1_names = ["state bank of india", "amazon web services inc", "tata consultancy services", "walmart supercenter"] * (n_pairs // 4)
c_names = ["state bank india", "amazon aws services", "tata consultancy", "target store"] * (n_pairs // 4)

s1_toks = [["state", "bank", "india"], ["amazon", "web", "services", "inc"], ["tata", "consultancy", "services"], ["walmart", "supercenter"]] * (n_pairs // 4)
c_toks = [["state", "bank", "india"], ["amazon", "aws", "services"], ["tata", "consultancy"], ["target", "store"]] * (n_pairs // 4)

s1_addrs = ["123 main street ny", "410 terry ave n seattle wa", "bandra kurla complex mumbai", "789 broadway st"] * (n_pairs // 4)
c_addrs = ["123 main st new york", "410 terry avenue seattle", "bkc mumbai", "789 broadway street"] * (n_pairs // 4)

s1_countries = ["US", "US", "INDIA", "US"] * (n_pairs // 4)
c_countries = ["US", "US", "INDIA", "US"] * (n_pairs // 4)

# 1. Benchmark NumPy / Python current approach
def fast_jaccard(a, b):
    if not a or not b: return 0.0
    inter = len(a & b)
    if inter == 0: return 0.0
    return inter / (len(a) + len(b) - inter)

t0 = time.time()
s1_tok_sets = [set(t) for t in s1_toks]
c_tok_sets = [set(t) for t in c_toks]

X = np.zeros((n_pairs, 13), dtype=np.float32)
for i in range(n_pairs):
    X[i, 0] = 1.0 if s1_names[i] == c_names[i] else 0.0
    X[i, 1] = 1.0 if s1_names[i] == c_names[i] else 0.0
    X[i, 3] = fast_jaccard(s1_tok_sets[i], c_tok_sets[i])
    X[i, 6] = abs(len(s1_names[i]) - len(c_names[i])) / max(len(s1_names[i]), len(c_names[i]), 1)
    X[i, 7] = 1.0 if not s1_addrs[i] or not c_addrs[i] else 0.0
    X[i, 8] = 1.0 if s1_addrs[i] == c_addrs[i] else 0.0
    X[i, 12] = 1.0 if s1_countries[i] == c_countries[i] else 0.0

time_numpy = time.time() - t0
print(f"NumPy/Python Current: {time_numpy:.4f}s ({n_pairs/time_numpy:,.0f} pairs/sec)")

# 2. Benchmark Polars Vectorized
t0 = time.time()
df = pl.DataFrame({
    "s1_name": s1_names,
    "c_name": c_names,
    "s1_toks": s1_toks,
    "c_toks": c_toks,
    "s1_addr": s1_addrs,
    "c_addr": c_addrs,
    "s1_country": s1_countries,
    "c_country": c_countries,
})

df_res = df.select([
    (pl.col("s1_name") == pl.col("c_name")).cast(pl.Float32).alias("name_exact_raw"),
    (pl.col("s1_name") == pl.col("c_name")).cast(pl.Float32).alias("name_exact_norm"),
    (
        pl.col("s1_toks").list.set_intersection(pl.col("c_toks")).list.len() /
        (pl.col("s1_toks").list.len() + pl.col("c_toks").list.len() - pl.col("s1_toks").list.set_intersection(pl.col("c_toks")).list.len())
    ).fill_nan(0.0).alias("name_jaccard"),
    ((pl.col("s1_name").str.len_chars() - pl.col("c_name").str.len_chars()).abs() / pl.max_horizontal(pl.col("s1_name").str.len_chars(), pl.col("c_name").str.len_chars(), 1)).alias("name_len_diff"),
    ((pl.col("s1_addr") == "") | (pl.col("c_addr") == "")).cast(pl.Float32).alias("addr_is_empty"),
    (pl.col("s1_addr") == pl.col("c_addr")).cast(pl.Float32).alias("addr_exact_norm"),
    (pl.col("s1_country") == pl.col("c_country")).cast(pl.Float32).alias("country_match"),
])

time_polars = time.time() - t0
print(f"Polars Vectorized:   {time_polars:.4f}s ({n_pairs/time_polars:,.0f} pairs/sec)")

# 3. Benchmark DuckDB
t0 = time.time()
con = duckdb.connect()
con.register("df_pairs", df)
duck_res = con.execute("""
    SELECT
        (s1_name = c_name)::FLOAT as name_exact_raw,
        (s1_name = c_name)::FLOAT as name_exact_norm,
        abs(length(s1_name) - length(c_name))::FLOAT / greatest(length(s1_name), length(c_name), 1) as name_len_diff,
        (s1_addr = '' or c_addr = '')::FLOAT as addr_is_empty,
        (s1_addr = c_addr)::FLOAT as addr_exact_norm,
        (s1_country = c_country)::FLOAT as country_match
    FROM df_pairs
""").fetchall()
time_duckdb = time.time() - t0
print(f"DuckDB SQL:          {time_duckdb:.4f}s ({n_pairs/time_duckdb:,.0f} pairs/sec)")
