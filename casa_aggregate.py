"""Usage: python casa_aggregate.py <raw_trace.csv[.gz]> data/minute_<name>.csv

Aggregate raw Azure LLM inference traces (2024, 1 week) into per-minute
prefill (context-token) and decode (generated-token) demand series."""
import sys, pandas as pd
src, dst = sys.argv[1], sys.argv[2]
parts = []
for ch in pd.read_csv(src, chunksize=4_000_000):
    ts = pd.to_datetime(ch["TIMESTAMP"], utc=True, format="ISO8601").dt.floor("min")
    g = ch.assign(m=ts).groupby("m").agg(req=("ContextTokens", "size"),
                                         ctx=("ContextTokens", "sum"),
                                         gen=("GeneratedTokens", "sum"))
    parts.append(g)
df = pd.concat(parts).groupby(level=0).sum().asfreq("min", fill_value=0)
df.to_csv(dst)
print(dst, df.shape, df.index.min(), df.index.max())
print(df.describe().T)
