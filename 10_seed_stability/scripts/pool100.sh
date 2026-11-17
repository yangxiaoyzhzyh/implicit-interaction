#!/bin/bash
# Pool of 100 molecules only; appearances per molecule 2..8; five seeds each.
cd /root/pool_scan || exit 1
mkdir -p out100
rm -f out100/*.csv

printf '%s\n' 2 3 4 5 6 7 8 | xargs -P 7 -I{} bash -c '
  OMP_NUM_THREADS=2 python pool_scan.py --nmol 100 --appear {} \
    --seeds 42 43 44 45 46 --out out100/r{}.csv
' 2>&1 | grep -v Warning

echo "=== merged ==="
echo "nmol,appear,n_train,train_r2,r2_k3,r2_k10,r2_k20,flip_pct,n_nontie,cos_median" > pooled100.csv
cat out100/*.csv | grep -v '^nmol' | sort -t, -k2,2n >> pooled100.csv
cat pooled100.csv
