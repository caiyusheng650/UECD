# -*- coding: utf-8 -*-
"""
Build a small training subset for a quick overfitting sanity check.
Reuses the real train.py training loop (no copied training logic).

Run normal training with:
    set KEEP=... (optional)
    python build_overfit_small.py
    <set UECD_TRAIN_FILE=data/overfit_small.json, run train.py as usual>
"""
import json
import os
import random

SRC = 'data/neg_exer_ours.json'
OUT = 'data/overfit_small.json'
VAL_SRC = 'data/Eedi/val_set.json'
VAL_OUT = 'data/overfit_val_small.json'
# keep >= one full batch (256) so the loader yields >0 batches each epoch
N = int(os.environ.get('N', '1024'))
# val groups needed just for the sanity check; a few hundred is plenty
VAL_N = int(os.environ.get('VAL_N', '200'))
# alternative: take a FRACTION of the full data, e.g. set FRAC=0.1 for 10%
FRAC = float(os.environ.get('FRAC', '0'))
VAL_FRAC = float(os.environ.get('VAL_FRAC', '0'))

random.seed(0)
print('loading %s ...' % SRC, flush=True)
with open(SRC, encoding='utf8') as f:
    data = json.load(f)
if FRAC > 0:
    N = max(1, int(FRAC * len(data)))
    print('FRAC=%.2f -> N=%d' % (FRAC, N))
random.shuffle(data)
sub = data[:N]
with open(OUT, 'w', encoding='utf8') as f:
    json.dump(sub, f, separators=(',', ':'))

print('%d records -> %s' % (len(sub), OUT))
print('unique users: %d' % len({x['user_id'] for x in sub}))
print('unique exers: %d' % len({x['exer_id'] for x in sub}))

print('building tiny val subset %s ...' % VAL_OUT, flush=True)
with open(VAL_SRC, encoding='utf8') as f:
    val_data = json.load(f)
if VAL_FRAC > 0:
    VAL_N = max(1, int(VAL_FRAC * len(val_data)))
    print('VAL_FRAC=%.2f -> VAL_N=%d' % (VAL_FRAC, VAL_N))
random.shuffle(val_data)
val_sub = val_data[:VAL_N]
with open(VAL_OUT, 'w', encoding='utf8') as f:
    json.dump(val_sub, f)
print('%d val groups -> %s' % (len(val_sub), VAL_OUT))