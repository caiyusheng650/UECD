# -*- coding: utf-8 -*-
"""
Reproduce data/neg_exer_ours.json (UECD stage-1 cluster-based coarse sampling).

Faithful re-implementation of the offline generator that is kept commented in
data_loader.py (next_batch, lines ~96-143) together with the helper logic of
TrainDataLoader.get_matching_exer / get_same_kn_exer_ids:

  1. students are grouped by the 50 clusters in data/Eedi/result50.json
     (produced by cluster.py);
  2. for every training log (u, e, kc) collect exercises that were answered by
     OTHER students in u's cluster, were NOT answered by u, and whose knowledge
     codes are EXACTLY kc;
  3. rank them by frequency among the cluster peers (Counter.most_common);
     - no candidate at all  -> pad with e repeated topk times
     - fewer than topk      -> fill with random.choices(..., replace=True)
     - topk or more         -> keep the whole ranked pool (stage-2 in
       model.item_similarity resamples topk from it);
  4. ids are written 0-based for user_id/exer_id (TrainDataLoader feeds them
     straight into nn.Embedding indexed 0..N-1), knowledge_code stays 1-based.

Usage:
    python build_neg_exer.py
"""

import json
import os
import random
import time
from collections import Counter, defaultdict

from tqdm import tqdm

TRAIN_FILE = 'data/Eedi/train_set.json'
CLUSTER_FILE = 'data/Eedi/result50.json'
OUT_FILE = 'data/neg_exer_ours.json'
TOPK = 20
SEED = 2024


def main():
    random.seed(SEED)

    print('loading %s ...' % CLUSTER_FILE, flush=True)
    with open(CLUSTER_FILE, encoding='utf8') as f:
        clusters_raw = json.load(f)
    # cid -> set of 0-based user ids
    cluster_members = {int(cid): set(members)
                       for cid, members in clusters_raw.items()}
    user_cluster = {}
    for cid, members in cluster_members.items():
        for u in members:
            user_cluster[u] = cid
    print('%d clusters, %d users' % (len(cluster_members), len(user_cluster)),
          flush=True)

    print('loading %s ...' % TRAIN_FILE, flush=True)
    with open(TRAIN_FILE, encoding='utf8') as f:
        data = json.load(f)
    print('%d training records' % len(data), flush=True)

    # exercises each user interacted with (0-based)
    user_interacted = defaultdict(set)
    # exercise -> sorted knowledge-code tuple (1-based), taken from the logs
    exer_concepts = {}
    for log in tqdm(data, desc='indexing users/exercises'):
        u = log['user_id'] - 1
        e = log['exer_id'] - 1
        kc = tuple(sorted(int(x) for x in log['knowledge_code']))
        user_interacted[u].add(e)
        exer_concepts[e] = kc

    # per-cluster exercise frequency, record order preserved so that
    # Counter.most_common tie-breaking matches the author implementation
    cluster_exer_count = defaultdict(Counter)
    for log in tqdm(data, desc='counting cluster exercises'):
        u = log['user_id'] - 1
        e = log['exer_id'] - 1
        cluster_exer_count[user_cluster[u]][e] += 1

    # (cluster, concept tuple) -> exercises ranked by peer frequency desc
    cluster_concept_ranked = defaultdict(list)
    ranked_lookup = {}
    for cid, counter in cluster_exer_count.items():
        buckets = defaultdict(list)
        for e, _cnt in counter.most_common():
            buckets[exer_concepts[e]].append(e)
        for kc, ranked in buckets.items():
            ranked_lookup[(cid, kc)] = ranked

    out = []
    empty_pools = 0
    padded_pools = 0
    t0 = time.perf_counter()
    for log in tqdm(data, desc='building negative pools'):
        u = log['user_id'] - 1
        e = log['exer_id'] - 1
        kc_l = [int(x) for x in log['knowledge_code']]
        kc = tuple(sorted(kc_l))
        cid = user_cluster[u]
        own = user_interacted[u]
        ranked = ranked_lookup.get((cid, kc), ())
        pool = [x for x in ranked if x not in own]

        if len(pool) == 0:
            pool = [e] * TOPK
            empty_pools += 1
        elif len(pool) < TOPK:
            pool = pool + random.choices(pool, k=TOPK - len(pool))
            padded_pools += 1

        out.append({
            'user_id': u,
            'exer_id': e,
            'score': log['score'],
            'knowledge_code': kc_l,
            'neg_exer_ids': pool,
        })
    print('pools: %d empty(filled with current exer), %d short(padded)'
          % (empty_pools, padded_pools), flush=True)
    print('building took %.1f s' % (time.perf_counter() - t0), flush=True)

    print('writing %s ...' % OUT_FILE, flush=True)
    t0 = time.perf_counter()
    tmp_file = OUT_FILE + '.tmp'
    with open(tmp_file, 'w', encoding='utf8') as f:
        json.dump(out, f, separators=(',', ':'))
    os.replace(tmp_file, OUT_FILE)
    print('wrote %d records in %.1f s' % (len(out), time.perf_counter() - t0),
          flush=True)


if __name__ == '__main__':
    main()
