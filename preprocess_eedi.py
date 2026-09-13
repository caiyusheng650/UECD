# -*- coding: utf-8 -*-
"""
Eedi (NeurIPS 2020 Education Challenge, task 1&2) -> UECD/NCDM json preprocessing.

  python preprocess_eedi.py stats   # one streaming load, evaluate filter grid
  python preprocess_eedi.py build   # produce data/Eedi/{train,val,test}_set.json

Author-style recipe (reverse-engineered from the data to match the paper
config 17,740 students / 8,987 exercises / 286 concepts; see README note
printed by build):
  1. Q-matrix : SubjectId path per question, root subject 3 (Mathematics)
                removed; every remaining subject becomes a knowledge concept.
  2. Items    : keep questions answered by >= ITEM_POP answers (calibrated to
                ~8,987 exercises; actual numbers reported at build time).
  3. Students : keep users with [USER_LO, USER_HI] retained logs (light/medium
                users, matching the ~52 logs/student profile of the paper).
  4. Subsample: fixed-seed sample of at most USER_SUBSAMPLE students.
  5. Split    : per student 80% train / 10% val / 10% test; train is a flat
                list, val/test grouped per student -- exactly what data_loader
                expects. All ids are 1-based and contiguous.

Memory: the 410MB answer log is parsed once with stdlib csv into compact int
arrays (~140MB). pandas is NOT required.
"""
import csv
import json
import os
import sys
from array import array

import numpy as np
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'data')
ANSWER_CSV = os.path.join(DATA_DIR, 'train_task_1_2.csv')
QUESTION_CSV = os.path.join(DATA_DIR, 'question_metadata_task_1_2.csv')
OUT_DIR = os.path.join(DATA_DIR, 'Eedi')

CONCEPT_ROOT = 3          # SubjectId 3 = "Mathematics", present in every path
SEED = 2024
ITEM_POP = 625            # questions need >= 625 total answers
USER_LO, USER_HI = 20, 100
USER_SUBSAMPLE = 17740    # None -> keep every eligible student


def load_question_concepts():
    """Raw QuestionId -> sorted list of SubjectId excluding the root."""
    q2c = {}
    with open(QUESTION_CSV, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            qid = int(row[0])
            subjects = json.loads(row[1])
            q2c[qid] = sorted({int(s) for s in subjects
                               if int(s) != CONCEPT_ROOT})
    return q2c


def load_answers():
    """Stream the giant csv into compact arrays, relabel ids contiguously."""
    au, aq, ay = array('i'), array('i'), array('b')
    total = 0
    file_size = os.path.getsize(ANSWER_CSV)
    # byte-accurate progress: iterate raw binary lines, count bytes, decode.
    # (tqdm.wrapattr's callback object is not iterable for csv.reader, and
    # TextIOWrapper drives the underlying buffer via read1 which bypasses it.)
    with open(ANSWER_CSV, 'rb') as f_bin, \
            tqdm(total=file_size, desc='parsing answer csv',
                 unit='B', unit_scale=True, unit_divisor=1024) as pbar:
        def _lines():
            for raw in f_bin:
                pbar.update(len(raw))
                yield raw.decode('utf-8')

        reader = csv.reader(_lines())
        next(reader)
        for row in reader:
            try:
                qid, uid, y = int(row[0]), int(row[1]), int(row[3])
            except (ValueError, IndexError):
                continue
            if y not in (0, 1):
                continue
            au.append(uid)
            aq.append(qid)
            ay.append(y)
            total += 1

    u_raw = np.asarray(au, dtype=np.int64)
    q_raw = np.asarray(aq, dtype=np.int64)
    y = np.asarray(ay, dtype=np.int8)
    del au, aq, ay

    print('relabeling ids (numpy unique) ...', flush=True)
    raw_uids, u = np.unique(u_raw, return_inverse=True)
    raw_qids, q = np.unique(q_raw, return_inverse=True)
    del u_raw, q_raw

    # dedupe repeated attempts on the same (user, question): first wins
    print('deduplicating repeated attempts ...', flush=True)
    key = u.astype(np.int64) * (len(raw_qids) + 1) + q
    _, first_idx = np.unique(key, return_index=True)
    order = np.sort(first_idx)
    n_dup = len(y) - len(order)
    u, q, y = u[order], q[order], y[order]
    print(f'raw rows {total}, unique logs {len(y)}, dropped retakes {n_dup}',
          flush=True)
    return u, q, y, raw_uids, raw_qids


def phase_stats(q2c):
    u, q, y, raw_uids, raw_qids = load_answers()
    n_u, n_q = len(raw_uids), len(raw_qids)
    ic_all = np.bincount(q, minlength=n_q)
    print(f'users={n_u}, questions={n_q}, unique logs={len(y)}')

    print('\n-- item popularity thresholds --')
    for T in (400, 450, 500, 550, 600, 625, 650, 700):
        kept_i = ic_all >= T
        concepts = {c for rq in raw_qids[kept_i] for c in q2c[int(rq)]}
        print(f'  items>={T}: n={kept_i.sum()}, concepts={len(concepts)}, '
              f'logs={int(ic_all[kept_i].sum())}')

    kept_i = ic_all >= ITEM_POP
    m = kept_i[q]
    uc = np.bincount(u[m], minlength=n_u)
    print(f'\n-- retained logs per user with item threshold {ITEM_POP} --')
    for lo, hi in ((15, 100), (20, 100), (20, 80), (30, 100), (50, 100)):
        sel = (uc >= lo) & (uc <= hi)
        print(f'  band [{lo},{hi}]: users={sel.sum()}, '
              f'logs={int(uc[sel].sum())}, avg={uc[sel].mean():.1f}')


def build(q2c):
    rng = np.random.default_rng(SEED)
    u, q, y, raw_uids, raw_qids = load_answers()
    n_u, n_q = len(raw_uids), len(raw_qids)

    # step 1: popular items
    ic_all = np.bincount(q, minlength=n_q)
    kept_i = ic_all >= ITEM_POP
    print(f'items with >={ITEM_POP} answers: {kept_i.sum()}')

    # step 2: users with a light/medium number of retained logs
    m = kept_i[q]
    uc = np.bincount(u[m], minlength=n_u)
    elig_u = np.where((uc >= USER_LO) & (uc <= USER_HI))[0]
    print(f'eligible users (retained logs in [{USER_LO},{USER_HI}]): '
          f'{len(elig_u)}')

    # step 3: fixed-seed student subsample to match paper scale
    if USER_SUBSAMPLE is not None and len(elig_u) > USER_SUBSAMPLE:
        pick = rng.choice(elig_u, size=USER_SUBSAMPLE, replace=False)
        kept_u = np.zeros(n_u, dtype=bool)
        kept_u[np.sort(pick)] = True
        print(f'subsampled students: {kept_u.sum()}')
    else:
        kept_u = np.zeros(n_u, dtype=bool)
        kept_u[elig_u] = True

    # drop items no selected user answered
    m = kept_u[u] & kept_i[q]
    kept_i &= np.bincount(q[m], minlength=n_q) > 0
    m = kept_u[u] & kept_i[q]
    u, q, y = u[m], q[m], y[m]

    # contiguous 1-based new ids, ordered by original id
    old_u = np.where(kept_u)[0]
    old_i = np.where(kept_i)[0]
    item_new = {int(old): i + 1 for i, old in enumerate(old_i)}
    kept_concepts = sorted({c for old in old_i for c in q2c[int(raw_qids[old])]})
    concept_new = {c: i + 1 for i, c in enumerate(kept_concepts)}
    n_item, n_concept = len(old_i), len(kept_concepts)
    print(f'final: students={len(old_u)}, exercises={n_item}, '
          f'concepts={n_concept}, logs={len(y)}')

    # step 4: group logs per student, 8/1/1 split
    by_user = np.argsort(u, kind='stable')
    u, q, y = u[by_user], q[by_user], y[by_user]
    bounds = np.searchsorted(u, np.arange(n_u + 1))

    train, val, test = [], [], []
    for new_uid, old_uid in enumerate(tqdm(old_u, desc='splitting students'),
                                      start=1):
        lo, hi = int(bounds[old_uid]), int(bounds[old_uid + 1])
        idx = np.arange(lo, hi)
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(n * 0.8)
        n_va = max(1, int(n * 0.1))
        parts = (idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:])

        for k in parts[0]:
            oq = int(q[k])
            train.append({
                'user_id': new_uid,
                'exer_id': item_new[oq],
                'score': float(y[k]),
                'knowledge_code': [concept_new[c]
                                   for c in q2c[int(raw_qids[oq])]],
            })
        for target, part in ((val, parts[1]), (test, parts[2])):
            if len(part) == 0:
                continue
            logs = []
            for k in part:
                oq = int(q[k])
                logs.append({
                    'exer_id': item_new[oq],
                    'score': float(y[k]),
                    'knowledge_code': [concept_new[c]
                                       for c in q2c[int(raw_qids[oq])]],
                })
            target.append({'user_id': new_uid, 'logs': logs})

    os.makedirs(OUT_DIR, exist_ok=True)
    for name, obj in (('train_set.json', train),
                      ('val_set.json', val),
                      ('test_set.json', test)):
        print(f'writing data/Eedi/{name} ...', flush=True)
        with open(os.path.join(OUT_DIR, name), 'w', encoding='utf-8') as f:
            json.dump(obj, f)
        print(f'wrote data/Eedi/{name}: {len(obj)} entries')

    print('\n=== update config files with this line ===')
    print(f'CONFIG (students,exercises,concepts): '
          f'{len(old_u)},{n_item},{n_concept}')


if __name__ == '__main__':
    q2c = load_question_concepts()
    phase = sys.argv[1] if len(sys.argv) > 1 else 'stats'
    if phase == 'stats':
        phase_stats(q2c)
    elif phase == 'build':
        build(q2c)
    else:
        print('usage: python preprocess_eedi.py [stats|build]')
