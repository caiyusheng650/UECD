import torch
import numpy as np
import json
import sys
import os
import csv
import time
import shutil
import glob as _glob
import re as _re
import subprocess
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from data_loader import ValTestDataLoader
from model import Net

_CONSOLE_OUT = sys.__stdout__
_CONSOLE_ERR = sys.__stderr__


# can be changed according to config.txt
# exer_n = 17746
# knowledge_n = 123
# student_n = 4163
# exer_n = 714
# knowledge_n = 39
# student_n = 10000
# #########assist17
# exer_n = 3162
# knowledge_n = 102
# student_n = 1709
topk=20


class _Tee(object):
    def __init__(self, console, log_file):
        self.console = console
        self.log_file = log_file

    def write(self, data):
        self.console.write(data)
        self.log_file.write(data)

    def flush(self):
        self.console.flush()
        self.log_file.flush()

    def isatty(self):
        return self.console.isatty()

    def fileno(self):
        return self.console.fileno()

    @property
    def encoding(self):
        return self.console.encoding


def setup_experiment():
    run_name = 'predict_' + time.strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('experiments', run_name)
    os.makedirs(run_dir, exist_ok=False)
    shutil.copy2('config.txt', os.path.join(run_dir, 'config_snapshot.txt'))
    if os.path.exists('config_Eedi.txt'):
        shutil.copy2('config_Eedi.txt',
                     os.path.join(run_dir, 'config_Eedi_snapshot.txt'))

    git_commit, git_dirty = 'unknown', False
    try:
        repo_root = os.path.dirname(os.path.abspath(__file__))
        r = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo_root,
                           capture_output=True, text=True)
        git_commit = r.stdout.strip() or 'unknown'
        d = subprocess.run(['git', 'status', '--porcelain'], cwd=repo_root,
                           capture_output=True, text=True)
        git_dirty = bool(d.stdout.strip())
    except Exception:
        pass

    metadata = {
        'run_name': run_name,
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'topk': topk,
        'student_n': student_n,
        'exer_n': exer_n,
        'knowledge_n': knowledge_n,
        'git_commit': git_commit,
        'git_dirty': git_dirty,
        'python': sys.version.split()[0],
        'torch': torch.__version__,
    }
    with open(os.path.join(run_dir, 'metadata.json'), 'w', encoding='utf8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    log_handle = open(os.path.join(run_dir, 'run.log'), 'w',
                      encoding='utf8', buffering=1)
    sys.stdout = _Tee(_CONSOLE_OUT, log_handle)
    sys.stderr = _Tee(_CONSOLE_ERR, log_handle)
    print('=' * 70)
    print('prediction run: %s' % run_dir)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print('=' * 70, flush=True)
    return run_dir, log_handle


def find_snapshots():
    """Return (tag, sorted_epochs) of checkpoints to evaluate.

    Tag is the run id embedded in snapshot filenames
    (model/NCDM_UECD_<tag>_epoch<N>). If UECD_PREDICT_RUN is set it is used
    verbatim; otherwise the most recently written run in model/ wins.
    """
    tag = os.environ.get('UECD_PREDICT_RUN', '')
    if tag:
        files = _glob.glob(os.path.join('model', 'NCDM_UECD_%s_epoch*' % tag))
        return tag, sorted(int(_re.search(r'epoch(\d+)', f).group(1))
                           for f in files)
    runs = {}
    for f in _glob.glob(os.path.join('model', 'NCDM_UECD_*_epoch*')):
        m = _re.search(r'NCDM_UECD_(.*)_epoch(\d+)', f)
        if not m:
            continue
        key = m.group(1)
        runs.setdefault(key, {'mtime': 0.0, 'epochs': []})
        runs[key]['mtime'] = max(runs[key]['mtime'], os.path.getmtime(f))
        runs[key]['epochs'].append(int(m.group(2)))
    if not runs:
        return '', []
    newest = max(runs, key=lambda k: runs[k]['mtime'])
    return newest, sorted(runs[newest]['epochs'])


def test():
    run_dir, log_handle = setup_experiment()
    metrics_path = os.path.join(run_dir, 'test_metrics.csv')
    with open(metrics_path, 'w', newline='', encoding='utf8') as f:
        csv.DictWriter(f, fieldnames=['epoch', 'accuracy', 'rmse', 'auc']
                       ).writeheader()

    tag, epochs = find_snapshots()
    if not epochs:
        raise SystemExit('no checkpoints found in model/ '
                         '(set UECD_PREDICT_RUN=<run tag> to select one)')
    print('testing model snapshots: tag=%s epochs=%s' % (tag, epochs))

    data_loader = ValTestDataLoader('test')
    device = torch.device('cpu')
    net = Net(student_n, exer_n, knowledge_n, topk, device)
    # net = Net(knowledge_n, exer_n, student_n)
    print('testing model...')
    best_acc = 0.0
    best_acc_epoch = -1
    best_auc = 0.0
    best_auc_epoch = -1
    epoch_bar = tqdm(epochs, desc='Test epochs', position=0,
                     file=_CONSOLE_ERR)
    for epoch in epoch_bar:
        data_loader.reset()
        load_snapshot(net, os.path.join(
            'model', 'NCDM_UECD_%s_epoch%d' % (tag, epoch)))
        net = net.to(device)
        net.eval()

        correct_count, exer_count = 0, 0
        pred_all, label_all = [], []
        test_bar = tqdm(total=len(data_loader.data), desc=f'Test epoch {epoch}',
                        position=1, leave=False, file=_CONSOLE_ERR)
        while not data_loader.is_end():
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
            out_put = net(input_stu_ids, input_exer_ids, input_knowledge_embs)
            out_put = out_put.view(-1)
            # compute accuracy
            for i in range(len(labels)):
                if (labels[i] == 1 and out_put[i] > 0.5) or (labels[i] == 0 and out_put[i] < 0.5):
                    correct_count += 1
            exer_count += len(labels)
            pred_all += out_put.tolist()
            label_all += labels.tolist()
            test_bar.update(1)
        test_bar.close()

        pred_all = np.array(pred_all)
        label_all = np.array(label_all)
        # compute accuracy
        accuracy = correct_count / exer_count
        # compute RMSE
        rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
        # compute AUC
        auc = roc_auc_score(label_all, pred_all)
        epoch_bar.set_postfix(acc='%.4f' % accuracy, auc='%.4f' % auc)
        tqdm.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f' % (epoch, accuracy, rmse, auc))
        if accuracy > best_acc:
            best_acc, best_acc_epoch = accuracy, epoch
        if auc > best_auc:
            best_auc, best_auc_epoch = auc, epoch
        with open(metrics_path, 'a', newline='', encoding='utf8') as f:
            csv.DictWriter(f, fieldnames=['epoch', 'accuracy', 'rmse', 'auc']
                           ).writerow({'epoch': epoch,
                                       'accuracy': round(accuracy, 6),
                                       'rmse': round(rmse, 6),
                                       'auc': round(auc, 6)})

    with open(os.path.join(run_dir, 'summary.txt'), 'w', encoding='utf8') as f:
        f.write('UECD prediction summary: %s\n'
                % time.strftime('%Y-%m-%d %H:%M:%S'))
        f.write('=' * 70 + '\n')
        f.write('best test accuracy: %.6f @ epoch %d\n'
                % (best_acc, best_acc_epoch))
        f.write('best test auc:      %.6f @ epoch %d\n'
                % (best_auc, best_auc_epoch))
        f.write('per-epoch metrics: test_metrics.csv\n')
        f.write('full console log:  run.log\n')
    print('best test accuracy: %.6f @ epoch %d' % (best_acc, best_acc_epoch))
    print('best test auc:      %.6f @ epoch %d' % (best_auc, best_auc_epoch))
    print('summary written to %s' % os.path.join(run_dir, 'summary.txt'))

    sys.stdout = _CONSOLE_OUT
    sys.stderr = _CONSOLE_ERR
    log_handle.close()


def load_snapshot(model, filename):
    f = open(filename, 'rb')
    model.load_state_dict(torch.load(f, map_location=lambda s, loc: s))
    f.close()



if __name__ == '__main__':


    # global student_n, exer_n, knowledge_n
    with open('config.txt') as i_f:
        i_f.readline()
        student_n, exer_n, knowledge_n = list(map(eval, i_f.readline().split(',')))

    test()