
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import sys
from sklearn.metrics import roc_auc_score
from data_loader import TrainDataLoader, ValTestDataLoader
from model import Net
import random
import os
import torch.nn.functional as F
# Debug switches: enable only when diagnosing CUDA errors, e.g.
#   set UECD_DEBUG=1 && python train.py cuda:0 10
# Both severely slow down training (synchronous kernel launches and
# per-op autograd checks) and must stay off for normal runs.
if os.environ.get('UECD_DEBUG') == '1':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    torch.autograd.set_detect_anomaly(True)
    print('[WARN] UECD_DEBUG=1: CUDA_LAUNCH_BLOCKING and anomaly detection '
          'are ON, training will be much slower.')
import pandas as pd
import csv
import time
import shutil
import subprocess
from tqdm import tqdm


# original console streams: progress bars render only on the console,
# while print()/tqdm.write() output is teed into experiments/<run>/run.log
_CONSOLE_OUT = sys.__stdout__
_CONSOLE_ERR = sys.__stderr__
RUN_DIR = None
METRICS_PATH = None


class _Tee(object):
    """Write to the real console and to a log file simultaneously."""

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


def setup_experiment(device):
    """Create experiments/<timestamp>/ with config snapshot, metadata and
    a full tee'd run.log; returns (run_dir, log_handle)."""
    global RUN_DIR, METRICS_PATH, RUN_TAG
    run_name = 'run_' + time.strftime('%Y%m%d_%H%M%S')
    RUN_TAG = run_name
    run_dir = os.path.join('experiments', run_name)
    os.makedirs(run_dir, exist_ok=False)
    RUN_DIR = run_dir
    METRICS_PATH = os.path.join(run_dir, 'metrics.csv')

    shutil.copy2('config.txt', os.path.join(run_dir, 'config_snapshot.txt'))
    if os.path.exists('config_Eedi.txt'):
        shutil.copy2('config_Eedi.txt', os.path.join(run_dir, 'config_Eedi_snapshot.txt'))

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
        'device': str(device),
        'seed': 2024,
        'epochs': epoch_n,
        'batch_size': batch_size,
        'topk': topk,
        'lr': 0.001,
        'student_n': student_n,
        'exer_n': exer_n,
        'knowledge_n': knowledge_n,
        'git_commit': git_commit,
        'git_dirty': git_dirty,
        'python': sys.version.split()[0],
        'torch': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_device': torch.cuda.get_device_name(device)
        if torch.cuda.is_available() and 'cuda' in str(device) else None,
    }
    with open(os.path.join(run_dir, 'metadata.json'), 'w', encoding='utf8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    log_handle = open(os.path.join(run_dir, 'run.log'), 'w',
                      encoding='utf8', buffering=1)
    sys.stdout = _Tee(_CONSOLE_OUT, log_handle)
    sys.stderr = _Tee(_CONSOLE_ERR, log_handle)

    print('=' * 70)
    print('experiment run: %s' % run_dir)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print('=' * 70, flush=True)
    return run_dir, log_handle


def write_summary(run_dir, total_train_time, epoch_timings,
                  best_acc, best_acc_epoch, best_auc, best_auc_epoch):
    path = os.path.join(run_dir, 'summary.txt')
    with open(path, 'w', encoding='utf8') as f:
        f.write('UECD training summary: %s\n'
                % time.strftime('%Y-%m-%d %H:%M:%S'))
        f.write('=' * 70 + '\n')
        f.write('epochs planned/run: %d / %d\n' % (epoch_n, len(epoch_timings)))
        f.write('total time: %.2f minutes\n' % (total_train_time / 60))
        f.write('mean per epoch: %.2f +/- %.2f seconds\n'
                % (np.mean(epoch_timings), np.std(epoch_timings)))
        f.write('best validation accuracy: %.6f @ epoch %d\n'
                % (best_acc, best_acc_epoch))
        f.write('best validation auc:      %.6f @ epoch %d\n'
                % (best_auc, best_auc_epoch))
        f.write('per-epoch metrics: metrics.csv\n')
        f.write('full console log:   run.log\n')
    print('summary written to %s' % path)


# can be changed according to config.txt
# exer_n = 17746
# knowledge_n = 123
# student_n = 4163
# exer_n = 714
# knowledge_n = 39
# student_n = 10000
# can be changed according to command parameter
# device = torch.device(('cuda:3') if torch.cuda.is_available() else 'cpu')
epoch_n = 5
topk=20
batch_size=256




def set_seed(seed: int) -> int:
    r"""
    Set the seed for the random number generators.

    Parameters:
    -----------
    seed : int
        The seed to set. If seed is -1, a random seed between 0 and 2048 will be generated.

    Returns:
    --------
    seed : int
        The actual seed used.
    """
    if seed == -1:
        seed = random.randint(0, 2048)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"[Seed] >>> Set seed: {seed}")
    return seed


def train(device):
    print('loading training data ...', flush=True)
    data_loader = TrainDataLoader(topk,batch_size)
    net = Net(student_n, exer_n, knowledge_n,topk,device).to(device)
    # net = Net(knowledge_n, exer_n, student_n).to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    global RUN_DIR, METRICS_PATH
    RUN_DIR, log_handle = setup_experiment(device)
    METRICS_PATH = os.path.join(RUN_DIR, 'metrics.csv')

    total_train_time = 0.0
    epoch_timings = []

    # per-epoch experiment record
    with open(METRICS_PATH, 'w', newline='', encoding='utf8') as csvfile:
        fieldnames = [
            'epoch',
            'start_time',
            'end_time',
            'duration_s',
            'avg_duration_s',
            'total_time_min',
            'batch_count',
            'avg_train_loss',
            'val_accuracy',
            'val_rmse',
            'val_auc',
        ]
        csv.DictWriter(csvfile, fieldnames=fieldnames).writeheader()

    print('training model...')
    # loss_function = nn.NLLLoss()
    negloss_function = nn.MSELoss()
    loss_function = nn.BCELoss()
    final_loss = []
    final_neg_losses = []
    best_acc = 0.0
    best_acc_epoch = -1
    best_auc = 0.0
    best_auc_epoch = -1
    epoch_bar = tqdm(range(epoch_n), desc='Epochs', position=0,
                     file=_CONSOLE_ERR)
    for epoch in epoch_bar:

        epoch_start = time.perf_counter()

        data_loader.reset()
        running_loss = 0.0
        epoch_pos_loss_sum = 0.0
        batch_count = 0
        neg_losses = 0.0


        if 'cuda' in str(device):
            torch.cuda.synchronize(device)

        n_batches = len(data_loader.data) // data_loader.batch_size
        batch_bar = tqdm(total=n_batches, desc=f'Train epoch {epoch + 1}',
                         position=1, leave=False, file=_CONSOLE_ERR)
        while not data_loader.is_end():
            batch_count += 1
            all_neg_loss=0

            stu_ids, pos_ids, knowledge_embs, labels, neg_ids = data_loader.next_batch()
            stu_ids, knowledge_embs, labels = stu_ids.to(device),  knowledge_embs.to(device), labels.to(device)
            # print("labels",labels.size())             [256]
            can_pos_ids, can_neg_ids = net.item_sim_sample(pos_ids, neg_ids, topk, device)
            output_pos = net.forward(stu_ids, can_pos_ids, knowledge_embs)
            pos_loss = loss_function(output_pos.squeeze(), labels.float())

            # bpr_losses, qdcc_losses = net.forward(stu_ids, can_neg_ids, knowledge_embs,
            #                                                                can_pos_ids,
            #                                                                labels)  # [256, topk, 1] & [256,topk]

            output_negs, neg_scores, bpr_losses, qdcc_losses = net.forward(stu_ids, can_neg_ids, knowledge_embs, can_pos_ids, labels)

            # bpr_losses1, bpr_losses2 = net.forward(stu_ids, neg_ids,knowledge_embs, exer_knowledge_data, mis_neg_ids, pos_ids, labels)
            # bpr_losses = bpr_losses1 + bpr_losses2

            for i in range(output_negs.size(1)):
                each_neg=output_negs[torch.arange(output_negs.size(0)), i].view(-1,1)
                neg_score=neg_scores[torch.arange(neg_scores.size(0)), i].squeeze()

                neg_loss = negloss_function(each_neg.squeeze(), neg_score)
                all_neg_loss+=neg_loss

            loss = pos_loss + all_neg_loss/output_negs.size(1) + 0.1 * bpr_losses + qdcc_losses
            # loss = pos_loss + 0.1 * bpr_losses + qdcc_losses

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            net.apply_clipper()

            running_loss += pos_loss.item()
            epoch_pos_loss_sum += pos_loss.item()
            # neg_losses += final_neg_loss.item()
            batch_bar.set_postfix(loss='%.3f' % pos_loss.item())
            batch_bar.update(1)
            if batch_count % 200 == 199:
                batch_bar.write('[%d, %5d] loss: %.3f' % (epoch + 1, batch_count + 1, running_loss / 200))
                # print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_count + 1, running_loss))
                running_loss = 0.0

        batch_bar.close()

        if 'cuda' in str(device):
            torch.cuda.synchronize(device)
        epoch_duration = time.perf_counter() - epoch_start


        total_train_time += epoch_duration
        epoch_timings.append(epoch_duration)
        avg_train_loss = epoch_pos_loss_sum / batch_count

        if os.environ.get('UECD_SKIP_VAL'):
            # sanity-check mode: only watch the train-loss curve, skip val
            acc, rmse, auc = -1.0, -1.0, -1.0
        else:
            acc, rmse, auc = validate(net, epoch,device)
        epoch_bar.set_postfix(acc='%.4f' % acc, auc='%.4f' % auc)

        if acc > best_acc:
            best_acc, best_acc_epoch = acc, epoch + 1
        if auc > best_auc:
            best_auc, best_auc_epoch = auc, epoch + 1

        # one experiment-record row per epoch
        with open(METRICS_PATH, 'a', newline='', encoding='utf8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[
                'epoch', 'start_time', 'end_time', 'duration_s',
                'avg_duration_s', 'total_time_min', 'batch_count',
                'avg_train_loss', 'val_accuracy', 'val_rmse', 'val_auc'])
            writer.writerow({
                'epoch': epoch + 1,
                'start_time': time.strftime("%Y-%m-%d %H:%M:%S",
                                            time.localtime(epoch_start)),
                'end_time': time.strftime("%Y-%m-%d %H:%M:%S",
                                          time.localtime(epoch_start + epoch_duration)),
                'duration_s': round(epoch_duration, 2),
                'avg_duration_s': round(np.mean(epoch_timings), 2),
                'total_time_min': round(total_train_time / 60, 2),
                'batch_count': batch_count,
                'avg_train_loss': round(avg_train_loss, 6),
                'val_accuracy': round(acc, 6),
                'val_rmse': round(rmse, 6),
                'val_auc': round(auc, 6),
            })
        print('[Epoch %d] time %.2fs (avg %.2fs, total %.1fmin) | '
              'train_loss %.6f | val acc %.6f rmse %.6f auc %.6f'
              % (epoch + 1, epoch_duration, np.mean(epoch_timings),
                 total_train_time / 60, avg_train_loss, acc, rmse, auc),
              flush=True)

        save_snapshot(net, 'model/NCDM_UECD_' + RUN_TAG + '_epoch' + str(epoch + 1), RUN_TAG)

    final_time_msg = "\nTraining Time Summary:\nTotal: %.2f minutes\nMean per epoch: %.2f±%.2f seconds" % (
        total_train_time / 60,
        np.mean(epoch_timings),
        np.std(epoch_timings)
    )
    print(final_time_msg)
    write_summary(RUN_DIR, total_train_time, epoch_timings,
                  best_acc, best_acc_epoch, best_auc, best_auc_epoch)

    sys.stdout = _CONSOLE_OUT
    sys.stderr = _CONSOLE_ERR
    log_handle.close()

def validate(model, epoch, device):
    data_loader = ValTestDataLoader('validation')
    net = Net(student_n, exer_n, knowledge_n,topk,device)
    # net = Net(knowledge_n, exer_n, student_n)
    print('validating model...')
    data_loader.reset()
    # load model parameters
    net.load_state_dict(model.state_dict())
    net = net.to(device)
    net.eval()

    correct_count, exer_count = 0, 0
    batch_count, batch_avg_loss = 0, 0.0
    pred_all, label_all = [], []
    val_bar = tqdm(total=len(data_loader.data), desc='Validating',
                  position=1, leave=False, file=_CONSOLE_ERR)
    while not data_loader.is_end():
        batch_count += 1
        # input_stu_ids, input_exer_ids, input_knowledge_embs, labels,exer_knowledge_data = data_loader.next_batch()
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
            device), input_knowledge_embs.to(device), labels.to(device)
        # output = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs,exer_knowledge_data)
        output = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
        output = output.view(-1)
        # compute accuracy
        for i in range(len(labels)):
            if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
                correct_count += 1
        exer_count += len(labels)
        pred_all += output.to(torch.device('cpu')).tolist()
        label_all += labels.to(torch.device('cpu')).tolist()
        val_bar.update(1)
    val_bar.close()

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    # compute accuracy
    accuracy = correct_count / exer_count
    # compute RMSE
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    # compute AUC
    auc = roc_auc_score(label_all, pred_all)
    tqdm.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f' % (epoch+1, accuracy, rmse, auc))

    return accuracy, rmse, auc


def save_snapshot(model, filename, tag):
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()
    f = None
    # keep a bounded sliding window of checkpoints so long runs don't fill the disk.
    # only clean up files written by the SAME run (tag in filename), so leftover
    # snapshots from other runs (with larger epoch numbers) can't evict ours.
    keep = int(os.environ.get('UECD_KEEP_CKPT', '2'))
    import glob as _glob
    import re as _re
    pattern = os.path.join(os.path.dirname(filename), 'NCDM_UECD_' + tag + '_epoch*')
    ckpts = sorted(_glob.glob(pattern),
                   key=lambda p: int(_re.search(r'epoch(\d+)', p).group(1)))
    for stale in ckpts[:-keep] if keep > 0 else ckpts:
        try:
            os.remove(stale)
        except OSError:
            pass


if __name__ == '__main__':
    if (len(sys.argv) != 3) or ((sys.argv[1] != 'cpu') and ('cuda:' not in sys.argv[1])) or (not sys.argv[2].isdigit()):
        print('command:\n\tpython train.py {device} {epoch}\nexample:\n\tpython train.py cuda:0 70')
        exit(1)
    else:
        device = torch.device(sys.argv[1])
        epoch_n = int(sys.argv[2])
    # global student_n, exer_n, knowledge_n, device
    with open('config.txt') as i_f:
        i_f.readline()
        student_n, exer_n, knowledge_n = list(map(eval, i_f.readline().split(',')))

    set_seed(2024)

    train(device)