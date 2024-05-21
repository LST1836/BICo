# -*- coding: utf-8 -*-
from parameters import parse_ideology_args
args = parse_ideology_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
import time
import logging
import tqdm
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from load_data import load_ideology_data
from text_helper import normalizeTweet
from model import IdeologyNet


tree_structure = [[1], [5], [2, 2, 3, 2, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]]


def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


setup_seed(args.seed)

torch.cuda.empty_cache()

# logger
logging.basicConfig(format='%(message)s', level=logging.INFO,
                    filename=os.path.join(args.log_dir, 'training.txt'),
                    filemode='a')

logger = logging.getLogger(__name__)


def printlog(message, printout=True):
    if printout:
        print(message)
    logger.info(message)


# tokenizer
text_tokenizer = AutoTokenizer.from_pretrained(args.plm_path)
concept_tokenizer = text_tokenizer

# load data
train_text, train_label, train_facet_idx = load_ideology_data(args.train_data_path)
val_text, val_label, val_facet_idx = load_ideology_data(args.val_data_path)
test_text, test_label, test_facet_idx = load_ideology_data(args.test_data_path)

train_text = [normalizeTweet(t) for t in train_text]
val_text = [normalizeTweet(t) for t in val_text]
test_text = [normalizeTweet(t) for t in test_text]
train_label, train_facet_idx = torch.LongTensor(train_label).cuda(), torch.LongTensor(train_facet_idx).cuda()
val_label, val_facet_idx = torch.LongTensor(val_label).cuda(), torch.LongTensor(val_facet_idx).cuda()
test_label, test_facet_idx = torch.LongTensor(test_label).cuda(), torch.LongTensor(test_facet_idx).cuda()
train_size = len(train_text)

# load concept
facet = open('concept_facet.txt', encoding='utf-8').readlines()
facet = [f.strip() for f in facet]
ideology = open('concept_ideology.txt', encoding='utf-8').readlines()
ideology = [i.strip() for i in ideology]

facet = [normalizeTweet(f) for f in facet]
ideology = [normalizeTweet(i) for i in ideology]

concept_encode_dict = concept_tokenizer(facet + ideology, padding=True, return_tensors='pt')
ids_concept, mask_concept = concept_encode_dict['input_ids'], concept_encode_dict['attention_mask']

# loss function
criterion = nn.CrossEntropyLoss().cuda()

# model
net = IdeologyNet(args, tree_structure, ids_concept, mask_concept).cuda()
multi_gpu = False
if len(args.gpu_ids) > 1:
    net = nn.DataParallel(net, device_ids=[eval(x.strip()) for x in args.gpu_ids.split(',')])
    multi_gpu = True

# optimizer
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
param_all = list(net.module.named_parameters() if multi_gpu else net.named_parameters())
param_text_encoder = [(n, p) for n, p in param_all if 'text_encoder' in n]
param_other = [(n, p) for n, p in param_all if not ('encoder' in n)]
del param_all
optimizer_grouped_params = [
        {"params": [p for n, p in param_text_encoder if not any(nd in n for nd in no_decay)],
         "lr": args.lr_text_encoder,
         "weight_decay": args.wd_plm},
        {"params": [p for n, p in param_text_encoder if any(nd in n for nd in no_decay)],
         "lr": args.lr_text_encoder,
         "weight_decay": 0.0},
        {"params": [p for n, p in param_other if not any(nd in n for nd in no_decay)],
         "lr": args.lr_other,
         "weight_decay": args.wd_other},
        {"params": [p for n, p in param_other if any(nd in n for nd in no_decay)],
         "lr": args.lr_other,
         "weight_decay": 0.0}
]
del param_text_encoder, param_other
optimizer = AdamW(optimizer_grouped_params, betas=(0.9, 0.999), eps=1e-8)

if args.warmup:
    total_steps = (train_size // args.batch_size + 1) * args.num_epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warm_ratio * total_steps, num_training_steps=total_steps
    )


@torch.no_grad()
def evaluate(text, label, facet_idx, task='Val'):
    net.eval()

    data_size = len(text)

    loss_epoch = 0.0
    pred_losses_epoch = np.zeros(12)
    facet_steps = np.zeros(12)
    pred_epoch, truth_epoch = None, None
    facet_idx_epoch = None

    all_indices = torch.arange(data_size).split(args.batch_size)
    for batch_indices in all_indices:
        batch_facet_idx = facet_idx[batch_indices]
        batch_text = [text[i] for i in batch_indices]
        batch_facet = [facet[i] for i in batch_facet_idx]
        encode_dict = text_tokenizer(batch_facet, text_pair=batch_text, padding=True, truncation=True,
                                     max_length=args.max_seq_length, return_tensors='pt')

        logits_list, _ = net(encode_dict['input_ids'].cuda(), encode_dict['attention_mask'].cuda(), batch_facet_idx)
        _, pred = torch.max(torch.cat(logits_list, dim=0), dim=1)  # [b]

        batch_label = label[batch_indices]
        label_list = []
        for i in range(12):
            batch_label_i = batch_label[batch_facet_idx == i]
            if len(batch_label_i) == 0:
                label_list.append(0)
                continue
            label_list.append(batch_label_i)
        batch_label_facet_sorted = torch.cat([x for x in label_list if not isinstance(x, int)], dim=0)
        batch_facet_idx_sorted, _ = torch.sort(batch_facet_idx)

        pred_losses = []
        j = 0
        for i in range(12):
            if isinstance(label_list[i], int):
                pred_losses.append(torch.tensor([0]).cuda())
                continue
            pred_losses.append((criterion(logits_list[j], label_list[i])).unsqueeze(0))
            j += 1
            facet_steps[i] += 1
        assert len(logits_list) == j
        pred_losses = torch.cat(pred_losses, dim=0)  # [12]

        loss = torch.sum(pred_losses) / len(logits_list)
        # loss = torch.sum(pred_losses)

        loss_epoch += loss.item()
        pred_losses_epoch += np.array(pred_losses.cpu())
        if pred_epoch is None:
            pred_epoch = pred.cpu()
            truth_epoch = batch_label_facet_sorted.cpu()
            facet_idx_epoch = batch_facet_idx_sorted.cpu()
        else:
            pred_epoch = torch.cat((pred_epoch, pred.cpu()), dim=0)
            truth_epoch = torch.cat((truth_epoch, batch_label_facet_sorted.cpu()), dim=0)
            facet_idx_epoch = torch.cat((facet_idx_epoch, batch_facet_idx_sorted.cpu()), dim=0)

    num_steps = data_size // args.batch_size
    loss_epoch /= num_steps
    pred_losses_epoch /= facet_steps
    results_dict = performance(pred_epoch, truth_epoch, facet_idx_epoch)
    printlog(f"\n{task}: loss={loss_epoch:.4f}, pred_losses={list(np.around(pred_losses_epoch, 4))}\n"
             f"avg_acc={results_dict['avg_acc']:.4f}, global_acc={results_dict['global_acc']:.4f}, "
             f"avg_f1={results_dict['avg_f1']:.4f}, global_f1={results_dict['global_f1']:.4f}, "
             f"global_p={results_dict['global_p']:.4f}, global_r={results_dict['global_r']:.4f}, "
             f"acc={results_dict['acc']}, f1={results_dict['f1']}, p={results_dict['p']}, r={results_dict['r']}")

    net.train()

    return results_dict


def train():
    best_epoch = 0
    best_val_global_f1 = 0.0
    report_dict = {'avg_acc': 0.0, 'global_acc': 0.0, 'avg_f1': 0.0, 'global_f1': 0.0, 'global_p': 0.0, 'global_r': 0.0,
                   'acc': [], 'f1': [], 'p': [], 'r': []}

    net.train()
    for epoch in range(args.num_epoch):
        printlog(f'\nEpoch: {epoch + 1}')

        printlog(f"lr_text_encoder: {optimizer.state_dict()['param_groups'][0]['lr']}")
        printlog(f"lr_other: {optimizer.state_dict()['param_groups'][2]['lr']}")

        loss_epoch = 0.0
        pred_losses_epoch = np.zeros(12)
        cl_losses_epoch = np.zeros(12)
        facet_steps = np.zeros(12)
        pred_epoch, truth_epoch = None, None
        facet_idx_epoch = None

        start = time.time()
        step = 0

        all_indices = torch.randperm(train_size).split(args.batch_size)
        for batch_indices in tqdm.tqdm(all_indices, desc='batch'):
            step += 1
            batch_facet_idx = train_facet_idx[batch_indices].cuda()  # [b]
            batch_text = [train_text[i] for i in batch_indices]
            batch_facet = [facet[i] for i in batch_facet_idx]
            encode_dict = text_tokenizer(batch_facet, text_pair=batch_text, padding=True, truncation=True,
                                         max_length=args.max_seq_length, return_tensors='pt')
            batch_label = train_label[batch_indices].cuda()  # [b, 12]

            logits_list, cl_loss_list = net(encode_dict['input_ids'].cuda(), encode_dict['attention_mask'].cuda(),
                                            batch_facet_idx, batch_label)
            cl_losses = torch.cat(cl_loss_list, dim=0)  # [12]
            _, pred = torch.max(torch.cat(logits_list, dim=0), dim=1)  # [b]

            label_list = []
            for i in range(12):
                batch_label_i = batch_label[batch_facet_idx == i]
                if len(batch_label_i) == 0:
                    label_list.append(0)
                    continue
                label_list.append(batch_label_i)
            batch_label_facet_sorted = torch.cat([x for x in label_list if not isinstance(x, int)], dim=0)
            batch_facet_idx_sorted, _ = torch.sort(batch_facet_idx)

            pred_losses = []
            j = 0
            for i in range(12):
                if isinstance(label_list[i], int):
                    pred_losses.append(torch.tensor([0]).cuda())
                    continue
                pred_losses.append((criterion(logits_list[j], label_list[i])).unsqueeze(0))
                j += 1
                facet_steps[i] += 1
            assert len(logits_list) == j
            pred_losses = torch.cat(pred_losses, dim=0)  # [12]

            loss = pred_losses + torch.tensor(args.cl_loss_weight).cuda() * cl_losses  # [12]
            loss = torch.sum(loss) / len(logits_list)
            # loss = torch.sum(loss)

            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.warmup:
                scheduler.step()

            loss_epoch += loss.item()
            pred_losses_epoch += np.array(pred_losses.detach().cpu())
            cl_losses_epoch += np.array(cl_losses.detach().cpu())
            if pred_epoch is None:
                pred_epoch = pred.cpu()
                truth_epoch = batch_label_facet_sorted.cpu()
                facet_idx_epoch = batch_facet_idx_sorted.cpu()
            else:
                pred_epoch = torch.cat((pred_epoch, pred.cpu()), dim=0)
                truth_epoch = torch.cat((truth_epoch, batch_label_facet_sorted.cpu()), dim=0)
                facet_idx_epoch = torch.cat((facet_idx_epoch, batch_facet_idx_sorted.cpu()), dim=0)

            # checkpoint
            if step % (2900 // args.batch_size) == 0:
                num_steps = 2900 // args.batch_size
                loss_epoch /= num_steps
                pred_losses_epoch /= facet_steps
                cl_losses_epoch /= facet_steps
                results_dict = performance(pred_epoch, truth_epoch, facet_idx_epoch)
                printlog(f"\nCheckpoint: loss={loss_epoch:.4f}, "
                         f"pred_losses={list(np.around(pred_losses_epoch, 4))}, cl_losses={list(np.around(cl_losses_epoch, 4))}\n"
                         f"avg_acc={results_dict['avg_acc']:.4f}, global_acc={results_dict['global_acc']:.4f}, "
                         f"avg_f1={results_dict['avg_f1']:.4f}, global_f1={results_dict['global_f1']:.4f}, "
                         f"global_p={results_dict['global_p']:.4f}, global_r={results_dict['global_r']:.4f}, "
                         f"acc={results_dict['acc']}, f1={results_dict['f1']}, p={results_dict['p']}, r={results_dict['r']}")
                loss_epoch = 0.0
                pred_losses_epoch = np.zeros(12)
                cl_losses_epoch = np.zeros(12)
                facet_steps = np.zeros(12)
                pred_epoch, truth_epoch = None, None
                facet_idx_epoch = None

        val_results = evaluate(val_text, val_label, val_facet_idx, 'Val')
        test_results = evaluate(test_text, test_label, test_facet_idx, 'Test')
        if val_results['global_f1'] > best_val_global_f1:
            best_epoch = epoch + 1
            best_val_global_f1 = val_results['global_f1']
            for key in report_dict.keys():
                report_dict[key] = test_results[key]

        end = time.time()
        printlog('Training Time: {:.2f}s'.format(end - start))

    printlog(f"\nReport: best_epoch={best_epoch}\n"
             f"avg_acc={report_dict['avg_acc']:.4f}, global_acc={report_dict['global_acc']:.4f}, "
             f"avg_f1={report_dict['avg_f1']:.4f}, global_f1={report_dict['global_f1']:.4f}, "
             f"global_p={report_dict['global_p']:.4f}, global_r={report_dict['global_r']:.4f}, \n"
             f"acc={report_dict['acc']}, f1={report_dict['f1']}, p={report_dict['p']}, r={report_dict['r']}")


def performance(pred, truth, facet_idx):
    pred = np.array(pred)
    truth = np.array(truth)
    facet_idx = np.array(facet_idx)

    global_acc = accuracy_score(truth, pred)
    global_f1 = f1_score(truth, pred, average='macro', labels=list(set(truth)), zero_division=0)
    global_p = precision_score(truth, pred, average='macro', labels=list(set(truth)), zero_division=0)
    global_r = recall_score(truth, pred, average='macro', labels=list(set(truth)), zero_division=0)

    acc, f1, p, r = [], [], [], []
    for i in range(12):
        facet_mask = facet_idx == i
        if np.sum(facet_mask) == 0:
            acc.append(-1)
            f1.append(-1)
            p.append(-1)
            r.append(-1)
            continue
        pred_i = pred[facet_mask]
        truth_i = truth[facet_mask]
        acc.append(round(accuracy_score(truth_i, pred_i), 4))
        f1.append(round(f1_score(truth_i, pred_i, average='macro', labels=list(set(truth_i)), zero_division=0), 4))
        p.append(round(precision_score(truth_i, pred_i, average='macro', labels=list(set(truth_i)), zero_division=0), 4))
        r.append(round(recall_score(truth_i, pred_i, average='macro', labels=list(set(truth_i)), zero_division=0), 4))

    avg_acc = np.mean([x for x in acc if x >= 0])
    avg_f1 = np.mean([x for x in f1 if x >= 0])

    result_dict = {'avg_acc': avg_acc, 'global_acc': global_acc, 'avg_f1': avg_f1, 'global_f1': global_f1,
                   'global_p': global_p, 'global_r': global_r,
                   'acc': acc, 'f1': f1, 'p': p, 'r': r}

    return result_dict


if __name__ == '__main__':
    train()

