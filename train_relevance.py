# -*- coding: utf-8 -*-
from parameters import parse_relevance_args
args = parse_relevance_args()

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
from sklearn.metrics import f1_score, precision_score, recall_score
from load_data import load_relevance_data
from text_helper import normalizeTweet
from model import RelevanceNet


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
train_text, train_label = load_relevance_data(args.train_data_path)
val_text, val_label = load_relevance_data(args.val_data_path)
test_text, test_label = load_relevance_data(args.test_data_path)

train_text = [normalizeTweet(t) for t in train_text]
val_text = [normalizeTweet(t) for t in val_text]
test_text = [normalizeTweet(t) for t in test_text]
train_label = torch.LongTensor(train_label).cuda()
val_label = torch.LongTensor(val_label).cuda()
test_label = torch.LongTensor(test_label).cuda()
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
pos_weight = torch.tensor([eval(w.strip()) for w in args.pos_weight.split(',')])
pos_weight = (args.weight_scale * pos_weight).clamp(min=1.0)
criterion_list = [nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight[i]])).cuda() for i in range(12)]

# model
net = RelevanceNet(args, tree_structure, ids_concept, mask_concept).cuda()
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
def evaluate(text, label, task='Val'):
    net.eval()

    data_size = len(text)

    loss_epoch = 0.0
    pred_losses_epoch = np.zeros(12)
    pred_epoch = None

    all_indices = torch.arange(data_size).split(args.batch_size)
    for batch_indices in all_indices:
        batch_text = [text[i] for i in batch_indices]
        encode_dict = text_tokenizer(batch_text, padding=True, truncation=True, max_length=args.max_seq_length, return_tensors='pt')

        logits, _ = net(encode_dict['input_ids'].cuda(), encode_dict['attention_mask'].cuda())
        _, pred = torch.max(logits, dim=1)  # [b, 12]

        batch_label = label[batch_indices]

        pred_loss_list = []
        logits = logits.permute(2, 0, 1)
        batch_label_t = batch_label.transpose(0, 1)
        for i in range(12):
            pred_loss_list.append(criterion_list[i](logits[i], batch_label_t[i]).unsqueeze(0))
        pred_losses = torch.cat(pred_loss_list, dim=0)

        loss = torch.mean(pred_losses)

        loss_epoch += loss.item()
        pred_losses_epoch += np.array(pred_losses.cpu())
        if pred_epoch is None:
            pred_epoch = pred.cpu()
        else:
            pred_epoch = torch.cat((pred_epoch, pred.cpu()), dim=0)

    num_steps = data_size // args.batch_size
    loss_epoch /= num_steps
    pred_losses_epoch /= num_steps
    results_dict = performance(pred_epoch, label.cpu())
    printlog(f"\n{task}: loss={loss_epoch:.4f}, pred_losses={list(np.around(pred_losses_epoch, 4))}\n"
             f"avg_rela_f1={results_dict['avg_rela_f1']:.4f}, global_rela_f1={results_dict['global_rela_f1']:.4f}, "
             f"global_rela_p={results_dict['global_rela_p']:.4f}, global_rela_r={results_dict['global_rela_r']:.4f}, "
             f"rela_f1={results_dict['rela_f1']}, rela_p={results_dict['rela_p']}, rela_r={results_dict['rela_r']}")

    net.train()

    return results_dict


def train():
    best_epoch = 0
    best_val_global_rela_f1 = 0.0
    report_dict = {'avg_rela_f1': 0.0, 'global_rela_f1': 0.0, 'global_rela_p': 0.0, 'global_rela_r': 0.0,
                   'rela_f1': [], 'rela_p': [], 'rela_r': []}

    net.train()
    for epoch in range(args.num_epoch):
        printlog(f'\nEpoch: {epoch + 1}')

        printlog(f"lr_text_encoder: {optimizer.state_dict()['param_groups'][0]['lr']}")
        printlog(f"lr_other: {optimizer.state_dict()['param_groups'][2]['lr']}")

        loss_epoch = 0.0
        pred_losses_epoch = np.zeros(12)
        cl_losses_epoch = np.zeros(12)
        pred_epoch, truth_epoch = None, None

        start = time.time()
        step = 0

        all_indices = torch.randperm(train_size).split(args.batch_size)
        for batch_indices in tqdm.tqdm(all_indices, desc='batch'):
            step += 1
            batch_text = [train_text[i] for i in batch_indices]
            encode_dict = text_tokenizer(batch_text, padding=True, truncation=True, max_length=args.max_seq_length, return_tensors='pt')
            batch_label = train_label[batch_indices].cuda()  # [b, 12]

            logits, cl_loss_list = net(encode_dict['input_ids'].cuda(), encode_dict['attention_mask'].cuda(), batch_label)
            cl_losses = torch.cat(cl_loss_list, dim=0)  # [12]
            _, pred = torch.max(logits, dim=1)  # [b, 12]

            pred_loss_list = []
            logits = logits.permute(2, 0, 1)
            batch_label_t = batch_label.transpose(0, 1)
            for i in range(12):
                pred_loss_list.append(criterion_list[i](logits[i], batch_label_t[i]).unsqueeze(0))
            pred_losses = torch.cat(pred_loss_list, dim=0)

            loss = pred_losses + torch.tensor(args.cl_loss_weight).cuda() * cl_losses
            loss = torch.mean(loss)

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
                pred_epoch, truth_epoch = pred.cpu(), batch_label.cpu()
            else:
                pred_epoch = torch.cat((pred_epoch, pred.cpu()), dim=0)
                truth_epoch = torch.cat((truth_epoch, batch_label.cpu()), dim=0)

            # checkpoint
            if step % (2900 // args.batch_size) == 0:
                num_steps = 2900 // args.batch_size
                loss_epoch /= num_steps
                pred_losses_epoch /= num_steps
                cl_losses_epoch /= num_steps
                results_dict = performance(pred_epoch, truth_epoch)
                printlog(f"\nCheckpoint: loss={loss_epoch:.4f}, "
                         f"pred_losses={list(np.around(pred_losses_epoch, 4))}, cl_losses={list(np.around(cl_losses_epoch, 4))}\n"
                         f"avg_rela_f1={results_dict['avg_rela_f1']:.4f}, global_rela_f1={results_dict['global_rela_f1']:.4f}, "
                         f"global_rela_p={results_dict['global_rela_p']:.4f}, global_rela_r={results_dict['global_rela_r']:.4f}, "
                         f"rela_f1={results_dict['rela_f1']}, rela_p={results_dict['rela_p']}, rela_r={results_dict['rela_r']}")
                loss_epoch = 0.0
                pred_losses_epoch = np.zeros(12)
                cl_losses_epoch = np.zeros(12)
                pred_epoch, truth_epoch = None, None

        val_results = evaluate(val_text, val_label, 'Val')
        test_results = evaluate(test_text, test_label, 'Test')
        if val_results['global_rela_f1'] > best_val_global_rela_f1:
            best_epoch = epoch + 1
            best_val_global_rela_f1 = val_results['global_rela_f1']
            for key in report_dict.keys():
                report_dict[key] = test_results[key]

        end = time.time()
        printlog('Training Time: {:.2f}s'.format(end - start))

    printlog(f"\nReport: best_epoch={best_epoch}\n"
             f"avg_rela_f1={report_dict['avg_rela_f1']:.4f}, global_rela_f1={report_dict['global_rela_f1']:.4f}, "
             f"global_rela_p={report_dict['global_rela_p']:.4f}, global_rela_r={report_dict['global_rela_r']:.4f}, \n"
             f"rela_f1={report_dict['rela_f1']}, rela_p={report_dict['rela_p']}, rela_r={report_dict['rela_r']}")


def performance(pred, truth):
    pred = np.array(pred.transpose(0, 1))  # [12, b]
    truth = np.array(truth.transpose(0, 1))  # [12, b]
    # sample_num = pred.shape[1]

    rela_f1, rela_p, rela_r = [], [], []
    for i in range(12):
        pred_i = pred[i]
        truth_i = truth[i]

        truth_i_rela = truth_i[truth_i != 0]
        if len(truth_i_rela) == 0:
            rela_f1.append(-1)
            rela_p.append(-1)
            rela_r.append(-1)
            continue

        rela_f1.append(round(f1_score(truth_i, pred_i, zero_division=0), 4))
        rela_p.append(round(precision_score(truth_i, pred_i, zero_division=0), 4))
        rela_r.append(round(recall_score(truth_i, pred_i, zero_division=0), 4))

    avg_rela_f1 = np.mean([x for x in rela_f1 if x >= 0])

    pred = pred.reshape(-1)
    truth = truth.reshape(-1)

    truth_rela = truth[truth != 0]
    if len(truth_rela) == 0:
        global_rela_f1 = -1
        global_rela_p = -1
        global_rela_r = -1
    else:
        global_rela_f1 = f1_score(truth, pred, zero_division=0)
        global_rela_p = precision_score(truth, pred, zero_division=0)
        global_rela_r = recall_score(truth, pred, zero_division=0)

    result_dict = {'avg_rela_f1': avg_rela_f1, 'global_rela_f1': global_rela_f1,
                   'global_rela_p': global_rela_p, 'global_rela_r': global_rela_r,
                   'rela_f1': rela_f1, 'rela_p': rela_p, 'rela_r': rela_r}

    return result_dict


if __name__ == '__main__':
    train()

