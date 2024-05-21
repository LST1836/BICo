# -*- coding: utf-8 -*-
import argparse
import datetime
import json
import os


def str2list(x):
    results = []
    for x in x.split(","):
        x = x.strip()
        try:
            x = eval(x)
        except:
            raise ValueError
        results.append(x)
    return results


def parse_relevance_args():
    parser = argparse.ArgumentParser(description='MID-RelevanceRecognition')

    parser.add_argument('--train_data_path', default='', type=str, required=True)
    parser.add_argument('--val_data_path', default='', type=str, required=True)
    parser.add_argument('--test_data_path', default='', type=str, required=True)
    parser.add_argument('--plm_path', default='', type=str, required=True)
    parser.add_argument('--cls_num', default=2, type=int)
    parser.add_argument('--log_dir', default='log', type=str)

    parser.add_argument('--max_seq_length', default=128, type=int)
    parser.add_argument('--concept_tree', default='True', type=eval)
    parser.add_argument('--num_iter_hie_tree', default=4, type=int)
    parser.add_argument('--mh_matching', default='False', type=eval)
    parser.add_argument('--num_head_matching', default=8, type=int)

    parser.add_argument('--gpu_ids', default='0', type=str)
    parser.add_argument('--seed', default=2023, type=int)
    parser.add_argument('--pos_weight', default='30,15,6,6,2,40,25,3,3,3,1.5,1.5')
    parser.add_argument('--weight_scale', default=0.4, type=float)
    parser.add_argument('--t1', default='0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5', type=str2list)
    parser.add_argument('--cl_loss_weight', default='0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3', type=str2list)
    parser.add_argument('--lr_text_encoder', default=2e-5, type=float)
    parser.add_argument('--lr_other', default=1e-3, type=float)
    parser.add_argument('--wd_plm', default=0.01, type=float)
    parser.add_argument('--wd_other', default=5e-4, type=float)
    parser.add_argument('--dropout_mlp', default=0.2, type=float)
    parser.add_argument('--act_mlp', default='leaky_relu', type=str)
    parser.add_argument('--warmup', default='False', type=eval)
    parser.add_argument('--warm_ratio', default=0.06, type=float)
    parser.add_argument('--num_epoch', default=15, type=int)
    parser.add_argument('--batch_size', default=64, type=int)

    args = parser.parse_args()

    ts = datetime.datetime.now().strftime('%m_%d_%H_%M_%S')
    args.log_dir = os.path.join(args.log_dir, ts)
    os.makedirs(args.log_dir, exist_ok=True)

    # save config
    with open(os.path.join(args.log_dir, "train.config"), "w") as f:
        json.dump(vars(args), f)
    f.close()

    return args


def parse_ideology_args():
    parser = argparse.ArgumentParser(description='MID-IdeologyAnalysis')

    parser.add_argument('--train_data_path', default='', type=str, required=True)
    parser.add_argument('--val_data_path', default='', type=str, required=True)
    parser.add_argument('--test_data_path', default='', type=str, required=True)
    parser.add_argument('--plm_path', default='', type=str, required=True)
    parser.add_argument('--cls_num', default=3, type=int)
    parser.add_argument('--log_dir', default='log', type=str)

    parser.add_argument('--max_seq_length', default=128, type=int)
    parser.add_argument('--concept_tree', default='True', type=eval)
    parser.add_argument('--num_iter_hie_tree', default=2, type=int)
    parser.add_argument('--mh_matching', default='False', type=eval)
    parser.add_argument('--num_head_matching', default=8, type=int)

    parser.add_argument('--gpu_ids', default='0', type=str)
    parser.add_argument('--seed', default=2023, type=int)
    parser.add_argument('--t2', default='0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1', type=str2list)
    parser.add_argument('--cl_loss_weight', default='0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3', type=str2list)
    parser.add_argument('--lr_text_encoder', default=2e-5, type=float)
    parser.add_argument('--lr_other', default=1e-3, type=float)
    parser.add_argument('--wd_plm', default=0.01, type=float)
    parser.add_argument('--wd_other', default=5e-4, type=float)
    parser.add_argument('--dropout_mlp', default=0.2, type=float)
    parser.add_argument('--act_mlp', default='leaky_relu', type=str)
    parser.add_argument('--warmup', default='False', type=eval)
    parser.add_argument('--warm_ratio', default=0.06, type=float)
    parser.add_argument('--num_epoch', default=15, type=int)
    parser.add_argument('--batch_size', default=64, type=int)

    args = parser.parse_args()

    ts = datetime.datetime.now().strftime('%m_%d_%H_%M_%S')
    args.log_dir = os.path.join(args.log_dir, ts)
    os.makedirs(args.log_dir, exist_ok=True)

    # save config
    with open(os.path.join(args.log_dir, "train.config"), "w") as f:
        json.dump(vars(args), f)
    f.close()

    return args
