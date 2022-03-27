import argparse
import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, AdamW

from data_loader import MRCNERDataLoader
from model import BertQueryNER
from metric.mrc_ner_evaluate import flat_ner_performance, nested_ner_performance


def args_parser():
    # start parser
    parser = argparse.ArgumentParser()

    # requires parameters
    parser.add_argument("--config_path", default="./pretrained_model/hfl_chinese-roberta-wwm-ext/config.json", type=str)
    parser.add_argument("--data_dir", default='./data/zh_msra/', type=str)
    parser.add_argument("--bert_model", default="./pretrained_model/hfl_chinese-roberta-wwm-ext/", type=str, )
    parser.add_argument("--task_name", default=None, type=str)
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--train_batch_size", default=2, type=int)
    parser.add_argument("--dev_batch_size", default=2, type=int)
    parser.add_argument("--test_batch_size", default=32, type=int)
    parser.add_argument("--checkpoint", default=500, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument("--export_model", type=bool, default=True)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--data_sign", type=str, default="MSRA")
    parser.add_argument("--weight_start", type=float, default=1.0)
    parser.add_argument("--weight_end", type=float, default=1.0)
    parser.add_argument("--weight_span", type=float, default=1.0)
    parser.add_argument("--entity_sign", type=str, default="flat")  # nested flat
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--entity_threshold", type=float, default=0.5)
    parser.add_argument("--data_cache", type=bool, default=True)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--clip_grad", type=int, default=1)
    parser.add_argument("--saved_model", type=str, default="bert_finetune_model_v2.bin")

    args = parser.parse_args()

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    return args


def load_data(config):
    print("-*-" * 10)
    if config.data_sign == 'MSRA':
        print("current data_sign: {}".format(config.data_sign))
        with open(os.path.join(config.data_dir, 'zh_msra.json'), 'r') as fp:
            label_list = json.loads(fp.read())['labels']
    # 这里额外需要加入O
    label_list = label_list + ['O']
    tokenizer = BertTokenizer.from_pretrained(config.bert_model)
    dataset_loaders = MRCNERDataLoader(config, label_list, tokenizer, mode="train", )
    test_dataloader = dataset_loaders.get_dataloader(data_sign="test")
    return test_dataloader, label_list


def load_model(config):
    model = BertQueryNER(config)
    checkpoint = torch.load(os.path.join(config.output_dir, config.saved_model))
    model.load_state_dict(checkpoint)
    model.to(config.device)
    return model


def test(model, test_dataloader, config, label_list):
    device = config.device
    model.eval()
    eval_loss = 0

    start_pred_lst = []
    end_pred_lst = []

    mask_lst = []
    start_gold_lst = []
    end_gold_lst = []

    eval_steps = 0
    ner_cate_lst = []

    for input_ids, input_mask, segment_ids, start_pos, end_pos, ner_cate in test_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        start_pos = start_pos.to(device)
        end_pos = end_pos.to(device)

        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, start_pos, end_pos)
            start_logits, end_logits = model(input_ids, segment_ids, input_mask)

        start_pos = start_pos.to("cpu").numpy().tolist()
        end_pos = end_pos.to("cpu").numpy().tolist()

        start_label = start_logits.detach().cpu().numpy().tolist()
        end_label = end_logits.detach().cpu().numpy().tolist()

        input_mask = input_mask.to("cpu").detach().numpy().tolist()

        ner_cate_lst += ner_cate.numpy().tolist()
        eval_loss += tmp_eval_loss.mean().item()
        mask_lst += input_mask
        eval_steps += 1

        start_pred_lst += start_label
        end_pred_lst += end_label

        start_gold_lst += start_pos
        end_gold_lst += end_pos

    span_pred_lst = [[[1] * len(start_pred_lst[0])] * len(start_pred_lst[0])] * len(start_pred_lst)
    span_gold_lst = [[[1] * len(start_gold_lst[0])] * len(start_gold_lst[0])] * len(start_gold_lst)

    if config.entity_sign == "flat":
        eval_accuracy, eval_precision, eval_recall, eval_f1, pred_bmes_label_lst = flat_ner_performance(start_pred_lst,
                                                                                                        end_pred_lst,
                                                                                                        span_pred_lst,
                                                                                                        start_gold_lst,
                                                                                                        end_gold_lst,
                                                                                                        span_gold_lst,
                                                                                                        ner_cate_lst,
                                                                                                        label_list,
                                                                                                        threshold=config.entity_threshold,
                                                                                                        dims=2)
    else:
        eval_accuracy, eval_precision, eval_recall, eval_f1, pred_bmes_label_lst = nested_ner_performance(
            start_pred_lst, end_pred_lst, span_pred_lst, start_gold_lst, end_gold_lst, span_gold_lst, ner_cate_lst,
            label_list, threshold=config.entity_threshold, dims=2)

    average_loss = round(eval_loss / eval_steps, 4)
    eval_f1 = round(eval_f1, 4)
    eval_precision = round(eval_precision, 4)
    eval_recall = round(eval_recall, 4)
    eval_accuracy = round(eval_accuracy, 4)

    return average_loss, eval_accuracy, eval_precision, eval_recall, eval_f1


if __name__ == '__main__':
    args_config = args_parser()
    args_config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(args_config)
    test_dataloader, label_list = load_data(args_config)
    model = load_model(args_config)
    loss, acc, pre, rec, f1 = test(model, test_dataloader, args_config, label_list)
    print('【test】 loss:{} accuracy:{} precision:{} recall:{} f1:{}'.format(
        loss, acc, pre, rec, f1
    ))
