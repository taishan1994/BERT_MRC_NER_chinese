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
    parser.add_argument("--saved_model", type=str, default="bert_finetune_model_v2.bin")
    parser.add_argument("--task_name", default=None, type=str)
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--dev_batch_size", default=16, type=int)
    parser.add_argument("--test_batch_size", default=2, type=int)
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
    train_dataloader = dataset_loaders.get_dataloader(data_sign="train")
    dev_dataloader = dataset_loaders.get_dataloader(data_sign="dev")
    test_dataloader = dataset_loaders.get_dataloader(data_sign="test")
    num_train_steps = dataset_loaders.get_num_train_epochs()
    return train_dataloader, dev_dataloader, test_dataloader, num_train_steps, label_list


def load_model(config):
    model = BertQueryNER(config)
    model.to(config.device)
    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=10e-8)
    sheduler = None

    return model, optimizer, sheduler


def train(config):
    device = config.device
    model, optimizer, sheduler = load_model(config)
    tr_loss = 0.0
    nb_tr_examples = 0
    nb_tr_steps = 0

    dev_best_acc = 0.0
    dev_best_precision = 0.0
    dev_best_recall = 0.0
    dev_best_f1 = 0.0
    dev_best_loss = float("inf")

    model.train()
    for idx in range(int(config.num_train_epochs)):
        train_dataloader, \
        dev_dataloader, text_dataloader, \
        num_train_steps, label_list = load_data(config)
        print("#######" * 10)
        print("EPOCH: ", str(idx))
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, start_pos, end_pos, ner_cate = batch
            loss = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                         start_positions=start_pos, end_positions=end_pos)
            model.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)
            optimizer.step()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if nb_tr_steps > 0 and nb_tr_steps % config.checkpoint == 0:
                print("-*-" * 15)
                # print("current training loss is : ")
                # print(loss.item())
                print("【train】 epoch:{}/{} step:{}/{} loss:{}".format(
                    idx, config.num_train_epochs, nb_tr_steps, num_train_steps, loss.item()
                ))
                tmp_dev_loss, tmp_dev_acc, tmp_dev_prec, tmp_dev_rec, tmp_dev_f1 = eval(model,
                                                                                        dev_dataloader,
                                                                                        config,
                                                                                        label_list,
                                                                                        )
                print("......" * 10)
                print("DEV: loss, acc, precision, recall, f1")
                print(tmp_dev_loss, tmp_dev_acc, tmp_dev_prec, tmp_dev_rec, tmp_dev_f1)

                if tmp_dev_f1 > dev_best_f1:
                    dev_best_acc = tmp_dev_acc
                    dev_best_loss = tmp_dev_loss
                    dev_best_precision = tmp_dev_prec
                    dev_best_recall = tmp_dev_rec
                    dev_best_f1 = tmp_dev_f1

                    # export model
                    if config.export_model:
                        model_to_save = model.module if hasattr(model, "module") else model
                        output_model_file = os.path.join(config.output_dir,
                                                         config.saved_model)
                        torch.save(model_to_save.state_dict(), output_model_file)
                        print("SAVED model path is :")
                        print(output_model_file)

                    # tmp_test_loss, tmp_test_acc, tmp_test_prec, tmp_test_rec, tmp_test_f1 = eval_checkpoint(model, dev_dataloader, config, device, n_gpu, label_list, eval_sign="test")
                    # print("......"*10)
                    # print("TEST: loss, acc, precision, recall, f1")
                    # print(tmp_test_loss, tmp_test_acc, tmp_test_prec, tmp_test_rec, tmp_test_f1)

                    # test_acc_when_dev_best = tmp_test_acc
                    # test_pre_when_dev_best = tmp_test_prec
                    # test_rec_when_dev_best = tmp_test_rec
                    # test_f1_when_dev_best = tmp_test_f1
                    # test_loss_when_dev_best = tmp_test_loss

                    print("-*-" * 15)

                    print("=&=" * 15)
                    print("Best DEV : overall best loss, acc, precision, recall, f1 ")
                    print(dev_best_loss, dev_best_acc, dev_best_precision, dev_best_recall, dev_best_f1)
                    # print("scores on TEST when Best DEV:loss, acc, precision, recall, f1 ")
                    # print(test_loss_when_dev_best, test_acc_when_dev_best, test_pre_when_dev_best, test_rec_when_dev_best, test_f1_when_dev_best)
                    print("=&=" * 15)


def eval(model, eval_dataloader, config, label_list):
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

    for input_ids, input_mask, segment_ids, start_pos, end_pos, ner_cate in eval_dataloader:
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
    # load_data(args_config)
    # model, optimizer, sheduler = load_model(args_config)
    train(args_config)
