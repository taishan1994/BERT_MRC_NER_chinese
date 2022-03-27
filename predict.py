import argparse
import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, AdamW

from data_loader import MRCNERDataLoader
from model import BertQueryNER
from metric.mrc_ner_evaluate import flat_ner_decode
from mrc_utils import InputExample, InputFeatures
from get_entity import get_entities
from tokenization import BasicTokenizer

basicTokenizer = BasicTokenizer(do_lower_case=True)


def args_parser():
    # start parser
    parser = argparse.ArgumentParser()

    # requires parameters
    parser.add_argument("--bert_model", default="./pretrained_model/hfl_chinese-roberta-wwm-ext/", type=str, )
    parser.add_argument("--data_dir", default='./data/zh_msra/', type=str)
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--predict_batch_size", default=3, type=int)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--entity_threshold", type=float, default=0.5)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--saved_model", type=str, default="bert_finetune_model_v2.bin")
    parser.add_argument("--weight_start", type=float, default=1.0)
    parser.add_argument("--weight_end", type=float, default=1.0)
    parser.add_argument("--weight_span", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--data_sign", type=str, default="MSRA")
    parser.add_argument("--entity_sign", type=str, default="flat")

    args = parser.parse_args()

    return args


def get_examples(texts, config):
    if isinstance(texts, str):
        texts = [texts]
    with open(os.path.join(config.data_dir, 'zh_msra.json'), 'r') as fp:
        query = json.loads(fp.read())['default']
    examples = []
    for text in texts:
        for k, q in query.items():
            examples.append(
                InputExample(
                    query_item=q,
                    context_item=text,
                    ner_cate=k
                )
            )
            # print(q, text, k)
    return examples


def predict_convert_examples_to_features(examples, tokenizer, label_lst, max_seq_length, pad_sign=True):
    label_map = {tmp: idx for idx, tmp in enumerate(label_lst)}
    features = []
    total = len(examples)
    contents = []
    for (example_idx, example) in enumerate(examples):
        print(example_idx + 1, total)
        whitespace_doc = basicTokenizer.tokenize(example.context_item)
        query_tokens = tokenizer.tokenize(example.query_item)

        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
        all_doc_tokens = []

        for token_item in whitespace_doc:
            tmp_subword_lst = tokenizer.tokenize(token_item)
            all_doc_tokens.extend(tmp_subword_lst)

        if len(all_doc_tokens) >= max_tokens_for_doc:
            all_doc_tokens = all_doc_tokens[: max_tokens_for_doc]

        contents.append(['[CLS]'] + query_tokens + ['[SEP]'] + all_doc_tokens + ['[SEP]'])
        input_tokens = []
        segment_ids = []
        input_mask = []

        input_tokens.append("[CLS]")
        segment_ids.append(0)
        for query_item in query_tokens:
            input_tokens.append(query_item)
            segment_ids.append(0)
        input_tokens.append("[SEP]")
        segment_ids.append(0)
        input_mask.append(1)
        input_tokens.extend(all_doc_tokens)
        segment_ids.extend([1] * len(all_doc_tokens))
        input_tokens.append("[SEP]")
        segment_ids.append(1)
        input_mask = [1] * len(input_tokens)

        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)

        # zero-padding up to the sequence length
        if len(input_ids) < max_seq_length and pad_sign:
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
        if example_idx < 3:
            print("example.context_item:", example.context_item)
            print("tokens:", input_tokens)
            print("input_ids:", input_ids)
            print("input_mask:", input_mask)
            print("segment_ids:", segment_ids)
            print("input_ids:", input_ids)
            print("ner_cate:", example.ner_cate)
        features.append(
            InputFeatures(
                tokens=input_tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                ner_cate=label_map[example.ner_cate]
            ))

    return features, contents


def predict_loader(texts, tokenizer, config):
    print("-*-" * 10)
    if config.data_sign == 'MSRA':
        print("current data_sign: {}".format(config.data_sign))
        with open(os.path.join(config.data_dir, 'zh_msra.json'), 'r') as fp:
            label_list = json.loads(fp.read())['labels']
    # 这里额外需要加入O
    label_list = label_list + ['O']
    examples = get_examples(texts, config)
    features, contents = predict_convert_examples_to_features(
        examples, tokenizer, label_list, config.max_seq_length
    )
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    ner_cate = torch.tensor([f.ner_cate for f in features], dtype=torch.long)
    dataset = TensorDataset(input_ids, input_mask, segment_ids, ner_cate)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=config.predict_batch_size, num_workers=2)
    return dataloader, label_list, contents


def load_model(config):
    model = BertQueryNER(config)
    checkpoint = torch.load(os.path.join(config.output_dir, config.saved_model))
    model.load_state_dict(checkpoint)
    model.to(config.device)
    return model


def predict(model, predict_dataloader, config, label_list, tokenizer, contents):
    device = config.device
    model.eval()

    start_pred_lst = []
    end_pred_lst = []

    mask_lst = []

    eval_steps = 0
    ner_cate_lst = []

    for input_ids, input_mask, segment_ids, ner_cate in predict_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        with torch.no_grad():
            start_logits, end_logits = model(input_ids, segment_ids, input_mask)

        start_label = start_logits.detach().cpu().numpy().tolist()
        end_label = end_logits.detach().cpu().numpy().tolist()

        input_mask = input_mask.to("cpu").detach().numpy().tolist()

        ner_cate_lst += ner_cate.numpy().tolist()
        mask_lst += input_mask
        eval_steps += 1

        start_pred_lst += start_label
        end_pred_lst += end_label

    span_pred_lst = [[[1] * len(start_pred_lst[0])] * len(start_pred_lst[0])] * len(start_pred_lst)

    if config.entity_sign == "flat":
        pred_bmes_label_lst = flat_ner_decode(start_pred_lst,
                                              end_pred_lst,
                                              span_pred_lst,
                                              ner_cate_lst,
                                              label_list,
                                              threshold=config.entity_threshold,
                                              dims=2)
        for content, pre in zip(contents, pred_bmes_label_lst):
            text = content[content.index('[SEP]') + 1:-1]
            chunks = get_entities(pre, content)
            print("".join(text), chunks)


if __name__ == '__main__':
    args_config = args_parser()
    args_config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(args_config)
    model = load_model(args_config)
    texts = [
        # "去年，我们又被评为“北京市首届家庭藏书状元明星户",
        # "每当接到海外书友或归国人员汇至一册精美>的食品图谱时，全家人欣喜若狂；而一册交换品离我而去，虽为复本，也大有李后主“挥泪别宫娥”之感。",
        # "我们是受到郑>振铎先生、阿英先生著作的启示，从个人条件出发，瞄准现代出版史研究的空白，重点集藏解放区、国民党毁禁出版物。",
        # "我们变而以书会友，以书结缘，把欧美、港台流行的食品类图谱、画册、工具书汇集一堂。",
        "我们藏有一册１９４５年６月油印的《北京文物保>存保管状态之调查报告》，调查范围涉及故宫、历博、古研所、北大清华图书馆、北图、日伪资料库等二十几家，言及文物二十>万件以上，洋洋三万余言，是珍贵的北京史料。",

    ]
    tokenizer = BertTokenizer.from_pretrained(args_config.bert_model)
    dataloader, label_list, contents = predict_loader(texts, tokenizer, args_config)
    predict(model, dataloader, args_config, label_list, tokenizer, contents)
