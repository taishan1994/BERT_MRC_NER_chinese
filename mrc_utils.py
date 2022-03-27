"""
将数据转换为所需要的
"""
import json
import os

from tokenization import BasicTokenizer

basicTokenizer = BasicTokenizer(do_lower_case=True)


class InputExample(object):
    def __init__(self,
                 query_item,
                 context_item,
                 doc_tokens=None,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 ner_cate=None):
        self.query_item = query_item
        self.context_item = context_item
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.ner_cate = ner_cate


class InputFeatures(object):
    """
    Desc:
        a single set of features of data
    Args:
        start_pos: start position is a list of symbol
        end_pos: end position is a list of symbol
    """

    def __init__(self,
                 tokens,
                 input_ids,
                 input_mask,
                 segment_ids,
                 ner_cate,
                 start_position=None,
                 end_position=None):
        self.tokens = tokens
        self.input_mask = input_mask
        self.input_ids = input_ids
        self.ner_cate = ner_cate
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position


def convert_examples_to_features(examples, tokenizer, label_lst, max_seq_length, pad_sign=True):
    label_map = {tmp: idx for idx, tmp in enumerate(label_lst)}
    features = []
    total = len(examples)
    for (example_idx, example) in enumerate(examples):
        # print(example_idx+1, total)
        query_tokens = tokenizer.tokenize(example.query_item)
        whitespace_doc = basicTokenizer.tokenize(example.context_item)
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        if len(example.start_position) == 0 and len(example.end_position) == 0:
            all_doc_tokens = []

            for token_item in whitespace_doc:
                tmp_subword_lst = tokenizer.tokenize(token_item)
                all_doc_tokens.extend(tmp_subword_lst)
            doc_start_pos = [0] * len(all_doc_tokens)
            doc_end_pos = [0] * len(all_doc_tokens)

        else:
            doc_start_pos = []
            doc_end_pos = []

            all_doc_tokens = []
            offset_idx_dict = {}

            fake_start_pos = [0] * len(whitespace_doc)
            fake_end_pos = [0] * len(whitespace_doc)
            for start_item in example.start_position:
                fake_start_pos[int(start_item)] = 1
            for end_item in example.end_position:
                # if int(end_item) >= len(fake_end_pos):
                #     print("stop")
                fake_end_pos[int(end_item)] = 1

            # improve answer span
            for idx, (token, start_label, end_label) in enumerate(zip(whitespace_doc, fake_start_pos, fake_end_pos)):
                tmp_subword_lst = tokenizer.tokenize(token)

                if len(tmp_subword_lst) > 1:
                    offset_idx_dict[idx] = len(all_doc_tokens)

                    doc_start_pos.append(start_label)
                    doc_start_pos.extend([0] * (len(tmp_subword_lst) - 1))

                    doc_end_pos.extend([0] * (len(tmp_subword_lst) - 1))
                    doc_end_pos.append(end_label)

                    all_doc_tokens.extend(tmp_subword_lst)
                elif len(tmp_subword_lst) == 1:
                    offset_idx_dict[idx] = len(all_doc_tokens)
                    doc_start_pos.append(start_label)
                    doc_end_pos.append(end_label)
                    all_doc_tokens.extend(tmp_subword_lst)
                else:
                    raise ValueError("Please check the result of tokenizer !!! !!! ")

            # for span_item in example.span_position:
            #     s_idx, e_idx = span_item.split(";")
            #     e_idx = int(e_idx) - 1
            #     if len(query_tokens)+2+offset_idx_dict[int(s_idx)] <= max_tokens_for_doc and \
            #     len(query_tokens)+2+offset_idx_dict[int(e_idx)] <= max_tokens_for_doc :
            #         doc_span_pos[len(query_tokens)+2+offset_idx_dict[int(s_idx)]][len(query_tokens)+2+offset_idx_dict[int(e_idx)]] = 1
            #     else:
            #         continue

        assert len(all_doc_tokens) == len(doc_start_pos)
        assert len(all_doc_tokens) == len(doc_end_pos)
        assert len(doc_start_pos) == len(doc_end_pos)

        if len(all_doc_tokens) >= max_tokens_for_doc:
            all_doc_tokens = all_doc_tokens[: max_tokens_for_doc]
            doc_start_pos = doc_start_pos[: max_tokens_for_doc]
            doc_end_pos = doc_end_pos[: max_tokens_for_doc]
        # if len(example.start_position) == 0 and len(example.end_position) == 0:
        #     doc_span_pos = np.zeros((max_seq_length, max_seq_length), dtype=int)

        # input_mask:
        #   the mask has 1 for real tokens and 0 for padding tokens.
        #   only real tokens are attended to.
        # segment_ids:
        #   segment token indices to indicate first and second portions of the inputs.
        input_tokens = []
        segment_ids = []
        input_mask = []
        start_pos = []
        end_pos = []

        input_tokens.append("[CLS]")
        segment_ids.append(0)
        start_pos.append(0)
        end_pos.append(0)

        for query_item in query_tokens:
            input_tokens.append(query_item)
            segment_ids.append(0)
            start_pos.append(0)
            end_pos.append(0)

        input_tokens.append("[SEP]")
        segment_ids.append(0)
        input_mask.append(1)
        start_pos.append(0)
        end_pos.append(0)

        input_tokens.extend(all_doc_tokens)
        segment_ids.extend([1] * len(all_doc_tokens))
        start_pos.extend(doc_start_pos)
        end_pos.extend(doc_end_pos)

        input_tokens.append("[SEP]")
        segment_ids.append(1)
        start_pos.append(0)
        end_pos.append(0)
        input_mask = [1] * len(input_tokens)

        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)

        # zero-padding up to the sequence length
        if len(input_ids) < max_seq_length and pad_sign:
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            start_pos += padding
            end_pos += padding

        if example_idx < 3:
            print("example.context_item:", example.context_item)
            print("tokens:", input_tokens)
            print("input_ids:", input_ids)
            print("input_mask:", input_mask)
            print("segment_ids:", segment_ids)
            print("input_ids:", input_ids)
            print("start_position:", start_pos)
            print("end_position:", end_pos)
            print("ner_cate:", example.ner_cate)

        features.append(
            InputFeatures(
                tokens=input_tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                start_position=start_pos,
                end_position=end_pos,
                ner_cate=label_map[example.ner_cate]
            ))

    return features


def read_mrc_ner_examples(path, input_file):
    """
    Desc:
        read MRC-NER data
    """

    with open(os.path.join(path, input_file), "r", encoding='utf-8') as f:
        input_data = json.load(f)

    # input_data = input_data[:10]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        query_item = entry["query"]
        context_item = entry["context"]
        context_item = "".join(context_item.split(" "))
        start_position = entry["start_position"]
        end_position = entry["end_position"]
        ner_cate = entry["entity_label"]

        example = InputExample(
            query_item=query_item,
            context_item=context_item,
            start_position=start_position,
            end_position=end_position,
            ner_cate=ner_cate)
        examples.append(example)
    print(len(examples))
    return examples


def read_one():
    input_data = [
        {
            "context": "我 们 藏 有 一 册 １ ９ ４ ５ 年 ６ 月 油 印 的 《 北 京 文 物 保 存 保 管 状 态 之 调 查 报 告 》 ， 调 查 范 围 涉 及 故 宫 、 历 博 、 古 研 所 、 北 >大 清 华 图 书 馆 、 北 图 、 日 伪 资 料 库 等 二 十 几 家 ， 言 及 文 物 二 十 万 件 以 上 ， 洋 洋 三 万 余 言 ， 是 珍 贵 的 北 京 史 料 。",
            "entity_label": "NS",
            "query": "按照地理位置划分的国家,城市,乡镇,大洲",
            "start_position": [14, 37, 40, 47, 55, 58, 91],
            "end_position": [15, 38, 41, 53, 56, 58, 92]
        }
    ]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        query_item = entry["query"]
        context_item = entry["context"]
        context_item = "".join(context_item.split(" "))
        start_position = entry["start_position"]
        end_position = entry["end_position"]
        ner_cate = entry["entity_label"]

        example = InputExample(
            query_item=query_item,
            context_item=context_item,
            start_position=start_position,
            end_position=end_position,
            ner_cate=ner_cate)
        examples.append(example)
    print(len(examples))
    return examples


if __name__ == '__main__':
    # examples = read_mrc_ner_examples('data/zh_msra/', 'train.mrc')
    examples = read_one()
    label_lst = ['NS', 'NR', 'NT']
    from transformers import BertTokenizer

    max_seq_length = 256
    tokenizer = BertTokenizer.from_pretrained('pretrained_model/hfl_chinese-roberta-wwm-ext/')
    convert_examples_to_features(examples, tokenizer, label_lst, max_seq_length)
