"""
用于处理数据为所需要的格式
"""
import json
from tokenization import BasicTokenizer

basicTokenizer = BasicTokenizer(do_lower_case=True)


class MSRAProcessor:
    def __init__(self):
        with open('data/zh_msra/zh_msra.json', 'r') as fp:
            labels = json.loads(fp.read())
        self.query2label = {}
        self.label2query = {}
        query = labels['default']
        for k, v in query.items():
            self.query2label[v] = k
            self.label2query[k] = v
        self.rlabel = labels['labels']

    def get_mid_data(self, in_path, out_path):
        with open(in_path, 'r') as fp:
            data = fp.read()
        data = data.split('\n')
        text = ''
        entity = []
        i = 0
        start = 0
        end = 0
        tmp = ''
        res = []
        for d in data:
            d = d.strip().split(' ')
            if len(d) == 2:
                text += d[0]
                if 'B-' in d[1]:
                    start = i
                    end = i
                    tmp += d[0]
                elif 'M-' in d[1]:
                    tmp += d[0]
                    end += 1
                elif 'E-' in d[1]:
                    e = d[1].split('-')[-1]
                    tmp += d[0]
                    end += 1
                    entity.append([tmp, e, start, end])
                    tmp = ''
                elif 'S-' in d[1]:
                    e = d[1].split('-')[-1]
                    entity.append([d[0], e, i, i])
                i += 1
            else:
                res.append(
                    {
                        "text": text,
                        'entity': entity,
                    }
                )
                text = ''
                entity = []
                i = 0
                start = 0
                end = 0
                tmp = ''
        else:
            i += 1

        with open(out_path, 'w') as fp:
            json.dump(res, fp, ensure_ascii=False)

    def get_mrc_data(self, in_path, out_path):
        with open(in_path, 'r') as fp:
            data = json.loads(fp.read())
        res = []
        for j,d in enumerate(data):
            text = d['text']
            entity = d['entity']
            if not entity:
                continue
            t_entity = []
            entity.sort(key=lambda x: x[2])
            for e in entity:
                left = text[:e[2]]
                t_left = basicTokenizer.tokenize(left)
                t_e = basicTokenizer.tokenize(e[0])
                t_entity.append([e[0], e[1], len(t_left), len(t_left) + len(t_e) - 1])
            for rl in self.rlabel:
                start_position = []
                end_position = []
                for i in t_entity:
                    if i[1] == rl:
                        start_position.append(i[2])
                        end_position.append(i[3])
                if j < 3:
                    print("="*20)
                    print("context:", " ".join(text))
                    print("entity_label:", rl)
                    print("query:", self.label2query[rl])
                    print("start_position:", start_position)
                    print("end_position:", end_position)
                    print("=" * 20)
                res.append(
                    {
                        "context": " ".join(text),
                        "entity_label": rl,
                        "query": self.label2query[rl],
                        "start_position": start_position,
                        "end_position": end_position,
                    }
                )
        with open(out_path, 'w') as fp:
            json.dump(res, fp, ensure_ascii=False)


if __name__ == '__main__':
    msraProcessor = MSRAProcessor()
    # 第一步：先生成中间文件和标签
    # msraProcessor.get_mid_data('data/zh_msra/train.char.bmes',
    #                            'data/zh_msra/train.mid')
    # msraProcessor.get_mid_data('data/zh_msra/dev.char.bmes',
    #                            'data/zh_msra/dev.mid')
    # msraProcessor.get_mid_data('data/zh_msra/test.char.bmes',
    #                            'data/zh_msra/test.mid')
    # 第二步：生成MRC所需要的数据
    msraProcessor.get_mrc_data('data/zh_msra/train.mid',
                               'data/zh_msra/train.mrc')
    msraProcessor.get_mrc_data('data/zh_msra/dev.mid',
                               'data/zh_msra/dev.mrc')
    msraProcessor.get_mrc_data('data/zh_msra/test.mid',
                               'data/zh_msra/test.mrc')
