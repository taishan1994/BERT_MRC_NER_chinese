import torch
from transformers import BertTokenizer

cache_path = './data/zh_msra/mrc-ner.train.cache.256'
bert_model = './pretrained_model/hfl_chinese-roberta-wwm-ext/'
# features = torch.load(cache_path)

tokenizer = BertTokenizer.from_pretrained(bert_model)
print(tokenizer.tokenize("１９４５"))
# print(len(features))
# for i,f in enumerate(features):
#     entity = []
#     if i < 12:
#         print('====================')
#         print(tokenizer.decode(f.input_ids))
#         print(f.start_position)
#         print(f.end_position)
#         print(f.ner_cate)
#         print('====================')

from tokenization import BasicTokenizer
basicTokenizer = BasicTokenizer(do_lower_case=True)
text = "我们藏有一册１９４５年６月油印的《北京文物保>存保管状态之调查报告》，调查范围涉及故宫、历博、古研所、北大清华图书馆、北图、日伪资料库等二十几家，言及文物二十>万件以上，洋洋三万余言，是珍贵的北京史料。"
print(basicTokenizer.tokenize(text))
