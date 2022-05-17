# BERT_MRC_NER_chinese
基于bert_mrc的中文命名实体识别，使用的数据集是MSRA，预训练模型是roberta。数据和训练好的模型：
链接：https://pan.baidu.com/s/1XzoZDaZdmtwrEENBNseX5g <br>
提取码：21v3
	
# 训练和验证
```python
python train.py
```

# 测试
```python
python test.py

【test】 loss:0.0023 accuracy:0.9339 precision:0.9558 recall:0.9203 f1:0.9377
```

# 预测
```python
python predict.py

我们藏有一册１##９##４##５年６月油印的《北京文物保>存保管状态之调查报告》，调查范围涉及故宫、历博、古研所、北大清华图书馆、北图、日伪资料库等二十几家，言及文物二十>万件以上，洋洋三万余言，是珍贵的北京史料。 []

我们藏有一册１##９##４##５年６月油印的《北京文物保>存保管状态之调查报告》，调查范围涉及故宫、历博、古研所、北大清华图书馆、北图、日伪资料库等二十几家，言及文物二十>万件以上，洋洋三万余言，是珍贵的北京史料。 [(['北', '京'], 39, 'NS'), (['故', '宫'], 63, 'NS'), (['北', '大', '清', '华', '图', '书', '馆'], 73, 'NS'), (['北', '图'], 81, 'NS'), (['日'], 84, 'NS'), (['北', '京'], 118, 'NS')]

我们藏有一册１##９##４##５年６月油印的《北京文物保>存保管状态之调查报告》，调查范围涉及故宫、历博、古研所、北大清华图书馆、北图、日伪资料库等二十几家，言及文物二十>万件以上，洋洋三万余言，是珍贵的北京史料。 []
```

# 感谢
> https://github.com/JavaStudenttwo/BERT_MRC <br>
> https://github.com/ShannonAI/mrc-for-flat-nested-ner

# 补充
[信息抽取三剑客：实体抽取、关系抽取、事件抽取](https://github.com/taishan1994/chinese_information_extraction)<br>
[pytorch_bert_bilstm_crf命名实体识别](https://github.com/taishan1994/pytorch_bert_bilstm_crf_ner)<br>
[W2NER：实体识别新sota](https://github.com/taishan1994/W2NER_predict/)<br>
[针对于基于阅读理解的实体识别每次只能预测一种实体，并且需要进行重复编码，ACL有这么一个模型被提出解决这些问题](https://github.com/tricktreat/piqn)
