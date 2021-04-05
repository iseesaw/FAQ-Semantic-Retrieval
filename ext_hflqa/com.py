import json
import jieba

# dct = json.load(open('./faq.json', encoding='utf8'))
# add = json.load(open('./add_faq.json', encoding='utf8'))
train = json.load(open('./add_faq.json', encoding='utf8'))

outputs = {}
for idx, (key, value) in enumerate(train.items()):
    outputs[str(idx)] = {'q': [' '.join(jieba.lcut(key))], 'a': value}
    #outputs[str(idx)] = {'q':[' '.join(jieba.lcut(p)) for p in value['post']], 'a': value['resp']}
json.dump(outputs, open('./add_faq_.json', 'w', encoding='utf8'), ensure_ascii=False, indent=2)
