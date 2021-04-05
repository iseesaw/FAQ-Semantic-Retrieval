import json
from tqdm import tqdm
dct = json.load(open('faq.json'), encoding='utf8')

new_dict = {}
for _, qas in tqdm(dct.items()):
    post = [q.replace(' ', '') for q in qas.get('q')]
    resp = [a.replace(' ', '') for a in qas.get('a')]
    new_dict[post[0]] = {
        'post': post,
        'resp': resp
    }

json.dump(new_dict, open('clean_faq.json', 'w', encoding='utf8'), ensure_ascii=False, indent=2)

