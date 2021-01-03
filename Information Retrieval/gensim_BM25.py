from utils import *
from collections import defaultdict
import pprint
from gensim import corpora, models, similarities
import os
import sys
from gensim.summarization.bm25 import get_bm25_weights, BM25

#%% ==========  LOAD DATA ===============
project_path = os.getcwd()
data_path = os.path.join(project_path, 'data')
qrels_file = os.path.join(data_path, '2019qrels-pass.txt')
test_topfile_name = os.path.join(data_path, 'msmarco-passagetest2019-43-sorted-top1000.tsv')
test_topfile = read_tsv(test_topfile_name)

save_name =  os.path.join(project_path, 'results/res_bm25.txt')
if os.path.exists(save_name):
    os.remove(save_name)


# 构造query dict和psg(passage) dict， 都以qid为key
# data_dicts: qid - [pid, psg]
data_dicts = {}
query_dicts = {}
for item in test_topfile:
    qid, pid, query, psg = item[0], item[1], item[2], item[3]
    if not qid in query_dicts.keys():
        query_dicts[qid] = query
        data_dicts[qid] = [[pid ,psg]]
    else:
        data_dicts[qid].append([pid, psg])

# 对每个qid中，利用对应的1000个psg训练BM25向量模型，并分别计算得分
qids_count = len(list(query_dicts.keys()))
for qid in tqdm(query_dicts.keys()):
    # 预处理
    text_corpus = [item[1] for item in data_dicts[qid]]
    stoplist = set('for a of the and to in'.split(' '))
    texts = [[word for word in document.lower().split() if word not in stoplist]
            for document in text_corpus]

    # 去掉只出现1次的token
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
    query = query_dicts[qid].lower().split()
    
    # 构建模型
    bm25_model = BM25(processed_corpus,k1=1.5, b=0.75, epsilon=0.25)
    pids_scores = []
    # 计算得分
    for i, cps in enumerate(processed_corpus):
        score = bm25_model.get_score(query, i)
        pids_scores.append([data_dicts[qid][i][0], score ])
    
    pids_scores = sorted(pids_scores, key=(lambda x: x[1]), reverse=True )

    # 存储文件
    with open(save_name, 'a') as f:
        for i in range(len(pids_scores)):
            pid, score = pids_scores[i][0], pids_scores[i][1]
            item = [qid, 'Q0', pid, str(i+1), '%.5f' % score]
            line = ''
            for i in range(len(item)):
                line += (item[i] + ' ')
            line += 'gensim_BM25\n'
            f.write(line)
print('')

#%% 评估
qrel = TrecQrel(qrels_file)
rst = TrecRun(save_name)
for depth in [5, 10, 15, 20, 30, 100, 200, 500, 1000]:
    r1_p25 = TrecEval(rst, qrel).get_ndcg(depth=depth)
    print('ndcg_cur_%d \t all \t %.4f' % (depth, r1_p25))