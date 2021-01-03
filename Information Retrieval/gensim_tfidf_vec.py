from utils import *
from collections import defaultdict
import pprint
from gensim import corpora, models, similarities
import os
import sys

#%% 载入数据
project_path = os.getcwd()
data_path = os.path.join(project_path, 'data')
qrels_file = os.path.join(data_path, '2019qrels-pass.txt')
test_topfile_name = os.path.join(data_path, 'msmarco-passagetest2019-43-sorted-top1000.tsv')
test_topfile = read_tsv(test_topfile_name)
save_name = os.path.join(project_path, 'results', 'res_tfidf.txt') 
if os.path.exists(save_name):
    os.remove(save_name)

# 按qid分组
# 构造query dict和psg dict， 都以qid为key
# data_dicts: qid - [pid - psg]
data_dicts = {}
query_dicts = {}
for item in test_topfile:
    qid, pid, query, psg = item[0], item[1], item[2], item[3]
    if not qid in query_dicts.keys():
        query_dicts[qid] = query
        data_dicts[qid] = [[pid ,psg]]
    else:
        data_dicts[qid].append([pid, psg])


# 按查询中的每个qid中的所有psg训练tf-idf向量模型，并计算cos相似度
qids_count = len(list(query_dicts.keys()))
for qid in tqdm(query_dicts.keys()):

    # 预处理
    text_corpus = [item[1] for item in data_dicts[qid]]
    stoplist = set('for a of the and to in'.split(' '))
    texts = [[word for word in document.lower().split() if word not in stoplist]
            for document in text_corpus]

    from collections import defaultdict
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]

    
    # 建立词典
    dictionary = corpora.Dictionary(processed_corpus)
    # 生成查询和所有语料词袋向量表示
    query = query_dicts[qid]
    bow_query = dictionary.doc2bow(query.lower().split())
    bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]

    # tf-idf模型
    tfidf = models.TfidfModel(bow_corpus)

    # score
    index = similarities.SparseMatrixSimilarity(tfidf[bow_corpus], num_features=tfidf.num_nnz)
    sims = index[tfidf[bow_query]]
    pids_scores = []
    for i, score in enumerate(sims):
        pid = data_dicts[qid][i][0]
        pids_scores.append([pid, score])

    pids_scores = sorted(pids_scores, key=(lambda x:x[1]), reverse=True)
    
    # 存储结果
    with open(save_name, 'a') as f:
        for i in range(len(pids_scores)):
            pid, score = pids_scores[i][0], pids_scores[i][1]
            item = [qid, 'Q0', pid, str(i+1), '%.5f' % score]
            line = ''
            for i in range(len(item)):
                line += (item[i] + ' ')
            line += 'gensim_tfidf\n'
            f.write(line)

#%% 评估
qrel = TrecQrel(qrels_file)
rst = TrecRun(save_name)
for depth in [5, 10, 15, 20, 30, 100, 200, 500, 1000]:
    r1_p25 = TrecEval(rst, qrel).get_ndcg(depth=depth)
    print('ndcg_cur_%d \t all \t %.4f' % (depth, r1_p25))




