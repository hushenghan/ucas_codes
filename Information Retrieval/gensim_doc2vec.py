import sys
import gensim
from gensim import models
import os
import smart_open
from utils import *
import time

#%% 载入数据
print_info('load data')
project_path = os.getcwd()
data_path = os.path.join(project_path, 'data')
qrels_file = os.path.join(data_path, '2019qrels-pass.txt')
test_topfile_name = os.path.join(data_path, 'msmarco-passagetest2019-43-sorted-top1000.tsv')
test_topfile = read_tsv(test_topfile_name)
train_data = read_tsv(os.path.join(data_path, 'collection.train.sampled.tsv'))

save_name = os.path.join(project_path, 'results', 'rst_doc2vec.txt') 
if os.path.exists(save_name):
    os.remove(save_name)


#%% 预处理
print_info('preprocess data')
doc_tag = 0
pids = set()
def read_corpus(data, pid_idx=1,tokens_only=False, ):
    global doc_tag
    for item in data:
        pid = item[pid_idx]
        if pid in pids:
            continue
        else:
            pids.add(pid) 
        tokens = gensim.utils.simple_preprocess(item[-1], min_len=1, max_len=100)
        
        if tokens_only:
            yield tokens
        else:
            yield gensim.models.doc2vec.TaggedDocument(tokens, [doc_tag])
            doc_tag += 1

train_corpus =  list(read_corpus(train_data, pid_idx=0))  #+ list(read_corpus(test_topfile,pid_idx=1))
print('train corpus count: ', len(train_corpus))



#%% 训练及预测
def main(vector_size=400, epochs=10, dm=1):
    # train
    print_info('train begin')
    start_time = time.time()
    model = models.doc2vec.Doc2Vec(vector_size=vector_size, min_count=2, epochs=epochs, workers=5, dm=dm)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    end_time = time.time()
    print('train time:', end_time-start_time)
    print('train time per epoch: ', (end_time-start_time) /model.epochs)

    # query
    print_info('query begin')
    rst_data = []
    qid_queryvec = {}
    top_file_count = len(test_topfile)
    for item in tqdm(test_topfile):
        qid, pid, query, psg = item[0], item[1], item[2], item[3]
        if qid not in qid_queryvec.keys():
            qid_queryvec[qid] = model.infer_vector(gensim.utils.simple_preprocess(query))
        psg_vec = model.infer_vector(gensim.utils.simple_preprocess(psg))
        score = cosine_similarity(qid_queryvec[qid], psg_vec)
        line = [qid, pid, score]
        rst_data.append(line)

    # 排序，先按qid分组，之后组内按分数排序
    qid_scores = {}
    for item in rst_data:
        qid = item[0]
        if not qid in qid_scores.keys():
            qid_scores[qid] = [item]
        else:
            qid_scores[qid].append(item)

    for qid in qid_scores.keys():
        query_scores = qid_scores[qid]
        qid_scores[qid] = sorted(query_scores, key=(lambda x:x[2]), reverse=True)

    print_info('write to rst file')
    with open(save_name, 'w') as f:
        for qid in qid_scores.keys():
            for i, item in enumerate(qid_scores[qid]):
                qid, pid, rank, score = item[0], item[1], str(i+1), "{:.4f}".format(item[2])
                line = qid + ' ' + 'Q0' + ' ' + pid + ' ' + rank + ' ' + score + ' ' + 'doc2vec' + '\n'
                f.write(line)

    print_info('trec eval')
    print('vec size: ', vector_size)
    print('epochs: ', epochs)
    print('dm: ', dm)    
    trec_eval_ndcg(save_name, data_path=data_path)

if __name__ == "__main__":
    main(vector_size=100, epochs=50, dm=0)

