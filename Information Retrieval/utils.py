from trectools import TrecEval, TrecRes, TrecRun, TrecQrel
import csv
import math
import numpy as np
import random
import os
from tqdm import tqdm

def trec_eval_ndcg(run_name, data_path='./data/', depths=[5, 10, 15, 20, 30, 100, 200, 500, 1000]):
    qrel_name = os.path.join(data_path, '2019qrels-pass.txt')
    qrel = TrecQrel(qrel_name)
    res = TrecRun(run_name)
    for depth in depths:
        score = TrecEval(res, qrel).get_ndcg(depth=depth)
        print('ndcg_cur_%d \t all \t %.4f' % (depth, score))


def testRst(pred, test_topfile, method='', project_path='./'):
    assert ((len(pred.shape) == 1) or (len(pred.shape)==2))
    if len(pred.shape) == 2:
        pred = pred[:, 1]
    # divide by qid
    test_topfile_dicts = {}
    for idx, item in enumerate(test_topfile):
        qid, pid, query, passage = item
        score = pred[idx]
        if not qid in test_topfile_dicts.keys():
            test_topfile_dicts[qid] = [[pid, score]]
        else:
            test_topfile_dicts[qid].append([pid, score])

    # sort by score for each query
    for qid in test_topfile_dicts.keys():
        query_top = test_topfile_dicts[qid]
        query_top = sorted(query_top, key=(lambda x:x[1]), reverse=True)
        test_topfile_dicts[qid] = query_top

    save_name = os.path.join(project_path,  'results', 'res_%s.txt' % method)
    with open(save_name, 'w') as f:
        for qid in test_topfile_dicts.keys():
            query_top = test_topfile_dicts[qid]
            for idx, item in enumerate(query_top):
                pid, score = item
                line = '%s  %s  %s  %s  %.5f  ' % (qid, 'Q0', pid, str(idx+1), score) 
                line += (method + '\n')
                f.write(line)
    trec_eval_ndcg(save_name, data_path=os.path.join(project_path, 'data'))

def read_txt(filename, delimiter=' '):
    data = []
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.split(delimiter)
            data.append(line)
    return data

def read_tsv(filename, delimiter='\t'):
    data = []
    with open(filename, 'r',encoding='UTF-8') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=delimiter)
        for row in  spamreader:
            data.append(row)
    return data

def cosine_similarity(x, y):
    x = np.array(x, dtype=np.float)
    y = np.array(y, dtype=np.float)
    return np.sum(x*y) / (np.linalg.norm(x) * np.linalg.norm(y))

def print_info(info):
    flen = 50
    ln = len(info)
    lrlen = int((flen - ln)/2)
    print('--------------------||%s||--------------' % str(info).ljust(ln+lrlen).rjust(flen))


