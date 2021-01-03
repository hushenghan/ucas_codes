from utils import *
from collections import defaultdict

def print_info(info):
    print('---------%s-----------' % info)


# read files
# 考虑
files = 'data/msmarco-passagetest2019-43-sorted-top1000.tsv'

# read
data = read_tsv(files)

print_info('预处理 start')
# 预处理
stoplist = set('for a of the and to in'.split())
passages = [item[3]  for item in data]
pids = []
passages = [
    [word for word in passage.lower().split() if word not in stoplist]
    for passage in passages
]

# 除去所有只出现一次的
frequency = defaultdict(int)
for passage in passages:
    for token in passage:
        frequency[token] += 1

# 作为data的第3列
passages = [
    [token for token in passage if frequency[token] > 1]
    for passage in passages
]

for token in frequency.keys():
    if not frequency[token]:
        frequency.pop(token)

print_info('预处理 end')


print_info('build idf start')
# idf 
idf = defaultdict(int)
for token in tqdm(frequency.keys()):
    for passage in passages:
        if token in passage:
            idf[token] += 1

N = len(passages)
V = len(idf.keys())
for token in tqdm(idf.keys()):
    idf[token] = math.log10(N/idf[token])

print_info('build idf end')

def gen_tf(psg, vocu, query=True):
    if query:
        psg = [word for word in psg.lower().split() if word in vocu]
    tfs = defaultdict(int)
    for word in psg:
        tfs[word] = math.log10(psg.count(word))
    return tfs

print_info('build passage vec start')
# 构建passage的向量
psgVecs = defaultdict(np.array)
idfs_list = [idf[token]  for token in idf.keys()]
psgVecs.setdefault(np.array(idfs_list, dtype=np.float))
for i in tqdm(range(len(data))):
    pid = data[i][1]
    tfs = gen_tf(passages[i], idf.keys())
    for i, token in enumerate(idf.keys()):
        if token in tfs.keys():
            psgVecs[token][i] *= (tfs[token] + 1)
print_info('build passage vec end')

# 构建查询的向量形式
print_info('build query vec start')
queryVecs = defaultdict(np.array)
idfs_list = [idf[token]  for token in idf.keys()]
queryVecs.setdefault(np.array(idfs_list, dtype=np.float))
for i in tqdm(range(len(data))):
    qid = data[i][0]
    query = data[i][2]
    if qid in queryVecs.keys():
        continue

    tfs = gen_tf(query, idf.keys(), query=True)
    for i, token in enumerate(idf.keys()):
        if token in tfs.keys():
            psgVecs[token][i] *= tfs[token]

print_info('build query vec end')


# 计算score
print_info('calculate socre begin')
scores = []
for i in tqdm(range(len(data))):
    qid, pid = data[0], data[1]
    qvec = queryVecs[qid]
    pvec = psgVecs[pid]
    #con q p
    score = np.sum(qvec*pvec) / ( np.linalg.norm(qvec) * np.linalg.norm(pvec))
    item = [qid, pid, score]
    scores.append(item)
print_info('calculate socre end')
# ranking

print_info('build write data')
qidDict = {}
for item in scores:
    if not item[0] in qidDict.keys():
        qidDict[item[0]] = [item[1], item[2]]
    else:
        qidDict[item[0]].append([item[1], item[2]])
res = []
for qid in qidDict.keys():
    qid_res = qidDict[qid]
    sorted(qid_res, key=(lambda x: x[1]), reverse=True)
    for i, t in enumerate(qid_res):
        item = [qid, 'Q0', t[0], str(i+1), '%.3f' % t[1]]
    res += item

with open('res_vec.txt', 'w') as f:
    for item in res:
        line = ''
        for i in range(len(item)): 
            line += (item[i] + ' ')
        line += ('random_commenOrderd' + '\n')
        f.write(line)
        
