# 任务
 - passage re-ranking。重排已经给出的top 1000 文档
 - 给定的43个待查查询，每个给出了1000个相关文档，对这些文档的顺序进行重排。
 - 训练数据：
 ```
    train passage: pid,   passage                     39820   collection.train.sampled
    train queries: qid,   query                       20000   queries.train.sampled
    train triples: qid,   Ppid,   Npid                20000   qidpidtriples.train.sampled
    test  queries: qid,   query                       43      msmarco-test2019-43-queries
    test  topfile: qid,   pid,    query,  passage     41042   msmarco-passagetest2019-43-sorted-top1000    43 qid & 40894 pid     
    test qrels:    qid,   "Q0",   pid,    rating      9260    2019qrels-pass                               43 qid & 9139 pid     
```
    - 已知：
        - 20k个训练查询，每个训练查询包含正负训练样本各一个，共40k个训练样本。
        - topfile 和 qrels中共同的pid，也就是共同的passage一共也就4623个
        - topfile 和train psg中有少数重复的，两者相加共80862, 但不重复的有80714， 重复了100左右
    - 问题？
        - qrels中的Pid是43个的那些，用于比较。
        - TREC格式：
            - 查询ID、 Q0、 文档排序、文档评分、系统ID
        - 43个测试查询在top 1000中，每个都基本都1000左右的passage，对每个查询的1000个passage计算相似度，进行排序。保存在res文件中。

 - 训练指标：
    - ground truth: qrels文件， 
    - ./trec_eval -m ndcg_cut qrels res
    - NDCG：文档的相关性并非相关和不相关，而是有相关度级别0,1,2,3， [blog](https://weirping.github.io/blog/Metrics-in-IR.html)
        - 相关度级别越高的结果越多越好
        - 相关度级别越高的结果越靠前越好(本次实验中，可认为都相关，因此要求排序，即是的相关度高的结果更加靠前)
        - normalized discounted  cumulated  direct gain
        - best vector = [3,3,3,3,2,2,2,2,1,1,1,1,1,0,0,0,0], real vector = [3,2,1,3,2,3,1,3,0,2,3,...]
        - $nDCG[i] = DCG'[i] / DCG'_{I}[i] $


- envs
    conda
    python=3.8.5

    pytorch=1.7.1
    conda install pytorch torchvision cpuonly -c pytorch 

    transformers
    pip install transformers
    trectools
    pip install trectools
    gensim
    pip install gensim
    tqdm

 
 - 流程：在训练集上使用正负样本进行训练，对passage的相关性进行评分，之后对test


# 现有工具
- [往年学生大作业 -python](E:\doc\usas\现代信息检索\大作业\etc\information-retrieval-master)，题目有所不同，实现方法有BM25, word2vec. 
- [trec tools -python](https://github.com/joaopalotti/trec_tools). 用于TREC各类比赛的封装和基础工具，Python实现
    - maybe 可以用于输出输出或者结果分析
- [gensim -python](https://radimrehurek.com/gensim/auto_examples/index.html): 用于从原始的非结构化的文本中，无监督地学习到文本隐层的主题向量表达。支持TF-IDF，LSA，LDA，和word2vec在内的多种主题模型算法，支持流式训练，并提供了诸如相似度计算，信息检索等一些常用任务的API接口.
- [lucene -java](https://lucene.apache.org/)： [简单教程](https://www.yiibai.com/lucene/)。简单而功能强大的基于Java的搜索库.  存在其他各种版本，主要是向量空间模型。
- [xapian- C++](https://xapian.org/)：C++版本的lucene，但是也提供了Python等接口。 内部实现了BM25
- [SMART -C]() 向量空间模型工具
- [anserini](https://github.com/castorini/anserini): 建立与lucene上的，用于搭建工业和学术界间之间的桥梁。
- [Lemur、Indri]:
- [reranking with bert](https://paperswithcode.com/paper/passage-re-ranking-with-bert): paper with code。 
- [ms macro](https://microsoft.github.io/msmarco/): microsoft 创建的深度学习相关的搜索竞赛，上面有开源的重排算法。很多使用了[openmatch](https://github.com/googleforgames/open-match)开源库
- [Master_Thesis_MSMARCO_Passage_Ranking_BERT](https://github.com/Tomjg14/Master_Thesis_MSMARCO_Passage_Ranking_BERT): 个人相关仓库。有部分实验总结。
- [trec 竞赛队伍报告](https://trec.nist.gov/pubs/trec28/xref.html#deep) 没有显式地给出代码。没有看出有re ranking任务。

# 方法
- 词袋模型 -> tf-idf
- LSA:  Latent Semantic Analysis 隐性语义分析
- LDA:  Latent Dirichlet Allocation 隐狄利克雷分配模型
- word2vec： gensim库。 [知乎1](https://zhuanlan.zhihu.com/p/26306795)、[知乎2](https://zhuanlan.zhihu.com/p/43736169)、[skip-gram](https://mp.weixin.qq.com/s/reT4lAjwo4fHV4ctR9zbxQ?)
- fasttext: [知乎1](https://zhuanlan.zhihu.com/p/32965521)


# 实验测试
- data feature: 每个查询对应的top file中文章个数以及qrel文章个数，以及两者共同的文章个数
    ```
    qid 19335        qtop: 1000      qrels 194       commen: 65
    qid 47923        qtop: 1000      qrels 143       commen: 106
    qid 87181        qtop: 1000      qrels 158       commen: 109
    qid 87452        qtop: 1000      qrels 139       commen: 68 
    qid 104861       qtop: 1000      qrels 306       commen: 121
    qid 130510       qtop: 1000      qrels 133       commen: 81 
    qid 131843       qtop: 1000      qrels 132       commen: 18 
    qid 146187       qtop: 1000      qrels 138       commen: 84 
    qid 148538       qtop: 1000      qrels 159       commen: 116
    qid 156493       qtop: 1000      qrels 300       commen: 235
    qid 168216       qtop: 1000      qrels 582       commen: 388
    qid 182539       qtop: 1000      qrels 132       commen: 51
    qid 183378       qtop: 1000      qrels 451       commen: 202
    qid 207786       qtop: 1000      qrels 137       commen: 56
    qid 264014       qtop: 1000      qrels 382       commen: 131
    qid 359349       qtop: 1000      qrels 139       commen: 115
    qid 405717       qtop: 1000      qrels 144       commen: 116
    qid 443396       qtop: 1000      qrels 188       commen: 55
    qid 451602       qtop: 1000      qrels 220       commen: 97
    qid 489204       qtop: 1000      qrels 175       commen: 51
    qid 490595       qtop: 1000      qrels 148       commen: 121
    qid 527433       qtop: 1000      qrels 160       commen: 83
    qid 573724       qtop: 1000      qrels 141       commen: 104
    qid 833860       qtop: 1000      qrels 157       commen: 70
    qid 855410       qtop: 5         qrels 183       commen: 5
    qid 915593       qtop: 1000      qrels 192       commen: 149
    qid 962179       qtop: 1000      qrels 161       commen: 113
    qid 1037798      qtop: 1000      qrels 154       commen: 67
    qid 1063750      qtop: 1000      qrels 392       commen: 38
    qid 1103812      qtop: 1000      qrels 141       commen: 75
    qid 1106007      qtop: 1000      qrels 178       commen: 86
    qid 1110199      qtop: 1000      qrels 175       commen: 70
    qid 1112341      qtop: 1000      qrels 223       commen: 138
    qid 1113437      qtop: 1000      qrels 180       commen: 124
    qid 1114646      qtop: 1000      qrels 151       commen: 57
    qid 1114819      qtop: 1000      qrels 470       commen: 321
    qid 1115776      qtop: 1000      qrels 152       commen: 69
    qid 1117099      qtop: 1000      qrels 257       commen: 50
    qid 1121402      qtop: 1000      qrels 146       commen: 90
    qid 1121709      qtop: 37        qrels 178       commen: 21
    qid 1124210      qtop: 1000      qrels 330       commen: 125
    qid 1129237      qtop: 1000      qrels 147       commen: 83
    qid 1133167      qtop: 1000      qrels 492       commen: 288
    ```




    - 使用gensim，tf-idf， vector方式： res_tfidf_vec.txt
    ```
        ndcg_cur_5       all     0.2333
        ndcg_cur_10      all     0.2317
        ndcg_cur_15      all     0.2269
        ndcg_cur_20      all     0.2274
        ndcg_cur_30      all     0.2403
        ndcg_cur_100     all     0.2686
        ndcg_cur_200     all     0.3103
        ndcg_cur_500     all     0.3881
        ndcg_cur_1000    all     0.4427
    ``` 
    - 使用gensim中BM25模型计算权重 res_bm25.txt
    ```
        ndcg_cur_5       all     0.2755
        ndcg_cur_10      all     0.2638
        ndcg_cur_15      all     0.2570
        ndcg_cur_20      all     0.2494
        ndcg_cur_30      all     0.2496
        ndcg_cur_100     all     0.2721
        ndcg_cur_200     all     0.2998
        ndcg_cur_500     all     0.3692
        ndcg_cur_1000    all     0.4469    
    ```
    - 使用gensim中doc2vec模型计算权重  CBOW epoch 100  vector size 100  dm=0  
        - 使用所有的train sample和top file，共80712个文档，
        ```
        ndcg_cur_5       all     0.2441
        ndcg_cur_10      all     0.2443
        ndcg_cur_15      all     0.2413
        ndcg_cur_20      all     0.2381
        ndcg_cur_30      all     0.2399
        ndcg_cur_100     all     0.2636
        ndcg_cur_200     all     0.3024
        ndcg_cur_500     all     0.3882
        ndcg_cur_1000    all     0.4373
        ```



    - distilbert-base-uncased epoch 2  warmup_steps=300 lr=5e-5
        ```
        ndcg_cur_5 	     all 	 0.6711
        ndcg_cur_10 	 all 	 0.6475
        ndcg_cur_15 	 all 	 0.6463
        ndcg_cur_20 	 all 	 0.6372
        ndcg_cur_30 	 all 	 0.6239
        ndcg_cur_100 	 all 	 0.5951
        ndcg_cur_200 	 all 	 0.6057
        ndcg_cur_500 	 all 	 0.6293
        ndcg_cur_1000 	 all 	 0.6364
        ```



