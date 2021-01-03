# 请设置激活的环境名称(此处为 “ir”)
env_name=ir

. ~/.bashrc
conda activate $env_name


echo -e "\n~~~~~~~~~运行所有脚本~~~~~~~~~~~ \n"

echo -e "\n\n\n！！！！！！！！！！！！！！！！ 一：使用预训练的蒸馏BERT模型 —— distilbert-base-uncased ！！！！！！！！！！！！！\n"
# 请限制使用的GPU数量，尽量不要超过4个，否则可能报错
CUDA_VISIBLE_DEVICES=0,1,2,3  python bert_trainer.py

echo -e "\n\n\n！！！！！！！！！！！！！！！！ 二：使用BM25 model ！！！！！！！！！！！\n"
python gensim_BM25.py

echo -e "\n\n\n！！！！！！！！！！！！！！！！ 三：使用 tfidf model ！！！！！！！！！！！\n"
python gensim_tfidf_vec.py

echo -e "\n\n\n！！！！！！！！！！！！！！！！ 四：使用 doc2vec model ！！！！！！！！！！！！！\n"
python gensim_doc2vec.py


