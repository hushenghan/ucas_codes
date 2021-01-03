from utils import *
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, TrainerCallback
import argparse


class TRECDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        if not labels is None:
            self.labels = labels
        else:
            self.labels = [0]*len(encodings.data['input_ids'])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)



def predict_test(model, test_loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            device = model.device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)[0]
            preds.append(outputs.cpu())
    preds = torch.cat(preds, dim=0)
    preds_softmax = torch.softmax(preds, dim=1)[:, 1].numpy()
    return preds_softmax


class EvaluateCallback(TrainerCallback):
    def __init__(self, test_loader, test_topfile, model_name=' ', project_path='./'):
        self.test_loader = test_loader
        self.test_topfile = test_topfile
        self.model_name = model_name
        self.project_path = project_path

    def on_epoch_end(self, args, state, control, model, metrics=None, **kwargs):
        print_info('epoch end && predict')
        preds_softmax = predict_test(model, self.test_loader)
        print('test result:\n')
        testRst(preds_softmax, self.test_topfile, method=self.model_name, project_path=self.project_path)
        model.train()
        print_info('predict over')



# =========================================================
class TRECTrain():
    def __init__(self, opts, project_path='./'):
        self.project_path = project_path
        self.model_name = opts.model

        self.training_args = TrainingArguments(
            output_dir='./check_points',                    # output directory
            num_train_epochs=opts.epoch,                    # total number of training epochs
            per_device_train_batch_size=opts.train_bs,      # batch size per device during training
            warmup_steps=opts.warmup_steps,                 # number of warmup steps for learning rate scheduler
            weight_decay=opts.weight_decay,                 # strength of weight decay
            logging_dir='./logs',                           # directory for storing logs
            logging_steps=1000,
            learning_rate=opts.lr,
            evaluation_strategy='no',
            save_steps=1500,
        )        
        print_info('load model')
        self.load_model()
        print_info('load data')
        self.load_data()

    def load_data(self):
        rst_path = os.path.join(self.project_path, 'results')
        data_path = os.path.join(self.project_path, 'data')

        train_passage = read_tsv(os.path.join(data_path, 'collection.train.sampled.tsv'))                       #pid,   passage         39820
        train_queries = read_tsv(os.path.join(data_path, 'queries.train.sampled.tsv'))                          #qid,   query           20000
        train_triples = read_tsv(os.path.join(data_path, 'qidpidtriples.train.sampled.tsv'))                    #qid,   Ppid,   Npid    20000
        test_queries = read_tsv(os.path.join(data_path, 'msmarco-test2019-43-queries.tsv'))                     #qid,   query           43 qid
        self.test_topfile = read_tsv(os.path.join(data_path, 'msmarco-passagetest2019-43-sorted-top1000.tsv'))  #qid,   pid,    query,  passage  41042
        # test qrels:    qid,   "Q0",   pid,    rating,  score

        train_passage_dict = dict(train_passage)
        train_queries_dict = dict(train_queries)
        train_texts = [[train_queries_dict[qid], train_passage_dict[Ppid]] for qid, Ppid, Npid in train_triples]  + \
                        [[train_queries_dict[qid], train_passage_dict[Npid]] for qid, Pid, Npid in train_triples]
        train_labels = [1 for _ in train_triples] + [0 for _ in train_triples]
        test_texts = [[item[2], item[3]] for item in self.test_topfile]

        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True)
        test_encodings = self.tokenizer(test_texts, truncation=True, padding=True)

        self.train_dataset = TRECDataset(train_encodings, train_labels)
        self.test_dataset = TRECDataset(test_encodings, None)
        self.test_loader = DataLoader(self.test_dataset, batch_size=128, shuffle=False)

    def load_model(self):
        if self.model_name == 'distilbert-base-uncased':
            from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_name)
            self.model = DistilBertForSequenceClassification.from_pretrained(self.model_name)
        elif self.model_name == 'distilbert-base-multilingual-cased':
            from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_name)
            self.model = DistilBertForSequenceClassification.from_pretrained(self.model_name)
        elif self.model_name == 'bert-base-uncased':
            from transformers import BertTokenizerFast, BertForSequenceClassification
            self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name)
            self.model = BertForSequenceClassification.from_pretrained(self.model_name)        
        elif self.model_name == 'bert-base-cased-finetuned-mrpc':
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        elif self.model_name == 'bert-base-multilingual-cased':
            from transformers import BertTokenizerFast, BertForSequenceClassification
            self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name)
            self.model = BertForSequenceClassification.from_pretrained(self.model_name)
        else:
            print('wrongly model name!')
            pass            

    def train(self,):
        print_info('begin train')

        evaluateCallback = EvaluateCallback(self.test_loader, self.test_topfile, model_name=self.model_name, project_path=self.project_path)

        trainer = Trainer(
            model=self.model,                         # the instantiated Transformers model to be trained
            args=self.training_args,                  # training arguments, defined above
            train_dataset=self.train_dataset,         # training dataset
            callbacks=[evaluateCallback]
        )        
        trainer.train()
        trainer.save_model('saved_model/%s/' % self.model_name)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="setting params!")
    parser.add_argument('--lr',
                        type=float,
                        default=5e-5)
    parser.add_argument('--warmup_steps',
                        type=int,
                        default=300)
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.1)
    parser.add_argument('--model',
                        type=str,
                        default='distilbert-base-uncased',
                        choices=['distilbert-base-uncased', 'distilbert-base-multilingual-cased', 'bert-base-uncased', 'bert-base-cased-finetuned-mrpc', 'bert-base-multilingual-cased'])
    parser.add_argument('--epoch',
                        type=int,
                        default=2)
    parser.add_argument('--test_bs',
                        type=int,
                        default=64)
    parser.add_argument('--train_bs',
                        type=int,
                          default=16)  

    args = parser.parse_args()
    project_path = os.getcwd()
    trecTrain = TRECTrain(args, project_path=project_path)   
    trecTrain.train() 