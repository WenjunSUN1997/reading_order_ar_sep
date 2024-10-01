from statsmodels.tsa.stl.tests.test_stl import file_path
import torch
from torch.utils.data import Dataset
from utils.xml_reader import XmlProcessor
import os
from transformers import AutoTokenizer

class ReadingOrderDataset(Dataset):
    def __init__(self, config):
        self.root_path = config['root_path']
        self.file_name_list = [x for x in os.listdir(config['root_path']) if 'xml' in x]
        self.tokenizer = AutoTokenizer.from_pretrained(config['text_model_name'])
        self.max_token_num = config['max_token_num']
        self.device = config['device']
        self.use_sep_fig = config['use_sep_fig']

    def get_tokenizer_result(self, annotation_list):
        text_list = [x['text'] for x in annotation_list]
        tokenized_result = self.tokenizer(text_list,
                                          max_length=self.max_token_num,
                                          truncation=True,
                                          return_tensors='pt',
                                          padding='max_length')
        return tokenized_result

    def get_fig_result(self, annotation_list):
        pass

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_path, self.file_name_list[idx])
        annotation_list = XmlProcessor(0, file_path).get_annotation()
        tokenized_result = self.get_tokenizer_result(annotation_list)

if __name__ == '__main__':
    config = {'root_path': r'../data/AS_TrainingSet_BnF_NewsEye_v2/',
              'text_model_name': "dbmdz/bert-base-historic-multilingual-64k-td-cased",
              'max_token_num': 512,
              'device': 'cpu',
              'use_sep_fig': True,}
    # file_path = r'../data/AS_TrainingSet_BnF_NewsEye_v2/'
    datasetor = ReadingOrderDataset(config)
    a = datasetor[2]



