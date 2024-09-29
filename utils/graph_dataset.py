from torch.utils.data import Dataset
from transformers import AutoTokenizer
import os
import pickle

class GraphArSepDataset(Dataset):
    def __init__(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        self.net_data, self.link_data = self.get_data(config['lang'], config['top_n'])

    def get_data(self, lang, top_n):
        net_list = []
        link_list = []
        file_name_list = os.listdir('../data/temp/' + lang + '/')
        for file_name in file_name_list:
            with open('../data/temp/' + lang + '/' + file_name + '/'+str(top_n)+'.nx', 'rb') as file:
                net = pickle.load(file)
                net_list.append(net)
                for edge in net.edges:
                    node_start = net.nodes[edge[0]]
                    node_end = net.nodes[edge[1]]
                    link_list.append([node_start, node_end])

        return (net_list, link_list)

if __name__ == "__main__":
    config = {'lang': 'fr',
              'top_n': 1,
              'model_name': 'google-bert/bert-base-uncased'}
    dataset_obj = GraphArSepDataset(config)
