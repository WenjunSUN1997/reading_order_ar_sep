import numpy as np
import torch
from openpyxl.styles.builtins import output
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.datasetor_reading_order import ArticleDataset

class Evaluator:
    def __init__(self,config):
        with open(config['lang'] + '_test_set.txt', 'r') as f:
            self.file_name_list = f.read().split('\n')[:-1]

        self.dataloader_list = [DataLoader(ArticleDataset(config, 'test', file_name),
                                           batch_size=config['batch_size'],
                                           shuffle=False)
                                for file_name in self.file_name_list]

    def evaluate_single_page(self, prediction, truth):
        correct_num = 0
        error_list = np.zeros((len(truth), len(prediction)))
        p_list = np.zeros((len(truth), len(prediction)))
        r_list = np.zeros((len(prediction), len(truth)))
        for truth_index, truth_value in enumerate(truth):
            for pre_index, pre_value in enumerate(prediction):
                if set(truth_value) == set(pre_value):
                    correct_num += 1

                jiaoji = set(truth_value).intersection(pre_value)
                bingji = set(truth_value + pre_value)
                p_list[truth_index][pre_index] = len(jiaoji) / len(truth_value)
                r_list[pre_index][truth_index] = len(jiaoji) / len(pre_value)
                error_list[truth_index][pre_index] = len(bingji-jiaoji) / len(bingji)

        error_value_list = np.min(error_list, axis=1)
        p = sum(np.max(p_list, axis=1).tolist()) / len(np.max(p_list, axis=1).tolist())
        r = sum(np.max(r_list, axis=1).tolist()) / len(np.max(r_list, axis=1).tolist())
        f1 = 2*(p*r) / (p+r)
        return {'error_value_list': error_value_list,
                'ppa': [correct_num / len(truth)],
                'p': [p],
                'r': [r],
                'f1': [f1]}

    def get_gt(self, input):
        pass

    def _inner_test_loop(self, model, dataloader):
        loss_this_article = []
        output_this_article = []
        with torch.no_grad():
            for _, data in enumerate(dataloader):
                output = model(data)
                output_this_article += torch.max(output['output'], dim=1).indices.detach().cpu().numpy().tolist()
                loss_this_article.append(output['loss'].item)

        return loss_this_article, output_this_article

    def __call__(self, model):
        loss = []
        for dataloader in tqdm(self.dataloader_list):
            loss_this_article, output_this_article = self._inner_test_loop(model, dataloader)
            loss += loss_this_article

    def model_evaluate(self, model, dataloader):
        loss = []
        p = []
        r = []
        f1 = []
        ppa = []
        error_value_list = []
        print('validating......\n')
        with torch.no_grad():
            for step, data in tqdm(enumerate(dataloader), total=len(dataloader)):
                output = model(data)
                loss.append(output['loss'].item())
                error_value_list += output['error_value_list'].tolist()
                r += output['r']
                f1 += output['f1']
                ppa += output['ppa']
                p += output['p']

        return {'loss': sum(loss) / len(loss),
                'error_value_list': sum(error_value_list) / len(error_value_list),
                'mac': 1 - sum(error_value_list) / len(error_value_list),
                'ppa': sum(ppa) / len(ppa),
                'p': sum(p) / len(p),
                'r': sum(r) / len(r),
                'f1': sum(f1) / len(f1)}

