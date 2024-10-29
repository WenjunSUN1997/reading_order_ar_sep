import numpy as np
import torch
from tqdm import tqdm

class Evaluator:
    def __init__(self):
        pass

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
