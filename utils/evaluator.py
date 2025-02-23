import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.datasetor_reading_order import ArticleDataset, get_dataet

def get_evaluator(config):
    if config['dataset_name'] == 'ar':
        return Evaluator(config)
    if config['dataset_name'] == 'fullcon':
        return EvaluatorFullConn(config)

class Evaluator:
    def __init__(self, config):
        with open(config['lang'] + '_test_set.txt', 'r') as f:
            self.file_name_list = f.read().split('\n')[:-1]

        self.dataloader_list =[x for x in [DataLoader(get_dataet(config, 'test', file_name),
                                           batch_size=config['batch_size'],
                                           shuffle=False)
                                          for file_name in self.file_name_list] if len(x) < 800]

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

    def gt_generate(self, length):
        length_record = []
        begin = 0
        while length >= 1 :
            end = length - 1 + begin
            length_record.append([begin, end])
            begin = end
            length -= 1

        return length_record

    def article_decode(self, max_index, length_record):
        def check_in(index, result):
            for result_item in result:
                if index in result_item:
                    return True

            return False

        result = []
        for index, length in enumerate(length_record):
            if check_in(index, result):
                continue

            result_item = [index]
            prediction = max_index[length[0]: length[1]]

            for prediction_index, prediction_item in enumerate(prediction):
                if prediction_item == 0 or check_in(prediction_index+index+1, result):
                    continue
                else:
                    result_item.append(prediction_index+index+1)

            result.append(result_item)

        if not check_in(length_record[0][1], result):
            result.append([length_record[0][1]])

        return result

    def _inner_test_loop(self, model, dataloader):
        loss_this_article = []
        output_this_article = []
        gt = []
        for _, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            # if _ < 28:
            #     continue
            output = model(data)
            gt+=(data['gt'].squeeze(0).cpu().detach().numpy().tolist())
            output_this_article += torch.max(output['output'], dim=1).indices.detach().cpu().numpy().tolist()
            loss_this_article.append(output['loss'].item())

        length_record = self.gt_generate(data['num_paragraph'][0][0].item())
        article_gt = self.article_decode(gt, length_record)
        article_prediction = self.article_decode(output_this_article, length_record)
        article_evaluation_result = self.evaluate_single_page(article_prediction, article_gt)
        return sum(loss_this_article), article_evaluation_result

    def __call__(self, model):
        loss = []
        p = []
        r = []
        f1 = []
        ppa = []
        mac = []
        for dataloader in tqdm(self.dataloader_list):
            loss_this_article, article_evaluation_result = self._inner_test_loop(model, dataloader)
            # try:
            #     loss_this_article, article_evaluation_result = self._inner_test_loop(model, dataloader)
            # except:
            #     continue
            loss.append(loss_this_article)
            p += article_evaluation_result['p']
            r += article_evaluation_result['r']
            f1 += article_evaluation_result['f1']
            ppa += article_evaluation_result['ppa']
            mac.append(1 - sum(article_evaluation_result['error_value_list']) / len(article_evaluation_result['error_value_list']))

        return {'loss': sum(loss) / len(loss),
                'mac': sum(mac) / len(mac),
                'ppa': sum(ppa) / len(ppa),
                'p': sum(p) / len(p),
                'r': sum(r) / len(r),
                'f1': sum(f1) / len(f1)
                }

class EvaluatorFullConn(Evaluator):
    def __init__(self, config):
        super().__init__(config)

    def article_decode(self, index_list, prediction_list):
        result_list = [[-1]]
        for index in range(len(index_list)):
            flag_0 = False
            flag_1 = False
            index_pair = index_list[index]
            prediction = prediction_list[index]
            if prediction == 1:
                for article in result_list:
                    if index_pair[0] in article or index_pair[1] in article:
                        article.append(index_pair[0])
                        article.append(index_pair[1])
                        flag_0 = True

                if not flag_0:
                    result_list.append(index_pair)

            if prediction == 0:
                for article in result_list:
                    if index_pair[0] in article:
                        flag_0 = True
                    if index_pair[1] in article:
                        flag_1 = True

                if not flag_0:
                    result_list.append([index_pair[0]])
                if not flag_1:
                    result_list.append([index_pair[1]])

        for index, article in enumerate(result_list):
            result_list[index] = list(set(article))
            result_list[index].sort()

        result_list.pop(0)
        result_merge = []
        while len(result_list) > 0:
            temp = result_list[0]
            result_list.pop(0)
            index_to_pop = []
            for index, article in enumerate(result_list):
                if len(set(temp).intersection(set(article))) > 0:
                    temp += result_list[index]
                    index_to_pop.append(index)

            result_list = [x for index, x in enumerate(result_list) if index not in index_to_pop]
            a = list(set(temp))
            a.sort()
            result_merge.append(a)

        return result_merge

    def _inner_test_loop(self, model, dataloader):
        index_this_article = []
        loss_this_article = []
        output_this_article = []
        gt = []
        for _, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            # if _ < 28:
            #     continue
            output = model(data)
            index_this_article += data['index'].cpu().detach().numpy().tolist()
            gt += (data['gt'].squeeze(0).cpu().detach().numpy().tolist())
            output_this_article += torch.max(output['output'], dim=1).indices.detach().cpu().numpy().tolist()
            loss_this_article.append(output['loss'].item())
            article_num = (data['length'][0][0] + 1).item()

        article_gt = self.article_decode(index_this_article, gt)
        single_article = []
        for index in range(article_num):
            flag = False
            for article in article_gt:
                if index in article:
                    flag = True

            if not flag:
                single_article.append([index])

        article_gt += single_article
        article_prediction = self.article_decode(index_this_article, output_this_article)
        article_prediction += single_article
        article_evaluation_result = self.evaluate_single_page(article_prediction, article_gt)
        return sum(loss_this_article), article_evaluation_result

    # def model_evaluate(self, model, dataloader):
    #     loss = []
    #     p = []
    #     r = []
    #     f1 = []
    #     ppa = []
    #     error_value_list = []
    #     print('validating......\n')
    #     with torch.no_grad():
    #         for step, data in tqdm(enumerate(dataloader), total=len(dataloader)):
    #             output = model(data)
    #             loss.append(output['loss'].item())
    #             error_value_list += output['error_value_list'].tolist()
    #             r += output['r']
    #             f1 += output['f1']
    #             ppa += output['ppa']
    #             p += output['p']
    #
    #     return {'loss': sum(loss) / len(loss),
    #             'error_value_list': sum(error_value_list) / len(error_value_list),
    #             'mac': 1 - sum(error_value_list) / len(error_value_list),
    #             'ppa': sum(ppa) / len(ppa),
    #             'p': sum(p) / len(p),
    #             'r': sum(r) / len(r),
    #             'f1': sum(f1) / len(f1)}
    #
