import torch
import submitit
import copy

from sympy.physics.units import becquerel
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.xml_reader import XmlProcessor
import os
from transformers import AutoTokenizer,  AutoProcessor
from PIL import Image
import cv2
import math
import numpy as np

class ReadingOrderDataset(Dataset):
    def __init__(self, config):
        self.lang = config['lang']
        if self.lang == 'fi':
            self.root_path = r'../data/AS_TrainingSet_NLF_NewsEye_v2/'
            self.img_root_path = r'../data/fi_reading_order/'
        else:
            self.root_path = r'../data/AS_TrainingSet_BnF_NewsEye_v2'
            self.img_root_path = r'../data/fr_reading_order/'

        self.file_name_list = [x for x in os.listdir(self.root_path) if 'xml' in x]
        self.tokenizer = AutoTokenizer.from_pretrained(config['text_model_name'])
        self.max_token_num = config['max_token_num']
        self.device = config['device']
        self.use_sep_fig = config['use_sep_fig']
        self.vision_processor = AutoProcessor.from_pretrained(config['vision_model_name'])
        self.resize = config['resize']
        self.is_benchmark = config['is_benchmark']

    def get_tokenizer_result(self, annotation_list):
        text_list = [x['text'] for x in annotation_list]
        text_list.insert(0, '###')
        text_list.append('###')
        tokenized_result = self.tokenizer(text_list,
                                          max_length=self.max_token_num,
                                          truncation=True,
                                          return_tensors='pt',
                                          padding='max_length')
        return tokenized_result

    def get_fig_result(self, file_name, annotation_list):
        img_folder = os.path.join(self.img_root_path, file_name.split('/')[-1].replace('.xml', '/'))
        benchmark_result = []
        with_sep_result = []
        no_sep_result = []
        for annotation in annotation_list:
            benchmark_result.append(Image.open(img_folder+str(annotation['index'])+'_benchmark.jpg').convert('RGB').resize(self.resize))
            with_sep_result.append(Image.open(img_folder + str(annotation['index']) + '_with_sep.jpg').convert('RGB').resize(self.resize))
            no_sep_result.append(Image.open(img_folder + str(annotation['index']) + '_no_sep.jpg').convert('RGB').resize(self.resize))

        background = Image.open(img_folder+'cls_eos.jpg').convert('RGB').resize(self.resize)
        benchmark_result.insert(0, background)
        benchmark_result.append(background)
        with_sep_result.insert(0, background)
        with_sep_result.append(background)
        no_sep_result.insert(0, background)
        no_sep_result.append(background)
        benchmark_result = self.vision_processor(benchmark_result)
        with_sep_result = self.vision_processor(with_sep_result)
        no_sep_result = self.vision_processor(no_sep_result)
        return benchmark_result, with_sep_result, no_sep_result

    def generate_gt(self, length):
        pass

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_path, self.file_name_list[idx])
        annotation_list = sorted(XmlProcessor(0, file_path).get_annotation(),
                                 key=lambda x: (int(x['paragraph_order'].split('a')[-1]),
                                                x['reading_order']))
        tokenized_result = self.get_tokenizer_result(annotation_list)
        benchmark_result, with_sep_result, no_sep_result = self.get_fig_result(file_path, annotation_list)
        gt = [x for x in range(len(annotation_list)+2)]
        gt_str =" ".join(str(x) for x in gt)

def process_img(lang):
    if lang == 'fr':
        root_path = r'../data/AS_TrainingSet_BnF_NewsEye_v2/'
        store_root_path = r'../data/fr_reading_order/'
    else:
        root_path = r'../data/AS_TrainingSet_NLF_NewsEye_v2/'
        store_root_path = r'../data/fi_reading_order/'

    os.makedirs(store_root_path, exist_ok=True)
    file_name_list = [x for x in os.listdir(root_path) if 'xml' in x]
    for _, file_name in tqdm(enumerate(file_name_list), total=len(file_name_list)):
        file_path = os.path.join(root_path, file_name)
        store_path = store_root_path+file_name.replace('.xml', '')
        os.makedirs(store_path, exist_ok=True)
        annotation_list = XmlProcessor(0, file_path).get_annotation()
        whole_fig = Image.open(file_path.replace('.xml', '.jpg')).convert('RGB')
        background = Image.new('RGB', whole_fig.size, 'black')
        background.save(store_path + '/' + 'cls_eos.jpg')
        raw_image = np.array(whole_fig)
        gray = cv2.cvtColor(raw_image, cv2.COLOR_RGB2GRAY)
        ret, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        h, w = binary.shape
        hors_k = int(math.sqrt(w) * 1.2)
        vert_k = int(math.sqrt(h) * 1.2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hors_k, 1))
        hors = ~cv2.dilate(binary, kernel, iterations=1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_k))
        verts = ~cv2.dilate(binary, kernel, iterations=1)
        borders = cv2.bitwise_or(hors, verts)
        gaussian = cv2.GaussianBlur(borders, (9, 9), 0)
        edges = cv2.Canny(gaussian, 70, 150)
        background_with_sep = Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))
        #        make benchmark:
        for annotation in annotation_list:
            new_background = copy.deepcopy(background)
            new_background_with_sep = copy.deepcopy(background_with_sep)
            location = tuple(annotation['bbox'][0] + annotation['bbox'][2])
            block_img = whole_fig.crop(location)
            new_background.paste(block_img, location)
            new_background_with_sep.paste(block_img, location)
            block_img.save(store_path+'/'+str(annotation['index'])+'_benchmark.jpg')
            new_background.save(store_path + '/' + str(annotation['index']) + '_no_sep.jpg')
            new_background_with_sep.save(store_path + '/' + str(annotation['index']) + '_with_sep.jpg')

        # break

if __name__ == '__main__':
    # executor = submitit.AutoExecutor(
    #     folder='/Utilisateurs/wsun01/logs/')# Can specify cluster='debug' or 'local' to run on the current node instead of on the cluster
    #
    # for lang in ['fr', 'fi']:
    #     executor.update_parameters(
    #         job_name='reading_'+lang,
    #         timeout_min=2160 * 4,
    #         # gpus_per_node=1,
    #         cpus_per_task=5,
    #         # mem_gb=40 * 2,
    #         # slurm_partition='gpu-a6000',
    #         slurm_additional_parameters={
    #             'nodelist': 'l3icalcul09'
    #         }
    #     )
    #     executor.submit(process_img, lang)

    # process_img('fr')
    config = {'lang': r'fi',
              'text_model_name': "dbmdz/bert-base-historic-multilingual-64k-td-cased",
              'max_token_num': 512,
              'device': 'cpu',
              'use_sep_fig': True,
              'vision_model_name': 'microsoft/trocr-base-handwritten',
              'resize': (512, 512),
              'is_benchmark': False,}
    # file_path = r'../data/AS_TrainingSet_BnF_NewsEye_v2/'
    datasetor = ReadingOrderDataset(config)
    a = datasetor[2]




