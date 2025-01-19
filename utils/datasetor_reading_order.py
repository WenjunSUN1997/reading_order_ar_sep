import torch
import submitit
import copy
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.xml_reader import XmlProcessor
import os
from transformers import AutoTokenizer,  AutoProcessor
from PIL import Image, ImageDraw
import cv2
import math
import numpy as np

def get_dataet(config, goal, file_name=None):
    if config['dataset_name'] == 'ar':
        return ArticleDataset(config, goal=goal, file_name=file_name)

    if config['dataset_name'] == 'fullcon':
        return BlockFullConDataset(config, goal=goal, file_name=file_name)

class DatasetPrototype(Dataset):
    def __init__(self, config, goal, file_name=None):
        self.config = config
        self.lang = config['lang']
        if goal == 'training':
            with open(config['lang'] + '_' + goal + '_set.txt', 'r') as f:
                self.file_name_list = f.read().split('\n')[:-1]
        else:
            self.file_name_list = [file_name]

        if self.lang == 'fi':
            # self.root_path = r'data/AS_TrainingSet_NLF_NewsEye_v2/'
            self.img_root_path = r'data/fi_reading_order/'
        else:
            # self.root_path = r'data/AS_TrainingSet_BnF_NewsEye_v2'
            self.img_root_path = r'data/fr_reading_order/'

        # self.file_name_list = [x for x in os.listdir(self.root_path) if 'xml' in x]
        self.tokenizer = AutoTokenizer.from_pretrained(config['text_model_name'])
        self.max_token_num = config['max_token_num']
        self.device = config['device']
        self.use_sep_fig = config['use_sep_fig']
        if self.config['vision_model_name'] != 'cnn':
            self.vision_processor = AutoProcessor.from_pretrained(config['vision_model_name'])
        else:
            self.vision_processor = AutoProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

        self.vision_processor.do_resize = False
        self.resize = config['resize']
        self.is_benchmark = config['is_benchmark']

    def generate_gt(self, annotation_list):
        length = len(annotation_list) + 2
        result = []
        end = 1
        for i in range(length-1):
            temp = [0] * length
            temp[end] = 1
            end += 1
            result.append(temp)

        result.append([0] * length)
        result = torch.tensor(result).to(self.device)
        article_matrix = []
        for annotation in annotation_list:
            matrix_this_item = []
            for i in range(len(annotation_list)):
                if annotation['paragraph_order'] == annotation_list[i]['paragraph_order']:
                    matrix_this_item.append(1)
                else:
                    matrix_this_item.append(0)
            article_matrix.append(matrix_this_item)

        article_matrix = torch.tensor(article_matrix).to(self.device)
        return result, article_matrix

class ReadingOrderDataset(DatasetPrototype):
    def __init__(self, config, goal):
        super(ReadingOrderDataset, self).__init__(config, goal)

    def get_tokenizer_result(self, annotation_list):
        text_list = [x['text'] for x in annotation_list]
        # text_list.insert(0, '###')
        # text_list.append('###')
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
            benchmark_result.append(Image.open(img_folder + str(annotation['index'])+'_benchmark.jpg').convert('RGB').resize(self.resize))
            with_sep_result.append(Image.open(img_folder + str(annotation['index']) + '_with_sep.jpg').convert('RGB').resize(self.resize))
            no_sep_result.append(Image.open(img_folder + str(annotation['index']) + '_no_sep.jpg').convert('RGB').resize(self.resize))

        background = Image.open(img_folder+'cls_eos.jpg').convert('RGB').resize(self.resize)
        background_seq = Image.open(img_folder + 'background_sep.jpg').convert('RGB').resize(self.resize)
        # benchmark_result.insert(0, background)
        # benchmark_result.append(background)
        # with_sep_result.insert(0, background)
        # with_sep_result.append(background)
        # no_sep_result.insert(0, background)
        # no_sep_result.append(background)
        background_seq_result = self.vision_processor([background_seq], return_tensors="pt", do_resize=False)
        benchmark_result = self.vision_processor(benchmark_result, return_tensors="pt", do_resize=False)
        with_sep_result = self.vision_processor(with_sep_result, return_tensors="pt", do_resize=False)
        no_sep_result = self.vision_processor(no_sep_result, return_tensors="pt", do_resize=False)
        return benchmark_result, with_sep_result, no_sep_result, background_seq_result

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, idx):
        file_path = self.file_name_list[idx]
        # annotation_list = sorted(XmlProcessor(0, file_path).get_annotation(),
        #                          key=lambda x: (int(x['paragraph_order'].split('a')[-1]),
        #                                         x['reading_order']))
        annotation_list = XmlProcessor(0, file_path).get_annotation()
        # print(len(annotation_list))
        gt_matrix, article_matrix = self.generate_gt(annotation_list)
        tokenized_result = self.get_tokenizer_result(annotation_list)
        benchmark_result, with_sep_result, no_sep_result, background_seq_result = (
            self.get_fig_result(file_path, annotation_list))
        gt = [x for x in range(len(annotation_list)+2)]
        gt_str =" ".join(str(x) for x in gt)

        inputs =  {'input_ids': tokenized_result['input_ids'].to(self.device),
                    'attention_mask': tokenized_result['attention_mask'].to(self.device),
                    'benchmark_fig': benchmark_result['pixel_values'].to(self.device),
                    'with_sep_fig': with_sep_result['pixel_values'].to(self.device),
                    'no_sep_fig': no_sep_result['pixel_values'].to(self.device),
                    'background_seq': background_seq_result['pixel_values'].to(self.device),
                    'gt_matrix': gt_matrix.to(self.device),
                    'gt_str': torch.tensor(gt).to(self.device),
                    'article_matirx': article_matrix.to(self.device),
                    }
        if self.config['half']:
            for key,item in inputs.items():
                inputs[key] = item.half()

        return inputs

class ArticleDataset(DatasetPrototype):
    def __init__(self, config, goal, file_name=None):
        super(ArticleDataset, self).__init__(config, goal, file_name=file_name)
        self.data_grouped_index = self._organize_by_index()

    def _organize_by_index(self):
        index_content_dict = {}
        index = 0
        for file_path in tqdm(self.file_name_list):
            annotation_list = XmlProcessor(0, file_path).get_annotation()
            if len(annotation_list) >= 1000:
                continue

            _, article_matrix = self.generate_gt(annotation_list)
            img_folder = os.path.join(self.img_root_path, file_path.split('/')[-1].replace('.xml', '/'))
            for index_1 in range(len(annotation_list)-1):
                for index_2 in range(index_1+1, len(annotation_list)):
                    text = [annotation_list[index_1]['text'], annotation_list[index_2]['text']]
                    benchmark = [img_folder + str(annotation_list[index_1]['index']) + '_benchmark.jpg',
                                 img_folder + str(annotation_list[index_2]['index']) + '_benchmark.jpg']
                    with_sep =  [img_folder + str(annotation_list[index_1]['index']) + '_with_sep.jpg',
                                 img_folder + str(annotation_list[index_2]['index']) + '_with_sep.jpg']
                    no_sep = [img_folder + str(annotation_list[index_1]['index']) + '_no_sep.jpg',
                              img_folder + str(annotation_list[index_2]['index']) + '_no_sep.jpg']
                    gt =1 if annotation_list[index_1]['paragraph_order'] == annotation_list[index_2]['paragraph_order'] else 0
                    index_content_dict[index] = {'text': text,
                                                 'benchmark': benchmark,
                                                 'with_sep': with_sep,
                                                 'no_sep': no_sep,
                                                 'gt': gt,
                                                 'gt_matrix': article_matrix
                                                 }
                    index += 1

        return index_content_dict

    def __len__(self):
        return len(self.data_grouped_index)

    def __getitem__(self, idx):
        data = self.data_grouped_index[idx]
        tokenize_result = self.tokenizer(data['text'],
                                         max_length=self.max_token_num,
                                         truncation=True,
                                         return_tensors='pt',
                                         padding='max_length')
        benchmark_0 = Image.open(data['benchmark'][0]).convert('RGB').resize(self.resize)
        benchmark_1 = Image.open(data['benchmark'][1]).convert('RGB').resize(self.resize)

        with_sep_0 = Image.open(data['with_sep'][0]).convert('RGB').resize(self.resize)
        with_sep_1 = Image.open(data['with_sep'][1]).convert('RGB').resize(self.resize)

        no_sep_0 = Image.open(data['no_sep'][0]).convert('RGB').resize(self.resize)
        no_sep_1 = Image.open(data['no_sep'][1]).convert('RGB').resize(self.resize)
        benchmark_result = self.vision_processor([benchmark_0, benchmark_1], return_tensors="pt", do_resize=False)
        if self.config['goal'] == 'single_fig':
            with_sep_result = self.vision_processor([with_sep_0, with_sep_1], return_tensors="pt", do_resize=False)
            no_sep_result = self.vision_processor([no_sep_0, no_sep_1], return_tensors="pt", do_resize=False)

        elif self.config['goal'] in ['merge_fig', 'qformer']:
            with_sep_0.paste(with_sep_1)
            no_sep_0.paste(no_sep_1)
            with_sep_result = self.vision_processor([with_sep_0], return_tensors="pt", do_resize=False)
            no_sep_result = self.vision_processor([no_sep_0], return_tensors="pt", do_resize=False)

        return {'input_ids': tokenize_result['input_ids'].to(self.device),
                'attention_mask': tokenize_result['attention_mask'].to(self.device),
                'benchmark_fig': benchmark_result['pixel_values'].to(self.device),
                'with_sep_fig': with_sep_result['pixel_values'].to(self.device),
                'no_sep_fig': no_sep_result['pixel_values'].to(self.device),
                'gt': torch.tensor(data['gt']).to(self.device),
                'num_paragraph': torch.tensor(data['gt_matrix'].size(0)).to(self.device)
                }

class BlockFullConDataset(DatasetPrototype):
    def __init__(self, config, goal, file_name=None):
        super(BlockFullConDataset, self).__init__(config, goal, file_name=file_name)
        self.data = self._organize_data()

    def _organize_data(self):
        all_index = 0
        data_dict = {}
        for file_path in tqdm(self.file_name_list):
            img_folder = os.path.join(self.img_root_path, file_path.split('/')[-1].replace('.xml', '/'))
            annotation_list = XmlProcessor(0, file_path).get_annotation()
            link_result_list = self._link_based_on_dis(annotation_list)
            self._draw(file_path,annotation_list, link_result_list)
            assert len(annotation_list) == len(link_result_list)
            for anno_index in link_result_list:
                if link_result_list[anno_index]['right'] is not None:
                    right_text = [annotation_list[anno_index]['text'],
                                  link_result_list[anno_index]['right']['text']]
                    right_benchmark = [img_folder + str(annotation_list[anno_index]['index']) + '_benchmark.jpg',
                                       img_folder + str(link_result_list[anno_index]['right']['index']) + '_benchmark.jpg']

                    right_with_sep = [img_folder + str(annotation_list[anno_index]['index']) + '_with_sep.jpg',
                                      img_folder + str(link_result_list[anno_index]['right']['index']) + '_with_sep.jpg']
                    right_no_sep = [img_folder + str(annotation_list[anno_index]['index']) + '_no_sep.jpg',
                                    img_folder + str(link_result_list[anno_index]['right']['index']) + '_no_sep.jpg']
                    right_gt = 1 if annotation_list[anno_index]['paragraph_order'] == \
                                    link_result_list[anno_index]['right']['paragraph_order'] else 0
                    data_dict[all_index] = {'text': right_text,
                                            'benchmark': right_benchmark,
                                            'with_sep': right_with_sep,
                                            'no_sep': right_no_sep,
                                            'gt': right_gt}
                    all_index += 1

                if link_result_list[anno_index]['bottom'] is not None:
                    bottom_text = [annotation_list[anno_index]['text'],
                                  link_result_list[anno_index]['bottom']['text']]
                    bottom_benchmark = [img_folder + str(annotation_list[anno_index]['index']) + '_benchmark.jpg',
                                        img_folder + str(link_result_list[anno_index]['bottom']['index']) + '_benchmark.jpg']
                    bottom_with_sep = [img_folder + str(annotation_list[anno_index]['index']) + '_with_sep.jpg',
                                      img_folder + str(link_result_list[anno_index]['bottom']['index']) + '_with_sep.jpg']
                    bottom_no_sep = [img_folder + str(annotation_list[anno_index]['index']) + '_no_sep.jpg',
                                     img_folder + str(link_result_list[anno_index]['bottom']['index']) + '_no_sep.jpg']
                    bottom_gt = 1 if annotation_list[anno_index]['paragraph_order'] == link_result_list[anno_index]['bottom']['paragraph_order'] else 0
                    data_dict[all_index] = {'text': bottom_text,
                                            'benchmark': bottom_benchmark,
                                            'with_sep': bottom_with_sep,
                                            'no_sep': bottom_no_sep,
                                            'gt': bottom_gt}
                    all_index += 1

        return data_dict

    def _link_based_on_dis(self, annotation_list):
        def euclidean_distance(point1, point2):
            a, b = point1
            c, d = point2
            distance = math.sqrt((c - a) ** 2 + (d - b) ** 2)
            return distance

        def judge(anno, anno_compare):
            if anno['index'] == anno_compare['index']:
                return False
            # if abs(anno_compare['center_point'][0]-anno['center_point'][0]) >= 500:
            #     return False
            # if abs(anno_compare['center_point'][1]-anno['center_point'][1]) >= 500:
            #     return False
            if euclidean_distance(anno_compare['center_point'], anno['center_point']) > 1500:
                return False

            return True

        result = {}
        for index, anno in enumerate(annotation_list):
            min_right = 1000000
            min_right_anno = None
            min_bottom = 1000000
            min_bottom_anno = None
            for index_compare, anno_compare in enumerate(annotation_list):
                if not judge(anno, anno_compare):
                    continue
                right_dis = anno_compare['center_point'][0] - anno['center_point'][0]
                bottom_dis = anno_compare['center_point'][1] - anno['center_point'][1]
                if right_dis >= 0:
                    if right_dis < min_right:
                        min_right = right_dis
                        min_right_anno = anno_compare

                if bottom_dis >= 0:
                    if bottom_dis < min_bottom:
                        min_bottom = bottom_dis
                        min_bottom_anno = anno_compare

            result[index] = {'right': min_right_anno,
                             'bottom': min_bottom_anno}

        return result

    def _draw(self, file_path, anno_list, link_result_list):
        os.makedirs('data/link_result/', exist_ok=True)
        img_draw = Image.open(file_path.replace('xml', 'jpg')).convert("RGB")
        draw = ImageDraw.Draw(img_draw, "RGB")
        for anno_index in link_result_list:
            anno = anno_list[anno_index]
            draw.rectangle([(anno['bbox'][0][0], anno['bbox'][0][1]), (anno['bbox'][2][0], anno['bbox'][2][1])], outline='green', width=20)
            link_result = link_result_list[anno_index]
            if link_result['right'] is not None:
                draw.line([(anno['center_point'][0], anno['center_point'][1]),
                           (link_result['right']['center_point'][0], link_result['right']['center_point'][1])], 'red', width=10)
            if link_result['bottom'] is not None:
                draw.line([(anno['center_point'][0], anno['center_point'][1]),
                           (link_result['bottom']['center_point'][0], link_result['bottom']['center_point'][1])], 'red', width=10)

        img_draw.save('data/link_result/'+file_path.split('/')[-1]+'.jpg')


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data_grouped_index[idx]
        tokenize_result = self.tokenizer(data['text'],
                                         max_length=self.max_token_num,
                                         truncation=True,
                                         return_tensors='pt',
                                         padding='max_length')
        benchmark_0 = Image.open(data['benchmark'][0]).convert('RGB').resize(self.resize)
        benchmark_1 = Image.open(data['benchmark'][1]).convert('RGB').resize(self.resize)

        with_sep_0 = Image.open(data['with_sep'][0]).convert('RGB').resize(self.resize)
        with_sep_1 = Image.open(data['with_sep'][1]).convert('RGB').resize(self.resize)

        no_sep_0 = Image.open(data['no_sep'][0]).convert('RGB').resize(self.resize)
        no_sep_1 = Image.open(data['no_sep'][1]).convert('RGB').resize(self.resize)
        benchmark_result = self.vision_processor([benchmark_0, benchmark_1], return_tensors="pt", do_resize=False)
        if self.config['goal'] == 'single_fig':
            with_sep_result = self.vision_processor([with_sep_0, with_sep_1], return_tensors="pt", do_resize=False)
            no_sep_result = self.vision_processor([no_sep_0, no_sep_1], return_tensors="pt", do_resize=False)

        elif self.config['goal'] in ['merge_fig', 'qformer']:
            with_sep_0.paste(with_sep_1)
            no_sep_0.paste(no_sep_1)
            with_sep_result = self.vision_processor([with_sep_0], return_tensors="pt", do_resize=False)
            no_sep_result = self.vision_processor([no_sep_0], return_tensors="pt", do_resize=False)

        return {'input_ids': tokenize_result['input_ids'].to(self.device),
                'attention_mask': tokenize_result['attention_mask'].to(self.device),
                'benchmark_fig': benchmark_result['pixel_values'].to(self.device),
                'with_sep_fig': with_sep_result['pixel_values'].to(self.device),
                'no_sep_fig': no_sep_result['pixel_values'].to(self.device),
                'gt': torch.tensor(data['gt']).to(self.device)}

def process_img(lang):
    if lang == 'fr':
        root_path = r'../data/AS_TrainingSet_BnF_NewsEye_v2/'
        store_root_path = r'../data/fr_reading_order/'
    else:
        root_path = r'../data/AS_TrainingSet_NLF_NewsEye_v2/'
        store_root_path = r'../data/fi_reading_order/'

    os.makedirs(store_root_path, exist_ok=True)
    finished_file_list = os.listdir(store_root_path)
    file_name_list = [x for x in os.listdir(root_path) if 'xml' in x
                      and x.replace('.xml', '')  not in finished_file_list]
    for _, file_name in tqdm(enumerate(file_name_list), total=len(file_name_list)):
        file_path = os.path.join(root_path, file_name)
        store_path = store_root_path+file_name.replace('.xml', '')
        os.makedirs(store_path, exist_ok=True)
        annotation_list = XmlProcessor(0, file_path).get_annotation()
        whole_fig = Image.open(file_path.replace('.xml', '.jpg')).convert('RGB')
        background = Image.new('RGB', whole_fig.size, 'black')
        background.resize((512, 512)).save(store_path + '/' + 'cls_eos.jpg')
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
        background_with_sep.resize((512, 512)).save(store_path+'/' + 'background_sep.jpg')
        # continue
        #        make benchmark:
        for annotation in annotation_list:
            new_background = copy.deepcopy(background)
            new_background_with_sep = copy.deepcopy(background_with_sep)
            location = tuple(annotation['bbox'][0] + annotation['bbox'][2])
            block_img = whole_fig.crop(location)
            new_background.paste(block_img, location)
            new_background_with_sep.paste(block_img, location)
            block_img.resize((512, 512)).save(store_path+'/'+str(annotation['index'])+'_benchmark.jpg')
            new_background.resize((512, 512)).save(store_path + '/' + str(annotation['index']) + '_no_sep.jpg')
            new_background_with_sep.resize((512, 512)).save(store_path + '/' + str(annotation['index']) + '_with_sep.jpg')

        # break

def convert_img_to_nparray(lang):
    if lang == 'fr':
        store_root_path = r'../data/fr_reading_order/'
    else:
        store_root_path = r'../data/fi_reading_order/'

    folder_list = os.listdir(store_root_path)
    for _, folder in tqdm(enumerate(folder_list), total=len(folder_list)):
        file_list = [x for x in os.listdir(os.path.join(store_root_path, folder)) if 'jpg' in x]
        for file_name in file_list:
            img = Image.open(os.path.join(store_root_path, folder, file_name)).convert('RGB')
            img = np.array(img)
            np.save(os.path.join(store_root_path, folder, file_name.replace('.jpg', '.npy')), img)

def create_dataset(lang):
    pass

if __name__ == '__main__':
    process_img('fi')
    executor = submitit.AutoExecutor(
        folder='/Utilisateurs/wsun01/logs/')# Can specify cluster='debug' or 'local' to run on the current node instead of on the cluster

    for lang in ['fr', 'fi']:
        executor.update_parameters(
            job_name='reading_'+lang,
            timeout_min=2160 * 4,
            # gpus_per_node=1,
            cpus_per_task=5,
            # mem_gb=40 * 2,
            # slurm_partition='gpu-a6000',
            slurm_additional_parameters={
                'nodelist': 'l3icalcul09'
            }
        )
        executor.submit(process_img, lang)
    # config = {'lang': r'fi',
    #           'text_model_name': "dbmdz/bert-base-historic-multilingual-64k-td-cased",
    #           'max_token_num': 512,
    #           'device': 'cpu',
    #           'use_sep_fig': True,
    #           'vision_model_name': 'microsoft/trocr-base-handwritten',
    #           'resize': (512, 512),
    #           'is_benchmark': False,}
    # datasetor = ArticleDataset(config, 'training')
    # a = datasetor[2]

   # def _organize_by_article(self, annotation_list, file_path):
   #      tokenized_result = []
   #      attention_mask_result = []
   #      benchmark_result = []
   #      with_sep_result = []
   #      no_sep_result = []
   #      background_seq_result = []
   #      gt = []
   #      # gt_matrix, article_matrix = self.generate_gt(annotation_list)
   #      tokenized = self.get_tokenizer_result(annotation_list)
   #      benchmark, with_sep, no_sep, background_seq =  self.get_fig_result(file_path, annotation_list)
   #      for index_1 in range(len(annotation_list)-1):
   #          for index_2 in range(index_1+1,len(annotation_list)):
   #              tokenized_result.append(torch.stack((tokenized['input_ids'][index_1], tokenized['input_ids'][index_2]), dim=0))
   #              attention_mask_result.append(torch.stack((tokenized['attention_mask'][index_1], tokenized['attention_mask'][index_2])))
   #              benchmark_result.append(torch.stack((benchmark['pixel_values'][index_1], benchmark['pixel_values'][index_2])))
   #              with_sep_result.append(torch.stack((with_sep['pixel_values'][index_1], with_sep['pixel_values'][index_2])))
   #              no_sep_result.append(torch.stack((no_sep['pixel_values'][index_1], no_sep['pixel_values'][index_2])))
   #              background_seq_result.append(background_seq['pixel_values'])
   #              if annotation_list[index_1]['paragraph_order'] == annotation_list[index_2]['paragraph_order']:
   #                  gt.append(1)
   #              else:
   #                  gt.append(0)
   #
   #      return tokenized_result, attention_mask_result, with_sep_result, no_sep_result, background_seq_result, gt
   #
   #  def _organize_by_article_loop(self):
#         tokenized_result = []
#         attention_mask_result = []
#         gt_result = []
#         with_sep_result = []
#         no_sep_result = []
#         for file_path in tqdm(self.file_name_list):
#             annotation_list = XmlProcessor(0, '../'+file_path).get_annotation()
#             (tokenized_this_article, attention_mask_this_article,
#              with_sep_this_article, no_sep_this_article,
#              background_seq_this_article, gt_this_article) = self._organize_by_article(annotation_list, file_path)
#             tokenized_result += tokenized_this_article
#             attention_mask_result += attention_mask_this_article
#             with_sep_result += with_sep_this_article
#             no_sep_result += no_sep_this_article
#             gt_result += gt_this_article
#         return tokenized_result, attention_mask_result, with_sep_result, no_sep_result, gt_result
   #
