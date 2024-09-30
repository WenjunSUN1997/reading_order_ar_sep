import os
from utils.xml_reader import XmlProcessor
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import submitit

def organize_reading_order(data_list):
    pass

def draw_based_order(img_path, data_list):
    os.makedirs('../data/reading_order_gt/', exist_ok=True)
    font = ImageFont.load_default()
    font.size = 20
    img = Image.open(img_path.replace('xml', 'jpg')).convert('RGB')
    draw = ImageDraw.Draw(img, "RGB")
    data_list = sorted(data_list, key=lambda x: (int(x['paragraph_order'].split('a')[-1]),
                                                 x['reading_order']))
    # print()
    for index in range(len(data_list)-1):
        draw.rectangle(data_list[index]['bbox'][0] + data_list[index]['bbox'][2],
                       outline='red',
                       width=5)
        draw.line([tuple(data_list[index]['center_point']), tuple(data_list[index+1]['center_point'])],
                  'green', width=10)
        draw.text(data_list[index]['center_point'], str(index), (255, 255, 255), font=font)

    draw.rectangle(data_list[-1]['bbox'][0] + data_list[-1]['bbox'][2],
                   outline='red',
                   width=5)
    draw.text(data_list[-1]['center_point'], str(len(data_list)-1), 'green', font=font)
    img.save(r'../data/reading_order_gt/' + img_path.split('/')[-1].replace('xml', 'png'))

def main():
    executor = submitit.AutoExecutor(folder='/Utilisateurs/wsun01/logs/')  # Can specify cluster='debug' or 'local' to run on the current node instead of on the cluster
    executor.update_parameters(
        job_name='ocr',
        timeout_min=2160 * 4,
        gpus_per_node=1,
        cpus_per_task=5,
        mem_gb=40 * 2,
        # slurm_partition='gpu-a6000',
        slurm_additional_parameters={
            'nodelist': 'l3icalcul10'
        }
    )

if __name__ == "__main__":
    lang = 'fi'
    if lang == 'fi':
        root_path = '../data/AS_TrainingSet_NLF_NewsEye_v2/'
    else:
        root_path = '../data/AS_TrainingSet_BnF_NewsEye_v2/'

    img_path_list = [x for x in os.listdir(root_path) if 'xml' in x]
    for index, img_path in tqdm(enumerate(img_path_list), total=len(img_path_list)):
        img_path = root_path + img_path
        data_list = XmlProcessor(0, img_path).get_annotation()
        draw_based_order(img_path, data_list)

    print('done')
