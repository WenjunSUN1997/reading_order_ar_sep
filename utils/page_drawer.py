import os
from utils.xml_reader import XmlProcessor
from PIL import Image, ImageDraw
from tqdm import tqdm

def draw_based_order(img_path, data_list):
    os.makedirs('../data/reading_order_gt/', exist_ok=True)
    img = Image.open(img_path.replace('xml', 'jpg')).convert('RGB')
    draw = ImageDraw.Draw(img, "RGB")
    data_list = sorted(data_list, key=lambda x: (x['reading_order']))
    for index in range(len(data_list)-1):
        draw.rectangle(data_list[index]['bbox'][0] + data_list[index]['bbox'][2],
                       fill=(0, 0, 255),
                       outline='red',
                       width=5)
        draw.line([tuple(data_list[index]['center_point']), tuple(data_list[index+1]['center_point'])],
                  'green', width=10)

    draw.rectangle(data_list[-1]['bbox'][0] + data_list[-1]['bbox'][2],
                   fill=(0, 0, 255),
                   outline='red',
                   width=5)
    img.save(r'../data/reading_order_gt/' + img_path.replace('xml', 'jpg'))

if __name__ == "__main__":
    lang = 'fr'
    if lang == 'fr':
        img_path_list = '../data/AS_TrainingSet_NLF_NewsEye_v2/'
    else:
        img_path_list = '../data/AS_TrainingSet_BnF_NewsEye_v2/'

    img_path_list = [x for x in os.listdir(img_path_list) if 'xml' in x]
    for index, img_path in tqdm(enumerate(img_path_list), total=len(img_path_list)):
        data_list = XmlProcessor(0, img_path).get_annotation()
        draw_based_order(img_path, data_list)

    print('done')
