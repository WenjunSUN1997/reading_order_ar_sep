import os
from random import shuffle
from utils.xml_reader import XmlProcessor

def organize_data(lang):
    if lang == 'fi':
        root_path = '../data/AS_TrainingSet_NLF_NewsEye_v2/'
    else:
        root_path = '../data/AS_TrainingSet_BnF_NewsEye_v2/'

    file_list = [x for x in os.listdir(root_path) if 'xml' in x]
    num = 0
    result = []
    for file_name in file_list:
        path = root_path + file_name
        annotations = XmlProcessor(0, path).get_annotation()
        print(len(annotations))
        print('\n')
        if len(annotations) < 350:
            num += 1
            result.append(path.replace('../', ''))

    shuffle(result)
    training_set = result[:int(len(result) * 0.8)]
    test_set = result[int(len(result) * 0.8):]
    with open('../'+lang+'_training_set.txt', 'w', encoding='utf-8') as f:
        for item in training_set:
            f.write(item+'\n')

    with open('../'+lang+'_test_set.txt', 'w', encoding='utf-8') as f:
        for item in test_set:
            f.write(item+'\n')

    print(len(file_list), num)
    print(num)

if __name__ == '__main__':
    organize_data('fi')