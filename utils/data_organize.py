import os

from Cython.Compiler.Future import annotations

from utils.xml_reader import XmlProcessor

def organize_data(lang):
    if lang == 'fi':
        root_path = '../data/AS_TrainingSet_NLF_NewsEye_v2/'
    else:
        root_path = '../data/AS_TrainingSet_BnF_NewsEye_v2/'

    file_list = [x for x in os.listdir(root_path) if 'xml' in x]
    num = 0
    for file_name in file_list:
        path = root_path + file_name
        annotations = XmlProcessor(0, path).get_annotation()
        print(len(annotations))
        print('\n')
        if len(annotations) < 150:
            num += 1

    print(len(file_list), num)
    print(num)

if __name__ == '__main__':
    organize_data('fr')