import os
import networkx as nx
from xml_reader import XmlProcessor
import numpy as np
from PIL import Image, ImageDraw
import pickle
from tqdm import tqdm

def distance(point_1, point_2):
    pass

def link_only_position(lang, top_n):
    os.makedirs('../data/temp/', exist_ok=True)
    os.makedirs('../data/temp/'+lang+'/', exist_ok=True)
    if lang == 'fr':
        base_path = '../../seperator_ar_sep/article_dataset/AS_TrainingSet_BnF_NewsEye_v2/'
    else:
        base_path = '../../seperator_ar_sep/article_dataset/AS_TrainingSet_NLF_NewsEye_v2/'

    anno_file_name_list = [x for x in os.listdir(base_path) if '.xml' in x]
    for index, anno_file_name in tqdm(enumerate(anno_file_name_list), total=len(anno_file_name_list)):
        xml_processor = XmlProcessor(index, base_path+anno_file_name)
        anno_data = xml_processor.get_annotation()
        net = nx.Graph()
        node_list = [(annotation['index'], annotation) for annotation in anno_data]
        net.add_nodes_from(node_list)
        link_list = []
        for node_index in range(len(node_list)-1):
            top = []
            down = []
            left = []
            right = []
            center_point = node_list[node_index][1]['bbox'][0] + node_list[node_index][1]['bbox'][2]
            center_point = [(center_point[0]+center_point[2])/2, (center_point[1]+center_point[3])/2]
            for next_node_index in range(node_index+1, len(node_list)):
                center_point_next = node_list[next_node_index][1]['bbox'][0] \
                                        + node_list[next_node_index][1]['bbox'][2]
                center_point_next = [(center_point_next[0] + center_point_next[2]) / 2,
                                     (center_point_next[1] + center_point_next[3]) / 2]
                distance = np.linalg.norm(np.array(center_point_next) - np.array(center_point))
                if center_point_next[0] > center_point[0]:
                    down.append((node_list[node_index][0],
                                 node_list[next_node_index][0],
                                 distance,
                                 center_point,
                                 center_point_next))
                elif center_point_next[0] < center_point[0]:
                    top.append((node_list[node_index][0],
                                node_list[next_node_index][0],
                                distance,
                                center_point,
                                center_point_next
                                ))
                if center_point_next[1] < center_point[1]:
                    right.append((node_list[node_index][0],
                                 node_list[next_node_index][0],
                                 distance,
                                  center_point,
                                  center_point_next
                                  ))
                elif center_point_next[1] > center_point[1]:
                    left.append((node_list[node_index][0],
                                 node_list[next_node_index][0],
                                 distance,
                                 center_point,
                                 center_point_next
                                 ))

            top = sorted(top, key=lambda x: x[2])
            down = sorted(down, key=lambda x: x[2])
            left = sorted(left, key=lambda x: x[2])
            right = sorted(right, key=lambda x: x[2])
            for group in [top, down, left, right]:
                if len(group) == 0:
                    continue

                for index_top in range(min([top_n, len(group)])):
                    link_list.append(group[index_top])

        # link_list = set(link_list)
        net.add_edges_from([(x[0], x[1]) for x in link_list])
        img = Image.open(base_path+anno_file_name.replace('xml', 'jpg')).convert("RGB")
        draw = ImageDraw.Draw(img, "RGB")
        for node in net.nodes:
            draw.rectangle(net.nodes[node]['bbox'][0]+net.nodes[node]['bbox'][2], outline='green', width=10)

        # plt.imshow(img)
        for link in link_list:
            draw.line(link[3]+link[4], 'red', width=5)

        save_path = '../data/temp/' + lang + '/' +anno_file_name.replace('.xml', '') + '/'
        os.makedirs(save_path, exist_ok=True)
        img.save(save_path + str(top_n) + '.png')
        pickle.dump(net, open(save_path + str(top_n) + '.nx', 'wb'))
        # print()

if __name__ == "__main__":
    lang = 'fr'
    top_n = 1
    link_only_position(lang, top_n)