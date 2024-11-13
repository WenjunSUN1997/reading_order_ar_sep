import torch
import argparse
from model.model_factory import model_factory
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from utils.datasetor_reading_order import ReadingOrderDataset, ArticleDataset
from utils.evaluator import Evaluator
import os
from datetime import datetime

torch.manual_seed(3407)

def train(config):
    log_root = 'logs/'
    os.makedirs(log_root, exist_ok=True)
    current_time = datetime.now().strftime("%m%d%H%M%S")
    log_folder_path = 'logs/'+ current_time + '/'
    os.makedirs(log_folder_path, exist_ok=True)
    best_result = None
    best_epoch = 0
    epoch_num = 1000
    training_dataseter = ArticleDataset(config, goal='training')
    test_dataseter = ArticleDataset(config, goal='test')
    train_dataloader = DataLoader(training_dataseter, batch_size=config['batch_size'], shuffle=False)
    test_dataloader = DataLoader(test_dataseter, batch_size=config['batch_size'], shuffle=False)
    reading_order_model = model_factory(config)
    reading_order_model.to(config['device'])
    model_evaluator = Evaluator()
    # reading_order_model = torch.nn.DataParallel(reading_order_model)
    # if torch.cuda.device_count() > 1 and config['vision_model_name'] !='cnn':
    #     reading_order_model.vision_model.to('cuda:1')

    if config['half']:
        reading_order_model.half()

    optimizer = torch.optim.AdamW(reading_order_model.parameters(), lr=config['lr'])
    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='min',
                                  factor=0.5,
                                  patience=2,
                                  verbose=True)
    loss_all = []
    for epoch_index in range(epoch_num):
        for step, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            # break
            output = reading_order_model(data)
            loss = output['loss']
            loss_all.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(str(epoch_index) + str(sum(loss_all) / len(loss_all)))
        evaluate_result = model_evaluator.model_evaluate(reading_order_model, test_dataloader)
        scheduler.step(evaluate_result['loss'])
        if best_result == None or evaluate_result['mac'] > best_result['mac']:
            best_mac = evaluate_result['mac']
            best_epoch = epoch_index
            best_result = evaluate_result
            torch.save(reading_order_model.state_dict(), log_folder_path + 'best_model.pt')

        output_string = str(epoch_index) + '\n' + str(evaluate_result) + '\n' + 'bset: \n' + str(best_epoch) +str(best_result)
        with open(log_folder_path + 'log.txt', 'a') as f:
            f.write(output_string)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default='fr', choices=['fr', 'fi'])
    parser.add_argument("--text_model_name", default='dbmdz/bert-base-historic-multilingual-64k-td-cased')
    parser.add_argument("--vision_model_name", default='cnn')
    parser.add_argument("--max_token_num", default=256*2, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--device", default='cuda:0')
    parser.add_argument("--half", default='0')
    parser.add_argument("--resize", default='512,512')
    parser.add_argument("--goal", default='develop', choices=['develop', 'benchmark'])
    parser.add_argument("--use_sep_fig", default=False)
    parser.add_argument('--is_benchmark', default=False, action='store_true')
    parser.add_argument('--use_seq_background', default=False, action='store_true')
    args = parser.parse_args()
    print(args)
    config = vars(args)
    config['resize'] = tuple(map(int, config['resize'].split(',')))
    if config['half'] == '1':
        config['half'] = True
    else:
        config['half'] = False

    train(config)

    # config = {'lang': r'fi',
    #           'text_model_name': "dbmdz/bert-base-historic-multilingual-64k-td-cased",
    #           'max_token_num': 512,
    #           'device': 'cuda:0',
    #           'use_sep_fig': True,
    #           'vision_model_name': 'hustvl/yolos-tiny',
    #           'resize': (512, 512),
    #           'is_benchmark': False,
    #           'use_seq_background': True,
    #           'batch_size': 1, }
