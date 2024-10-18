import torch
import argparse
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from utils.datasetor_reading_order import ReadingOrderDataset
from model.reading_order_model import ReadingOrderModel

torch.manual_seed(3407)

def train(config):
    epoch_num = 1000
    reading_order_dataseter = ReadingOrderDataset(config)
    reading_order_dataloader = DataLoader(reading_order_dataseter, batch_size=config['batch_size'], shuffle=False)
    reading_order_model = ReadingOrderModel(config)
    reading_order_model.to(config['device'])
    # reading_order_model = torch.nn.DataParallel(reading_order_model)
    if torch.cuda.device_count() > 1:
        reading_order_model.vision_model.to('cuda:1')
    # reading_order_model.half()
    optimizer = torch.optim.AdamW(reading_order_model.parameters(), lr=config['lr'])
    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='min',
                                  factor=0.5,
                                  patience=1,
                                  verbose=True)
    loss_all = [0]
    for epoch_index in range(epoch_num):
        for step, data in tqdm(enumerate(reading_order_dataloader), total=len(reading_order_dataloader)):
            # break
            output = reading_order_model(data)
    #         loss = output['loss']
    #         loss_all.append(loss.item())
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    # print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default='fr', choices=['fr', 'fi'])
    parser.add_argument("--text_model_name", default='dbmdz/bert-base-historic-multilingual-64k-td-cased')
    parser.add_argument("--vision_model_name", default='google/vit-base-patch16-224')
    parser.add_argument("--max_token_num", default=512, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--device", default='cuda:0')
    parser.add_argument("--resize", default='256,256')
    parser.add_argument("--use_sep_fig", default=False, action='store_true')
    parser.add_argument('--is_benchmark', default=False, action='store_true')
    parser.add_argument('--use_seq_background', default=False, action='store_true')
    args = parser.parse_args()
    print(args)
    config = vars(args)
    config['resize'] = tuple(map(int, config['resize'].split(',')))
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
