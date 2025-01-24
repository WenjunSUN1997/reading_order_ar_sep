import torch
import argparse
from model.model_factory import model_factory
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from utils.datasetor_reading_order import get_dataet
from utils.evaluator import get_evaluator
import os
from datetime import datetime
import submitit

torch.manual_seed(3407)

def train(config):
    best_result = None
    best_epoch = 0
    epoch_num = 1000
    reading_order_model = model_factory(config)
    reading_order_model.to(config['device'])
    training_dataseter = get_dataet(config, goal='training')
    train_dataloader = DataLoader(training_dataseter,
                                  batch_size=config['batch_size'],
                                  shuffle=False)
    model_evaluator = get_evaluator(config)
    if config['half']:
        reading_order_model.half()

    optimizer = torch.optim.AdamW(reading_order_model.parameters(), lr=config['lr'])
    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='min',
                                  factor=0.8,
                                  patience=2,
                                  verbose=True)
    loss_all = [0]
    log_root = 'logs/'
    os.makedirs(log_root, exist_ok=True)
    current_time = datetime.now().strftime("%m%d%H%M%S")
    log_folder_path = 'logs/' + current_time + '/'
    os.makedirs(log_folder_path, exist_ok=True)
    for epoch_index in range(epoch_num):
        for step, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            # evaluate_result = model_evaluator(reading_order_model)
            output = reading_order_model(data)
            loss = output['loss']
            loss_all.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (step + 1 ) % int(len(train_dataloader)/2) == 0:
                training_loss = sum(loss_all)/(len(loss_all))
                # print('training loss:', training_loss)
                # print('\n')
                loss_all = []
                evaluate_result = model_evaluator(reading_order_model)
                # print('eva_result:', evaluate_result)
                # print('\n')
                scheduler.step(evaluate_result['loss'])
                if best_result == None or evaluate_result['mac'] > best_result['mac']:
                    best_epoch = epoch_index
                    best_result = evaluate_result
                    torch.save(reading_order_model.state_dict(), log_folder_path + 'best_model.pt')

                output_string = (str(epoch_index) + '\n'
                                 +'training_loss: ' + str(training_loss)+ '\n'
                                 + 'eva_result: ' + str(evaluate_result) + '\n'
                                 + 'bset: \n' + str(best_epoch) +str(best_result) + '\n'
                                 + '_______________________\n')
                print(output_string)
                with open(log_folder_path + 'log.txt', 'a') as f:
                    f.write(output_string)

if __name__ == "__main__":
    # tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    # bert = AutoModel.from_pretrained('bert-base-cased')
    # inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    # outputs = bert(**inputs)
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default='fr', choices=['fr', 'fi'])
    parser.add_argument("--text_model_name", default='dbmdz/bert-base-historic-multilingual-64k-td-cased')
    parser.add_argument("--vision_model_name", default="google/vit-base-patch16-224-in21k")
    # parser.add_argument("--vision_model_name", default="cnn")
    parser.add_argument("--max_token_num", default=256*2, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--query_number", default=32, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--device", default='cuda:0')
    parser.add_argument("--half", default='0')
    parser.add_argument("--resize", default='512,512')
    parser.add_argument("--goal", default='qformer', choices=['single_fig', 'merge_fig', 'benchmark', 'qformer'])
    parser.add_argument("--use_sep_fig", default=False)
    parser.add_argument('--is_benchmark', default=False, action='store_true')
    parser.add_argument('--use_seq_background', default=False, action='store_true')
    parser.add_argument("--dataset_name", default='fullcon', choices=['ar', 'fullcon'])
    args = parser.parse_args()
    print(args)
    config = vars(args)
    config['resize'] = tuple(map(int, config['resize'].split(',')))
    if config['half'] == '1':
        config['half'] = True
    else:
        config['half'] = False

    train(config)
    executor = submitit.AutoExecutor(
        folder='/Utilisateurs/wsun01/logs/')  # Can specify cluster='debug' or 'local' to run on the current node instead of on the cluster
    executor.update_parameters(
        job_name='fr_article_sep_trans',
        timeout_min=2160 * 4,
        gpus_per_node=1,
        cpus_per_task=5,
        mem_gb=100,
        # slurm_partition='gpu-a6000',
        slurm_additional_parameters={
            'nodelist': 'l3icalcul10'
        }
    )
    # executor.submit(train, config )



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
