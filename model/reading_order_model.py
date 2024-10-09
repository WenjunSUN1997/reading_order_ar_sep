import torch
import torch.nn as nn
from transformers import AutoModel
from utils.datasetor_reading_order import ReadingOrderDataset
from torch.utils.data.dataloader import DataLoader

class ReadingOrderModel(nn.Module):
    def __init__(self, config):
        super(ReadingOrderModel, self).__init__()
        self.lang_model = AutoModel.from_pretrained(config['text_model_name'])
        self.vision_model = AutoModel.from_pretrained(config['vision_model_name'])
        self.loss_func = torch.nn.CrossEntropyLoss()
        text_dim = self.lang_model.config.hidden_size
        vision_dim = self.vision_model.config.hidden_size
        commu_dim = text_dim + vision_dim
        if config['use_seq_background']:
            self.use_seq_background = True
            self.background_vision_model = AutoModel.from_pretrained(config['vision_model_name'])
            # commu_dim = text_dim + 2 * vision_dim
            self.linear_dim = 2 * commu_dim + vision_dim
        else:
            self.use_seq_background = False
            self.linear_dim = 2 * commu_dim
        self.text_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=text_dim,
                                                                   nhead=4,
                                                                   batch_first=True)
        self.vision_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=vision_dim,
                                                                     nhead=4,
                                                                     batch_first=True)
        self.commu_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=commu_dim,
                                                                     nhead=4,
                                                                     batch_first=True)
        self.text_encoder = torch.nn.TransformerEncoder(self.text_encoder_layer,
                                                        num_layers=2)
        self.vision_encoder = torch.nn.TransformerEncoder(self.vision_encoder_layer,
                                                          num_layers=2)
        self.commu_encoder = torch.nn.TransformerEncoder(self.commu_encoder_layer,
                                                         num_layers=2)
        self.linear = torch.nn.Linear(self.linear_dim, 2)
        self.activate = nn.Softmax(dim=-1)

    def classificaiton_loss(self, output, gt):
        loss = []
        item_num = output.shape[0]
        for i in range(item_num):
            loss_value = self.loss_func(output[i], gt[i])
            loss.append(loss_value)

        return sum(loss)

    def enter_exit_loss(self, output):
        loss = []
        item_num = output.shape[0]
        link_sub_matirx = output[:, :, 1]
        for i in range(item_num-1):
            loss_value = torch.sum(link_sub_matirx[i, :]) + torch.sum(link_sub_matirx[:, i]) -1
            loss.append(torch.abs(loss_value))

        return sum(loss)

    def decode(self, output):
        rout = []
        item_num = output.shape[0]
        link_sub_matirx = output[:, :-1, 1]
        while True:
            pass

    def circle_loss(self, output):
        pass

    def organize(self, all_embedding, background_sep_embedding=None):
        item_num, dim = all_embedding.shape
        result = torch.zeros((item_num, item_num, self.linear_dim)).to(all_embedding.device)
        for i in range(item_num):
            for j in range(item_num):
                if self.use_seq_background:
                    result[i, j, :] = torch.cat((background_sep_embedding, all_embedding[i, :], all_embedding[j, :]),
                                                dim=0)
                else:
                    result[i, j, :] = torch.cat((all_embedding[i, :], all_embedding[j, :]),
                                                dim=0)

        return result

    def forward(self, input):
        text_emdedding = self.lang_model(input_ids=input['input_ids'].squeeze(0),
                                            attention_mask=input['attention_mask'].squeeze(0))['last_hidden_state']
        text_emdedding = torch.mean(text_emdedding, dim=1)
        vision_emdedding = self.vision_model(input['benchmark_fig'].squeeze(0))['last_hidden_state']
        vision_emdedding = torch.mean(vision_emdedding, dim=1)
        text_embedding_after_encoder = self.text_encoder(text_emdedding)
        all_embedding = torch.cat((text_embedding_after_encoder, vision_emdedding), dim=1)
        all_embedding = self.commu_encoder(all_embedding)
        if self.use_seq_background:
            embedding_background = self.background_vision_model(input['background_seq'].squeeze(0))['last_hidden_state']
            embedding_background = torch.mean(embedding_background, dim=1).squeeze(0)

        linear_input = self.organize(all_embedding, embedding_background)
        classification_result = self.activate(self.linear(linear_input))
        classification_loss = self.classificaiton_loss(classification_result, input['gt_matrix'].squeeze(0))
        enter_loss = self.enter_exit_loss(classification_result)
        route = self.decode(classification_result)
        print()


if __name__ == "__main__":
    config = {'lang': r'fi',
              'text_model_name': "dbmdz/bert-base-historic-multilingual-64k-td-cased",
              'max_token_num': 512,
              'device': 'cuda:0',
              'use_sep_fig': True,
              'vision_model_name': 'hustvl/yolos-tiny',
              'resize': (512, 512),
              'is_benchmark': False,
              'use_seq_background': True,
              'batch_size': 1,}
    dataset = ReadingOrderDataset(config)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
    reading_order_model = ReadingOrderModel(config)
    reading_order_model.to(config['device'])
    for data in dataloader:
        output = reading_order_model(data)

