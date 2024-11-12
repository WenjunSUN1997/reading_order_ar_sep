import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoConfig
from utils.datasetor_reading_order import ReadingOrderDataset
from torch.utils.data.dataloader import DataLoader

class ModlePrototype(nn.Module):
    def __init__(self, config):
        super(ModlePrototype, self).__init__()
        self.config = config
        self.lang_model = AutoModel.from_pretrained(config['text_model_name'])
        if config['vision_model_name'] != 'cnn':
            # self.vision_config = AutoConfig.from_pretrained(config['vision_model_name'])
            # self.vision_config.image_size = config['resize'][0]
            # self.vision_model = AutoModel.from_config(self.vision_config)
            self.vision_model = AutoModel.from_pretrained(config['vision_model_name'], torch_dtype=torch.float16)
            vision_dim = self.vision_model.config.hidden_size
        else:
            self.vision_model = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=30, stride=20, padding=1,
                                                bias=False)
            vision_dim = 625

        self.loss_func = torch.nn.CrossEntropyLoss()
        text_dim = self.lang_model.config.hidden_size
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
                                                                     nhead=5,
                                                                     batch_first=True)
        self.commu_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=commu_dim,
                                                                    nhead=7,
                                                                    batch_first=True)
        self.text_encoder = torch.nn.TransformerEncoder(self.text_encoder_layer,
                                                        num_layers=2)
        self.vision_encoder = torch.nn.TransformerEncoder(self.vision_encoder_layer,
                                                          num_layers=2)
        self.commu_encoder = torch.nn.TransformerEncoder(self.commu_encoder_layer,
                                                         num_layers=2)
        # self.normal = torch.nn.LayerNorm(self.linear_dim)
        self.linear = torch.nn.Linear(self.linear_dim, 2)
        self.activate = nn.Softmax(dim=-1)