import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class ModlePrototype(nn.Module):
    def __init__(self, config):
        super(ModlePrototype, self).__init__()
        self.config = config
        self.lang_model = AutoModel.from_pretrained(config['text_model_name'])
        if config['vision_model_name'] != 'cnn':
            vision_config = AutoConfig.from_pretrained(config['vision_model_name'])
            vision_config.image_size = config['resize'][0]
            self.vision_model = AutoModel.from_config(vision_config)
            # self.vision_model = AutoModel.from_pretrained(config['vision_model_name'], torch_dtype=torch.float16)
            vision_dim = self.vision_model.config.hidden_size
        else:
            self.vision_model = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=30, stride=20, padding=1,
                                                bias=False)
            vision_dim = 625

        self.loss_func = torch.nn.CrossEntropyLoss()
        text_dim = self.lang_model.config.hidden_size
        if config['goal'] != 'single_fig':
            commu_dim = text_dim + vision_dim
        else:
            commu_dim = 2*text_dim + vision_dim

        text_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=text_dim,
                                                              nhead=12,
                                                              batch_first=True)
        if config['vision_model_name'] != 'cnn':
            vision_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=vision_dim,
                                                                    nhead=12,
                                                                    batch_first=True)

            commu_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=commu_dim,
                                                                   nhead=12,
                                                                   batch_first=True)
        else:
            vision_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=vision_dim,
                                                                    nhead=5,
                                                                    batch_first=True)

            commu_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=commu_dim,
                                                                   nhead=7,
                                                                   batch_first=True)

        self.text_encoder = torch.nn.TransformerEncoder(text_encoder_layer,
                                                        num_layers=6)
        self.vision_encoder = torch.nn.TransformerEncoder(vision_encoder_layer,
                                                          num_layers=6)
        self.commu_encoder = torch.nn.TransformerEncoder(commu_encoder_layer,
                                                         num_layers=6)
        if config['use_seq_background']:
            self.use_seq_background = True
            self.background_vision_model = AutoModel.from_pretrained(config['vision_model_name'])
            linear_dim = 2 * commu_dim + vision_dim
        else:
            self.use_seq_background = False
            linear_dim = 2 * commu_dim

        if config['goal'] == 'merge_fig':
            linear_dim = 2 * text_dim + vision_dim

        self.linear = torch.nn.Sequential(torch.nn.Linear(linear_dim, 2*linear_dim, bias=True),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(2*linear_dim, 4*linear_dim, bias=True),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(4*linear_dim, 2*linear_dim, bias=True),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(2*linear_dim, linear_dim, bias=True),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(linear_dim, 2, bias=True)
                                          )
        self.activate = nn.Softmax()
        self.vision_layer_norm = nn.LayerNorm(vision_dim)
        self.text_layer_norm = nn.LayerNorm(text_dim)
        self.layer_norm = nn.LayerNorm(commu_dim)

    def _get_vision_embedding(self, inputs):
        if self.config['goal'] == 'benchmark':
            imgs = inputs['benchmark_fig'].squeeze(0)
        else:
            if self.use_seq_background:
                imgs = inputs['with_sep_fig']
            else:
                imgs = inputs['no_sep_fig']

        if self.config['vision_model_name'] != 'cnn':
            imgs.to(self.vision_model.device)
            vision_emdedding = self.vision_model(imgs)['last_hidden_state']
            vision_emdedding = torch.mean(vision_emdedding, dim=1)
            vision_emdedding = vision_emdedding.to(self.lang_model.device)
        else:
            vision_emdedding = self.vision_model(imgs).squeeze(1)
            vision_emdedding = vision_emdedding.view(vision_emdedding.shape[0], -1)
            vision_emdedding = self.vision_encoder(vision_emdedding)
            vision_emdedding = self.vision_layer_norm(vision_emdedding)

        return vision_emdedding

    def _commu_endoer_loop(self, all_embedding):
        result = []
        all_embedding_grouped = all_embedding.view(int(all_embedding.size(0)/2), 2, -1)
        for batch_index in range(int(all_embedding_grouped.size(0))):
            embedding = all_embedding_grouped[batch_index]
            result.append(self.commu_encoder(embedding))

        return torch.stack(result)

    def _reshape_input(self, inputs):
        for key, value in inputs.items():
            if 'fig' not in key or 'gt' in key:
                inputs[key] = value.view(-1, value.size(-1))
            else:
                inputs[key] = value.view(value.size(0)*value.size(1), value.size(2), value.size(3), value.size(4))

        return inputs

    def _get_text_embedding(self, inputs):
        text_emdedding = self.lang_model(input_ids=inputs['input_ids'].squeeze(0),
                                         attention_mask=inputs['attention_mask'].squeeze(0))['last_hidden_state']
        text_embedding_after_encoder = self.text_encoder(text_emdedding)
        text_embedding_after_encoder = torch.mean(text_embedding_after_encoder, dim=1)
        text_embedding_after_encoder = self.text_layer_norm(text_embedding_after_encoder)
        return text_embedding_after_encoder

