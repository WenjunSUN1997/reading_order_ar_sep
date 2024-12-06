from transformers import AutoModel, BertConfig
import torch
from model.model_prototype import ModlePrototype
from utils.focal_loss import FocalLoss

class ArQFormer(ModlePrototype):
    def __init__(self, config=None):
        super(ArQFormer, self).__init__(config)
        q_former_config = BertConfig.from_pretrained("bert-base-uncased")
        q_former_config.add_cross_attention=True
        q_former_config.query_length = config['query_number']
        q_former_config.is_decoder = True
        self.query_number = config['query_number']
        self.q_former = AutoModel.from_pretrained("bert-base-uncased", config=q_former_config)
        self.text_query = torch.nn.Parameter(torch.zeros(1, config['query_number'], self.q_former.config.hidden_size))
        self.vision_query = torch.nn.Parameter(torch.zeros(1, config['query_number'], self.q_former.config.hidden_size))
        self.classification_linear = torch.nn.Linear(self.q_former.config.hidden_size, 2)
        self.loss_func = FocalLoss(gamma=2, alpha=0.25, task_type='binary')

    def forward(self, inputs):
        inputs = self._reshape_input(inputs)
        text_embedding = self.lang_model(input_ids=inputs['input_ids'],
                                         attention_mask=inputs['attention_mask'],
                                         return_dict=True)['last_hidden_state']
        text_embedding = text_embedding.view(int(text_embedding.size(0)/2), int(text_embedding.size(1)*2), -1)
        if self.use_seq_background:
            imgs = inputs['with_sep_fig']
        else:
            imgs = inputs['no_sep_fig']

        vision_embedding = self.vision_model(imgs)['last_hidden_state']
        cross_embedding = torch.cat([text_embedding, vision_embedding], dim=1)
        text_query_embedding = torch.stack([self.text_query] * vision_embedding.size(0)).squeeze(1)
        vision_query_embedding = torch.stack([self.vision_query] * vision_embedding.size(0)).squeeze(1)
        all_query_embedding = torch.cat((text_query_embedding, vision_query_embedding), dim=1)
        q_former_mask = torch.ones([int(all_query_embedding.size(0)), int(all_query_embedding.size(1))]).to(self.vision_query.device)
        output_q_former = self.q_former(inputs_embeds=all_query_embedding,
                                        encoder_hidden_states=cross_embedding,
                                        attention_mask=q_former_mask)
        text_result = torch.mean(output_q_former['last_hidden_state'][:, :self.query_number, :], dim=1)
        vision_result = torch.mean(output_q_former['last_hidden_state'][:, self.query_number:, :], dim=1)
        overall_embedding = output_q_former['pooler_output']
        text_output_linear = self.activate(self.classification_linear(text_result))
        vision_output_linear = self.activate(self.classification_linear(vision_result))
        overall_output_linear = self.activate(self.classification_linear(overall_embedding))
        text_loss = self.loss_func(text_output_linear, inputs['gt'].squeeze(0))
        vision_loss = self.loss_func(vision_output_linear, inputs['gt'].squeeze(0))
        overall_loss = self.loss_func(overall_output_linear, inputs['gt'].squeeze(0))
        loss = text_loss + vision_loss + overall_loss
        return {'loss': loss, 'output': overall_output_linear}
