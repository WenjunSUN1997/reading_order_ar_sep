from utils.evaluator import Evaluator
import torch
from utils.focal_loss import FocalLoss
from model.model_prototype import ModlePrototype

class ArticleSingleFigModel(ModlePrototype):
    def __init__(self, config):
        super(ArticleSingleFigModel, self).__init__(config)
        # self.loss_func = FocalLoss(gamma=2, alpha=0.25, task_type='binary')

    def linear_input_generate(self, all_embedding):
        result = []
        for index_1 in range(len(all_embedding) - 1):
            for index_2 in range(index_1 + 1, len(all_embedding)):
                result.append(torch.cat((all_embedding[index_1], all_embedding[index_2]), dim=0).unsqueeze(0))

        return torch.cat(result, dim=0)

    def forward(self, inputs):
        inputs = self._reshape_input(inputs)
        text_embedding_after_encoder = self._get_text_embedding(inputs)
        vision_emdedding = self._get_vision_embedding(inputs)
        # text_embedding_after_encoder = self.text_encoder(text_emdedding)
        all_embedding = torch.cat((text_embedding_after_encoder, vision_emdedding), dim=1)
        all_embedding = self.layer_norm(all_embedding)
        all_embedding = self._commu_endoer_loop(all_embedding)
        # input_to_linear = self.linear_input_generate(all_embedding)
        # input_to_linear = normalize(input_to_linear)
        input_to_linear = all_embedding.view(int(all_embedding.size(0)) , -1)
        output_linear = self.linear(input_to_linear)
        output_linear = self.activate(output_linear)
        loss = self.loss_func(output_linear, inputs['gt'].squeeze(0))
        return {'loss': loss, 'output': output_linear}
        # gt, length_record = self.gt_generate(inputs['article_matirx'])

        # max_index = torch.argmax(output_linear, dim=1).detach().cpu().numpy()
        # article_result = self.article_decode(max_index, length_record)
        # article_gt =  self.article_decode(gt.detach().cpu().numpy(), length_record)
        # performance = self.evaluator.evaluate_single_page(article_result, article_gt)





