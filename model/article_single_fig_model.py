from openpyxl.styles.builtins import output

from model.reading_order_model import ReadingOrderModel
import torch
from torch.nn.functional import normalize

class ArticleSingleFigModel(ReadingOrderModel):
    def __init__(self, config):
        super().__init__(config)

    def gt_generate(self, original_gt):
        #TODO： re-organize and flat the gt
        result = []
        gt_matrix = original_gt.squeeze(0)
        for index_1 in range(len(gt_matrix) - 1):
            for index_2 in range(index_1 + 1, len(gt_matrix)):
                result.append(gt_matrix[index_1, index_2])

        result = torch.stack(result)
        return result

    def linear_input_generate(self, all_embedding):
        result = []
        for index_1 in range(len(all_embedding) - 1):
            for index_2 in range(index_1 + 1, len(all_embedding)):
                result.append(torch.cat((all_embedding[index_1], all_embedding[index_2]), dim=0).unsqueeze(0))

        return torch.cat(result, dim=0)

    def decode(self, output_linear):
        result = []
        max_index = torch.argmax(output_linear, dim=1).detach().cpu().numpy()
        

    def forward(self, input):
        text_emdedding = self.lang_model(input_ids=input['input_ids'].squeeze(0),
                                         attention_mask=input['attention_mask'].squeeze(0))['last_hidden_state']
        text_emdedding = torch.mean(text_emdedding, dim=1)
        vision_emdedding = self.vision_model(input['benchmark_fig'].squeeze(0).to(self.vision_model.device))[
            'last_hidden_state']
        vision_emdedding = torch.mean(vision_emdedding, dim=1)
        vision_emdedding = vision_emdedding.to(self.lang_model.device)
        text_embedding_after_encoder = self.text_encoder(text_emdedding)
        all_embedding = torch.cat((text_embedding_after_encoder, vision_emdedding), dim=1)
        all_embedding = normalize(all_embedding)
        all_embedding = self.commu_encoder(all_embedding)
        input_to_linear = self.linear_input_generate(all_embedding)
        input_to_linear = normalize(input_to_linear)
        output_linear = self.linear(input_to_linear)
        gt = self.gt_generate(input['article_matirx'])
        loss = self.loss_function(output_linear, gt)
        article_result = self.decode(output_linear)



