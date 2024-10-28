from utils.evaluator import Evaluator
from model.reading_order_model import ReadingOrderModel
import torch
from torch.nn.functional import normalize

class ArticleSingleFigModel(ReadingOrderModel):
    def __init__(self, config):
        super().__init__(config)
        self.evaluator = Evaluator()

    def gt_generate(self, original_gt):
        length_record = []
        result = []
        gt_matrix = original_gt.squeeze(0)
        for index_1 in range(len(gt_matrix) - 1):
            length_item = 0
            for index_2 in range(index_1 + 1, len(gt_matrix)):
                result.append(gt_matrix[index_1, index_2])
                length_item += 1

            length_record.append(length_item)

        result = torch.stack(result)
        return result, length_record

    def linear_input_generate(self, all_embedding):
        result = []
        for index_1 in range(len(all_embedding) - 1):
            for index_2 in range(index_1 + 1, len(all_embedding)):
                result.append(torch.cat((all_embedding[index_1], all_embedding[index_2]), dim=0).unsqueeze(0))

        return torch.cat(result, dim=0)

    def article_decode(self, max_index, length_record):
        def check_in(index, result):
            for result_item in result:
                if index in result_item:
                    return True

            return False

        result = []
        for index, length in enumerate(length_record):
            if check_in(index, result):
                continue

            result_item = [index]
            if index == 0:
                prediction = max_index[:length]
            else:
                prediction = max_index[length_record[index-1]: length_record[index-1]+length]

            for prediction_index, prediction_item in enumerate(prediction):
                if prediction_item == 0 or check_in(prediction_index+index+1, result):
                    continue
                else:
                    result_item.append(prediction_index+index+1)

            result.append(result_item)

        return result

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
        gt, length_record = self.gt_generate(input['article_matirx'])
        loss = self.loss_func(output_linear, gt)
        max_index = torch.argmax(output_linear, dim=1).detach().cpu().numpy()
        article_result = self.article_decode(max_index, length_record)
        article_gt =  self.article_decode(gt.detach().cpu().numpy(), length_record)
        performance = self.evaluator.evaluate(article_result, article_gt)
        return {'loss': loss, 'performance': performance}




