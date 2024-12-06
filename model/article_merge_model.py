from model.model_prototype import ModlePrototype
import torch

class ArticleMergeModel(ModlePrototype):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, inputs):
        inputs = self._reshape_input(inputs)
        text_embedding_after_encoder = self._get_text_embedding(inputs)
        vision_emdedding = self._get_vision_embedding(inputs)
        text_embedding = text_embedding_after_encoder.view(int(text_embedding_after_encoder.shape[0]/2), -1)
        input_to_linear = torch.cat((text_embedding, vision_emdedding), dim=1)
        output_linear = self.linear(input_to_linear)
        output_linear = self.activate(output_linear)
        loss = self.loss_func(output_linear, inputs['gt'].squeeze(0))
        return {'loss': loss, 'output': output_linear}
