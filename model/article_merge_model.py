from model.reading_order_model import ReadingOrderModel

class ArticleMergeModel(ReadingOrderModel):
    def __init__(self, config):
        super().__init__(config)

    #TODO: the function to merge the figs
    def merge_image(self, input):
        pass

    #TODO: the forward function will merge the fig of 2 inputs then combine their text semantic vector and vision vector
    def forward(self, input):
        pass
