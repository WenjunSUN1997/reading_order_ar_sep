from torch.utils.data import DataLoader
from transformers import Trainer

class ArticleTrainer(Trainer):
    def __init__(self, train_dataloader=None, test_dataloader=None):
        super(ArticleTrainer, self).__init__()
        self.dataloader = train_dataloader
        self.test_dataloader = test_dataloader

    def get_train_dataloader(self) -> DataLoader:
        pass