from model.article_single_fig_model import ArticleSingleFigModel
from model.article_merge_model import ArticleMergeModel

def model_factory(config):
    return ArticleSingleFigModel(config)

