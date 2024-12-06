from model.article_single_fig_model import ArticleSingleFigModel
from model.article_merge_model import ArticleMergeModel
from model.ar_q_former import ArQFormer

def model_factory(config):
    if config['goal'] == 'single_fig':
        return ArticleSingleFigModel(config)

    if config['goal'] == 'merge_fig':
        return ArticleMergeModel(config)

    if config['goal'] == 'qformer':
        return ArQFormer(config)
