from torch import nn
from torch.nn import functional as F


class MultimodalLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.alpha = args['base']['alpha']
        self.beta = args['base']['beta']
        self.gamma = args['base']['gamma']
        self.sigma = args['base']['sigma']
        self.CE_Fn = nn.CrossEntropyLoss() #Cross entropy loss, used for classification
        self.MSE_Fn = nn.MSELoss() #Mean square error loss MSE, used for regression


    def forward(self, out, label):

        l_cc = self.MSE_Fn(out['w'], label['completeness_labels']) if out['w'] is not None else 0 #MSE 计算预测完整度和真实完整度之间的误差,让模型更好地感知数据的完整性

        l_adv = self.CE_Fn(out['effectiveness_discriminator_out'], label['effectiveness_labels']) if out['effectiveness_discriminator_out'] is not None else 0 #交叉熵（CrossEntropy）计算分类误差,训练判别器，使其学会识别真实有效的模态特征

        l_rec = self.MSE_Fn(out['rec_feats'], out['complete_feats']) if out['rec_feats'] is not None and out['complete_feats'] is not None else 0

        l_sp = self.MSE_Fn(out['sentiment_preds'], label['sentiment_labels']) #MSE 计算重构误差,训练模型重建丢失的模态信息
        
        loss = self.alpha * l_cc + self.beta * l_adv + self.gamma * l_rec + self.sigma * l_sp #监督最终的情感预测任务

        return {'loss': loss, 'l_sp': l_sp, 'l_cc': l_cc, \
                'l_adv': l_adv, 'l_rec': l_rec}

