import numpy as np
import torch
import torch.nn as nn
from modules.swintrans import SwinTransformer as STBackbone
from modules.base_cmn import BaseCMN
from modules.visual_extractor import VisualExtractor


class BaseCMNModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(BaseCMNModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.swintrans = STBackbone(
            img_size=384,
            embed_dim=192,
            depths=[2, 2, 18, 2],
            num_heads=[6, 12, 24, 48],
            window_size=12,
            num_classes=1000
        )
        # 得要预训练权重
        print('load pretrained weights!')
        self.swintrans.load_weights(
            './swin_tiny_patch4_window7_224.pth'
        )
        # Freeze parameters 这里要冻结参数，我们这里是迁移学习
        for _name, _weight in self.swintrans.named_parameters():
            _weight.requires_grad = False
        # 这里定义编码解码结构
        self.encoder_decoder = BaseCMN(args, tokenizer)
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        elif args.dataset_name == 'LGK':
            self.forward = self.forward_LGK
        elif args.dataset_name == 'mimic_cxr':
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, mode='train', update_opts={}):
#         att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
#         att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
#         fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
# #         print(att_feats_0.shape)
#         att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        images_0 = torch.squeeze(images[:, 0])
        images_1 = torch.squeeze(images[:, 1])
        att_f_0 = self.swintrans(images_0)  # 它的输入就是一个图像特征
        att_f_1 = self.swintrans(images_1)
        att_feats = torch.cat((att_f_0, att_f_1), dim=1)
        if mode == 'train':
            output = self.encoder_decoder(att_feats, targets, mode='forward')
            return output
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(att_feats, mode='sample', update_opts=update_opts)
            return output, output_probs
        else:
            raise ValueError
    def forward_mimic_cxr(self, images, targets=None, mode='train', update_opts={}):
        att_feats = self.swintrans(images)
        if mode == 'train':
            output = self.encoder_decoder(att_feats, targets, mode='forward')
            return output
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(att_feats, mode='sample', update_opts=update_opts)
            return output, output_probs
        else:
            raise ValueError
    def forward_LGK(self, images, targets=None, mode='train', update_opts={}):
        att_feats, _ = self.visual_extractor(images)
        if mode == 'train':
            output = self.encoder_decoder(att_feats, targets, mode='forward')
            return output
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(att_feats, mode='sample', update_opts=update_opts)
            return output, output_probs
        else:
            raise ValueError
