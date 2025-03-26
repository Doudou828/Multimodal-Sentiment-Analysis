import torch
from torch import nn
from .basic_layers import Transformer, CrossTransformer, HhyperLearningEncoder, GradientReversalLayer
from .bert import BertTextEncoder
from einops import rearrange, repeat


class ChannelAttention(nn.Module):
    def __init__(self, in_dim, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim // ratio, bias=False),#dimensionality reduction
            nn.ReLU(),
            nn.Linear(in_dim // ratio, in_dim, bias=False),
            nn.Sigmoid()  # Make sure the output is between 0 and 1
        )

    def forward(self, x):
        b, n, d = x.shape  # input shape: (batch, tokens, dim)
        x_permuted = x.permute(0, 2, 1)  #  (batch, dim, tokens)

        # Global Avg Pooling
        avg_pool = self.avg_pool(x_permuted)  # (b, d, 1)
        avg_out = self.fc(avg_pool.squeeze(-1))  # (b, d)

        # Global Max Pooling
        max_pool = self.max_pool(x_permuted)  # (b, d, 1)
        max_out = self.fc(max_pool.squeeze(-1))  # (b, d)

        # Combined attention weight
        channel_weights = avg_out + max_out  # (b, d)
        channel_weights = channel_weights.unsqueeze(-1)  # (b, d, 1)

        # Apply attention and restore the original dimension, 
        # multiplying weights with original input features 
        scaled_x = x_permuted * channel_weights  # (b, d, tokens)
        scaled_x = scaled_x.permute(0, 2, 1)  # (b, tokens, dim)

        return scaled_x



class LNLN(nn.Module):
    def __init__(self, args):
        super(LNLN, self).__init__()

        self.h_hyper = nn.Parameter(torch.ones(1, args['model']['feature_extractor']['token_length'][0], 128))
        self.h_p = nn.Parameter(torch.ones(1, args['model']['feature_extractor']['token_length'][0], 128))

        self.bertmodel = BertTextEncoder(use_finetune=True, transformers='bert', pretrained=args['model']['feature_extractor']['bert_pretrained'])

        # Add CBAM-like Attention
        self.channel_attention_l = ChannelAttention(args['model']['feature_extractor']['hidden_dims'][0])
        self.channel_attention_a = ChannelAttention(args['model']['feature_extractor']['hidden_dims'][2])
        self.channel_attention_v = ChannelAttention(args['model']['feature_extractor']['hidden_dims'][1])

        self.proj_l = nn.Sequential(
            nn.Linear(args['model']['feature_extractor']['input_dims'][0], args['model']['feature_extractor']['hidden_dims'][0]),
            Transformer(num_frames=args['model']['feature_extractor']['input_length'][0],
                        save_hidden=False,
                        token_len=args['model']['feature_extractor']['token_length'][0],
                        dim=args['model']['feature_extractor']['hidden_dims'][0],
                        depth=args['model']['feature_extractor']['depth'],
                        heads=args['model']['feature_extractor']['heads'],
                        mlp_dim=args['model']['feature_extractor']['hidden_dims'][0])
        )

        self.proj_a = nn.Sequential(
            nn.Linear(args['model']['feature_extractor']['input_dims'][2], args['model']['feature_extractor']['hidden_dims'][2]),
            Transformer(num_frames=args['model']['feature_extractor']['input_length'][2],
                        save_hidden=False,
                        token_len=args['model']['feature_extractor']['token_length'][2],
                        dim=args['model']['feature_extractor']['hidden_dims'][2],
                        depth=args['model']['feature_extractor']['depth'],
                        heads=args['model']['feature_extractor']['heads'],
                        mlp_dim=args['model']['feature_extractor']['hidden_dims'][2])
        )

        self.proj_v = nn.Sequential(
            nn.Linear(args['model']['feature_extractor']['input_dims'][1], args['model']['feature_extractor']['hidden_dims'][1]),
            Transformer(num_frames=args['model']['feature_extractor']['input_length'][1],
                        save_hidden=False,
                        token_len=args['model']['feature_extractor']['token_length'][1],
                        dim=args['model']['feature_extractor']['hidden_dims'][1],
                        depth=args['model']['feature_extractor']['depth'],
                        heads=args['model']['feature_extractor']['heads'],
                        mlp_dim=args['model']['feature_extractor']['hidden_dims'][1])
        )


        self.proxy_dominate_modality_generator = Transformer(
            num_frames=args['model']['dmc']['proxy_dominant_feature_generator']['input_length'],
            save_hidden=False,
            token_len=args['model']['dmc']['proxy_dominant_feature_generator']['token_length'],
            dim=args['model']['dmc']['proxy_dominant_feature_generator']['input_dim'],
            depth=args['model']['dmc']['proxy_dominant_feature_generator']['depth'],
            heads=args['model']['dmc']['proxy_dominant_feature_generator']['heads'],
            mlp_dim=args['model']['dmc']['proxy_dominant_feature_generator']['hidden_dim'])

        self.GRL = GradientReversalLayer(alpha=1.0)

        self.effective_discriminator = nn.Sequential(
            nn.Linear(args['model']['dmc']['effectiveness_discriminator']['input_dim'],
                      args['model']['dmc']['effectiveness_discriminator']['hidden_dim']),
            nn.LeakyReLU(0.1),
            nn.Linear(args['model']['dmc']['effectiveness_discriminator']['hidden_dim'],
                      args['model']['dmc']['effectiveness_discriminator']['out_dim']),
        )

        self.completeness_check = nn.ModuleList([
            Transformer(num_frames=args['model']['dmc']['completeness_check']['input_length'],
                        save_hidden=False,
                        token_len=args['model']['dmc']['completeness_check']['token_length'],
                        dim=args['model']['dmc']['completeness_check']['input_dim'],
                        depth=args['model']['dmc']['completeness_check']['depth'],
                        heads=args['model']['dmc']['completeness_check']['heads'],
                        mlp_dim=args['model']['dmc']['completeness_check']['hidden_dim']),

            nn.Sequential(
                nn.Linear(args['model']['dmc']['completeness_check']['hidden_dim'], int(args['model']['dmc']['completeness_check']['hidden_dim']/2)),
                nn.LeakyReLU(0.1),
                nn.Linear(int(args['model']['dmc']['completeness_check']['hidden_dim']/2), 1),
                nn.Sigmoid()),
        ])


        self.reconstructor = nn.ModuleList([
            Transformer(num_frames=args['model']['reconstructor']['input_length'],
                        save_hidden=False,
                        token_len=None,
                        dim=args['model']['reconstructor']['input_dim'],
                        depth=args['model']['reconstructor']['depth'],
                        heads=args['model']['reconstructor']['heads'],
                        mlp_dim=args['model']['reconstructor']['hidden_dim']) for _ in range(3)
        ])


        self.dmml = nn.ModuleList([
            Transformer(num_frames=args['model']['dmml']['language_encoder']['input_length'],
                        save_hidden=True,
                        token_len=None,
                        dim=args['model']['dmml']['language_encoder']['input_dim'],
                        depth=args['model']['dmml']['language_encoder']['depth'],
                        heads=args['model']['dmml']['language_encoder']['heads'],
                        mlp_dim=args['model']['dmml']['language_encoder']['hidden_dim']),

            HhyperLearningEncoder(dim=args['model']['dmml']['hyper_modality_learning']['input_dim'],
                                  dim_head=int(args['model']['dmml']['hyper_modality_learning']['input_dim']/args['model']['dmml']['hyper_modality_learning']['heads']),
                                  depth=args['model']['dmml']['hyper_modality_learning']['depth'],
                                  heads=args['model']['dmml']['hyper_modality_learning']['heads']),

            CrossTransformer(source_num_frames=args['model']['dmml']['fuison_transformer']['source_length'],
                             tgt_num_frames=args['model']['dmml']['fuison_transformer']['tgt_length'],
                             dim=args['model']['dmml']['fuison_transformer']['input_dim'],
                             depth=args['model']['dmml']['fuison_transformer']['depth'],
                             heads=args['model']['dmml']['fuison_transformer']['heads'],
                             mlp_dim=args['model']['dmml']['fuison_transformer']['hidden_dim']),

            nn.Linear(args['model']['dmml']['regression']['input_dim'], args['model']['dmml']['regression']['out_dim'])
        ])



    def forward(self, complete_input, incomplete_input):
        vision, audio, language = complete_input
        vision_m, audio_m, language_m = incomplete_input

        b = vision_m.size(0)

        h_1_v = self.proj_v(vision_m)[:, :8]
        h_1_a = self.proj_a(audio_m)[:, :8]
        h_1_l = self.proj_l(self.bertmodel(language_m))[:, :8]

        # 加入 CBAM 类似的通道注意力
        h_1_v = self.channel_attention_v(h_1_v)
        h_1_a = self.channel_attention_a(h_1_a)
        h_1_l = self.channel_attention_l(h_1_l)

        feat_tmp = self.completeness_check[0](h_1_l)[:, :1].squeeze()
        w = self.completeness_check[1](feat_tmp) # completeness scores

        h_0_p = repeat(self.h_p, '1 n d -> b n d', b = b)
        h_1_p = self.proxy_dominate_modality_generator(torch.cat([h_0_p, h_1_a, h_1_v], dim=1))[:, :8]
        h_1_p = self.GRL(h_1_p)
        h_1_d = h_1_p * (1-w.unsqueeze(-1)) + h_1_l * w.unsqueeze(-1)

        h_hyper = repeat(self.h_hyper, '1 n d -> b n d', b = b)
        h_d_list = self.dmml[0](h_1_d)
        h_hyper = self.dmml[1](h_d_list, h_1_a, h_1_v, h_hyper)
        feat = self.dmml[2](h_hyper, h_d_list[-1])
        # Two ways to get the cls_output: using extra cls_token or using mean of all the features
        # output = self.dmml[3](feat[:, 0]) 
        output = self.dmml[3](torch.mean(feat[:, 1:], dim=1))

        rec_feats, complete_feats, effectiveness_discriminator_out = None, None, None
        if (vision is not None) and (audio is not None) and (language is not None):
            # Reconstruction
            # for layer in self.reconstructor:
            rec_feat_a = self.reconstructor[0](h_1_a)
            rec_feat_v = self.reconstructor[1](h_1_v)
            rec_feat_l = self.reconstructor[2](h_1_l)
            rec_feats = torch.cat([rec_feat_a, rec_feat_v, rec_feat_l], dim=1)

            # Compute the complete features as the label of reconstruction
            complete_language_feat = self.proj_l(self.bertmodel(language))[:, :8]
            complete_vision_feat = self.proj_v(vision)[:, :8]
            complete_audio_feat = self.proj_a(audio)[:, :8]

            effective_discriminator_input = rearrange(torch.cat([h_1_d, complete_language_feat]), 'b n d -> (b n) d')
            effectiveness_discriminator_out = self.effective_discriminator(effective_discriminator_input)
        
            complete_feats = torch.cat([complete_audio_feat, complete_vision_feat, complete_language_feat], dim=1) # as the label of reconstruction
  

        return {'sentiment_preds': output, 
                'w': w, 
                'effectiveness_discriminator_out': effectiveness_discriminator_out, 
                'rec_feats': rec_feats, 
                'complete_feats': complete_feats}



def build_model(args):
    return LNLN(args)