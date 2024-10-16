from transformers import AutoConfig
from transformers.models.bert.modeling_bert import BertEncoder
import torch
import numpy as np
import torch as th
import torch.nn as nn
from .nn import (
    SiLU,
    linear,
    timestep_embedding,
)
import pdb


class TransformerNetModel2(nn.Module):
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        config=None,
        config_name='bert-base-uncased',
        training_mode='emb',
        vocab_size=None,
        experiment_mode='lm',
        init_pretrained=False,
        logits_mode=1,
        num_grid=32401,
        num_week=8,
        num_hour=25,
        hidden_size=128,
        layer_num=12,
):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if config is None: # transformer config
            config = AutoConfig.from_pretrained(config_name)
            config.hidden_dropout_prob = dropout
            config.hidden_size = hidden_size
            config.layer_num = layer_num
            # config.max_position_embeddings = 256

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample
        self.logits_mode = logits_mode

        self.num_grid = num_grid
        self.num_week = num_week
        self.num_hour = num_hour
        self.hidden_size = hidden_size
        
        if training_mode == 'e2e':
            self.word_embedding = nn.Embedding(num_grid, self.in_channels)
            if self.logits_mode == 2:
                self.lm_head = nn.Linear(self.in_channels, num_grid, bias=True)
            else:
                self.lm_head = nn.Linear(self.in_channels, num_grid)
            with th.no_grad():
                self.lm_head.weight = self.word_embedding.weight

        elif training_mode == 'e2e-simple':
            self.word_embedding = nn.Embedding(vocab_size, self.in_channels)
            self.lm_head = nn.Linear(self.in_channels, vocab_size)
            with th.no_grad():
                self.lm_head.weight = self.word_embedding.weight

        if experiment_mode == 'conditional_gen':
            self.conditional_gen = True
            self.encoder_emb = nn.Embedding(num_grid, config.hidden_size)
            self.encoder = BertEncoder(config)
            config.is_decoder = True
            config.add_cross_attention = True
        elif experiment_mode == 'lm':
            self.conditional_gen = False


        time_embed_dim = model_channels * 4 # 
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, config.hidden_size),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_up_proj = nn.Sequential(nn.Linear(in_channels, config.hidden_size),
                                              nn.Tanh(), nn.Linear(config.hidden_size, config.hidden_size))
        self.input_transformers = BertEncoder(config)

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        # self.register_buffer("hour_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))


        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.week_embedding = nn.Embedding(num_week, config.hidden_size)
        self.hour_embedding = nn.Embedding(num_hour, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_down_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                              nn.Tanh(), nn.Linear(config.hidden_size, out_channels))



    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids)

    def get_logits(self, hidden_repr):
        if self.logits_mode == 1:
            return self.lm_head(hidden_repr)

        elif self.logits_mode == 2:
            text_emb = hidden_repr
            emb_norm = (self.lm_head.weight ** 2).sum(-1).view(-1, 1)  # vocab
            text_emb_t = th.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz*seqlen
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * th.mm(self.lm_head.weight,
                                                                     text_emb_t)  # (vocab, d) x (d, bsz*seqlen)
            scores = th.sqrt(th.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), hidden_repr.size(0),
                                                               hidden_repr.size(1)) # vocab, bsz*seqlen
            scores = -scores.permute(1, 2, 0).contiguous()
            return scores
        else:
            raise NotImplementedError


    def forward(self, x, timesteps, y=None, src_ids=None, src_mask=None, input_hour=None, input_week=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # input_hour: torch.Size([64, 169])
        # x: torch.Size([64, 169, 16])

        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        

        if self.conditional_gen:
            assert src_ids is not None
            src_emb = self.encoder_emb(src_ids)
            encoder_hidden_states = self.encoder(src_emb).last_hidden_state
            encoder_attention_mask = src_mask.unsqueeze(1).unsqueeze(1)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        emb_x = self.input_up_proj(x)
        seq_length = x.size(1)
        position_ids = self.position_ids[:,:seq_length]

        if input_hour is not None and input_week is not None:
            week_emb = self.week_embedding(input_week)
            hour_emb = self.hour_embedding(input_hour)
            # emb_x = emb_x * self.hidden_size ** 0.5
            emb_x = emb_x + week_emb + hour_emb
        else: 
            # week 从1开始
            # x: torch.Size([32, 169, 96])
            hour_ids = th.clone(position_ids)
            hour_ids[0][1:] = (hour_ids[0][1:]-1) % 24+1
            week_ids = th.clone(position_ids)
            for i in range(7):
                week_ids[0][i*24+1:(i+1)*24+1] = i+1
            hour_emb = self.hour_embedding(hour_ids)
            week_emb = self.week_embedding(week_ids)
            emb_x = emb_x + week_emb + hour_emb

        emb_inputs = self.position_embeddings(position_ids) + emb_x + emb.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))

        if self.conditional_gen:
            # if条件生成，取condition emb作为query
            input_trans_hidden_states = self.input_transformers(emb_inputs,
                                                                encoder_hidden_states=encoder_hidden_states, # 把q换成condition emb
                                                                encoder_attention_mask=encoder_attention_mask,
                                                                ).last_hidden_state
        else:
            # 无条件生成，取last_hidden_state
            input_trans_hidden_states = self.input_transformers(emb_inputs).last_hidden_state
        
        h = self.output_down_proj(input_trans_hidden_states)
        h = h.type(x.dtype)
        return h

    def get_feature_vectors(self, x, timesteps, y=None):
        """
        Apply the model and return all of the intermediate tensors.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: a dict with the following keys:
                 - 'down': a list of hidden state tensors from downsampling.
                 - 'middle': the tensor of the output of the lowest-resolution
                             block in the model.
                 - 'up': a list of hidden state tensors from upsampling.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result["down"].append(h.type(x.dtype))
        h = self.middle_block(h, emb)
        result["middle"] = h.type(x.dtype)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=-1)
            h = module(cat_in, emb)
            result["up"].append(h.type(x.dtype))
        return result