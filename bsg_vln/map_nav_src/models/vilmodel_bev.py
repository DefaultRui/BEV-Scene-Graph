import json
import logging
import math
import os
import sys
from io import open
from typing import Callable, List, Tuple
import numpy as np
import copy

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor, device, dtype

from transformers import BertPreTrainedModel

from .ops import create_transformer_encoder
from .ops import extend_neg_masks, gen_seq_masks, pad_tensors_wgrad
from .ops import findindex

from utils.ops import pad_tensors, gen_seq_masks
import h5py
import json

from r2r.parser import parse_args
args = parse_args()

logger = logging.getLogger(__name__)

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except (ImportError, AttributeError) as e:
    # logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
    BertLayerNorm = torch.nn.LayerNorm


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}



class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        """
        hidden_states: (N, L_{hidden}, D)
        attention_mask: (N, H, L_{hidden}, L_{hidden})
        """
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # recurrent vlnbert use attention scores
        outputs = (context_layer, attention_scores) if self.output_attentions else (context_layer,)
        return outputs

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs

class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states, attention_mask,
                None if head_mask is None else head_mask[i],
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size,
                                 config.vocab_size,
                                 bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class BertOutAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if ctx_dim is None:
            ctx_dim = config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_scores

class BertXAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        self.att = BertOutAttention(config, ctx_dim=ctx_dim)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        output, attention_scores = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output, attention_scores

class GraphLXRTXLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Lang self-att and FFN layer
        if config.use_lang2visn_attn:
            self.lang_self_att = BertAttention(config)
            self.lang_inter = BertIntermediate(config)
            self.lang_output = BertOutput(config)

        # Visn self-att and FFN layer
        self.visn_self_att = BertAttention(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)

        # The cross attention layer
        self.visual_attention = BertXAttention(config)

    def forward(
        self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask,
        graph_sprels=None
    ):      
        visn_att_output = self.visual_attention(
            visn_feats, lang_feats, ctx_att_mask=lang_attention_mask
        )[0]

        if graph_sprels is not None:
            visn_attention_mask = visn_attention_mask + graph_sprels
        visn_att_output = self.visn_self_att(visn_att_output, visn_attention_mask)[0]

        visn_inter_output = self.visn_inter(visn_att_output)
        visn_output = self.visn_output(visn_inter_output, visn_att_output)

        return visn_output

    def forward_lang2visn(
        self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask,
    ):
        lang_att_output = self.visual_attention(
            lang_feats, visn_feats, ctx_att_mask=visn_attention_mask
        )[0]
        lang_att_output = self.lang_self_att(
            lang_att_output, lang_attention_mask
        )[0]
        lang_inter_output = self.lang_inter(lang_att_output)
        lang_output = self.lang_output(lang_inter_output, lang_att_output)
        return lang_output

class LanguageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_l_layers = config.num_l_layers
        self.update_lang_bert = config.update_lang_bert

        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_l_layers)]
        )
        if not self.update_lang_bert:
            for name, param in self.layer.named_parameters():
                param.requires_grad = False

    def forward(self, txt_embeds, txt_masks):
        extended_txt_masks = extend_neg_masks(txt_masks)
        for layer_module in self.layer:
            temp_output = layer_module(txt_embeds, extended_txt_masks)
            txt_embeds = temp_output[0]
        if not self.update_lang_bert:
            txt_embeds = txt_embeds.detach()
        return txt_embeds
    
class CrossmodalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_x_layers = config.num_x_layers
        self.x_layers = nn.ModuleList(
            [GraphLXRTXLayer(config) for _ in range(self.num_x_layers)]
        )

    def forward(self, txt_embeds, txt_masks, img_embeds, img_masks, graph_sprels=None):
        extended_txt_masks = extend_neg_masks(txt_masks)
        extended_img_masks = extend_neg_masks(img_masks) # (N, 1(H), 1(L_q), L_v)
        for layer_module in self.x_layers:
            img_embeds = layer_module(
                txt_embeds, extended_txt_masks, 
                img_embeds, extended_img_masks,
                graph_sprels=graph_sprels
            )
        return img_embeds

class ImageEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.img_linear = nn.Linear(config.image_feat_size, config.hidden_size)
        self.img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.loc_linear = nn.Linear(config.angle_feat_size + 3, config.hidden_size)
        self.loc_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)

        if config.obj_feat_size > 0 and config.obj_feat_size != config.image_feat_size:
            self.obj_linear = nn.Linear(config.obj_feat_size, config.hidden_size)
            self.obj_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        else:
            self.obj_linear = self.obj_layer_norm = None

        # 0: non-navigable, 1: navigable, 2: object
        self.nav_type_embedding = nn.Embedding(3, config.hidden_size)

        # tf naming convention for layer norm
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if config.num_pano_layers > 0:
            self.pano_encoder = create_transformer_encoder(
                config, config.num_pano_layers, norm=True
            )
        else:
            self.pano_encoder = None

    def forward(
        self, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, type_embed_layer
    ):
        device = traj_view_img_fts.device
        has_obj = traj_obj_img_fts is not None

        traj_view_img_embeds = self.img_layer_norm(self.img_linear(traj_view_img_fts))
        if has_obj:
            if self.obj_linear is None:
                traj_obj_img_embeds = self.img_layer_norm(self.img_linear(traj_obj_img_fts))
            else:
                traj_obj_img_embeds = self.obj_layer_norm(self.obj_linear(traj_obj_img_embeds))
            traj_img_embeds = []
            for view_embed, obj_embed, view_len, obj_len in zip(
                traj_view_img_embeds, traj_obj_img_embeds, traj_vp_view_lens, traj_vp_obj_lens
            ):
                if obj_len > 0:
                    traj_img_embeds.append(torch.cat([view_embed[:view_len], obj_embed[:obj_len]], 0))
                else:
                    traj_img_embeds.append(view_embed[:view_len])
            traj_img_embeds = pad_tensors_wgrad(traj_img_embeds)
            traj_vp_lens = traj_vp_view_lens + traj_vp_obj_lens
        else:
            traj_img_embeds = traj_view_img_embeds
            traj_vp_lens = traj_vp_view_lens

        traj_embeds = traj_img_embeds + \
                      self.loc_layer_norm(self.loc_linear(traj_loc_fts)) + \
                      self.nav_type_embedding(traj_nav_types) + \
                      type_embed_layer(torch.ones(1, 1).long().to(device))
        traj_embeds = self.layer_norm(traj_embeds)
        traj_embeds = self.dropout(traj_embeds)

        traj_masks = gen_seq_masks(traj_vp_lens)
        if self.pano_encoder is not None:
            traj_embeds = self.pano_encoder(
                traj_embeds, src_key_padding_mask=traj_masks.logical_not()
            )

        split_traj_embeds = torch.split(traj_embeds, traj_step_lens, 0)
        split_traj_vp_lens = torch.split(traj_vp_lens, traj_step_lens, 0)
        return split_traj_embeds, split_traj_vp_lens
        
class LocalVPEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vp_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size*2 + 6, config.hidden_size),
            BertLayerNorm(config.hidden_size, eps=1e-12)
        )
        self.encoder = CrossmodalEncoder(config)

    def vp_input_embedding(self, split_traj_embeds, split_traj_vp_lens, vp_pos_fts):
        vp_img_embeds = pad_tensors_wgrad([x[-1] for x in split_traj_embeds])
        vp_lens = torch.stack([x[-1]+1 for x in split_traj_vp_lens], 0)
        vp_masks = gen_seq_masks(vp_lens)
        max_vp_len = max(vp_lens)

        batch_size, _, hidden_size = vp_img_embeds.size()
        device = vp_img_embeds.device
        # add [stop] token at beginning
        vp_img_embeds = torch.cat(
            [torch.zeros(batch_size, 1, hidden_size).to(device), vp_img_embeds], 1
        )[:, :max_vp_len]
        vp_embeds = vp_img_embeds + self.vp_pos_embeddings(vp_pos_fts)

        return vp_embeds, vp_masks

    def forward(
        self, txt_embeds, txt_masks, split_traj_embeds, split_traj_vp_lens, vp_pos_fts
    ):
        vp_embeds, vp_masks = self.vp_input_embedding(
            split_traj_embeds, split_traj_vp_lens, vp_pos_fts
        )
        vp_embeds = self.encoder(txt_embeds, txt_masks, vp_embeds, vp_masks)
        return vp_embeds

class GlobalMapEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gmap_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size + 3, config.hidden_size),
            BertLayerNorm(config.hidden_size, eps=1e-12)
        )
        self.gmap_step_embeddings = nn.Embedding(config.max_action_steps, config.hidden_size)
        self.encoder = CrossmodalEncoder(config)
        
        if config.graph_sprels:
            self.sprel_linear = nn.Linear(1, 1)
        else:
            self.sprel_linear = None

    def _aggregate_gmap_features(
        self, split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids
    ):
        batch_size = len(split_traj_embeds)
        device = split_traj_embeds[0].device

        batch_gmap_img_fts = []
        for i in range(batch_size):
            visited_vp_fts, unvisited_vp_fts = {}, {}
            vp_masks = gen_seq_masks(split_traj_vp_lens[i])
            max_vp_len = max(split_traj_vp_lens[i])
            i_traj_embeds = split_traj_embeds[i][:, :max_vp_len] * vp_masks.unsqueeze(2)
            for t in range(len(split_traj_embeds[i])):
                visited_vp_fts[traj_vpids[i][t]] = torch.sum(i_traj_embeds[t], 0) / split_traj_vp_lens[i][t]
                for j, vp in enumerate(traj_cand_vpids[i][t]):
                    if vp not in visited_vp_fts:
                        unvisited_vp_fts.setdefault(vp, [])
                        unvisited_vp_fts[vp].append(i_traj_embeds[t][j])

            gmap_img_fts = []
            for vp in gmap_vpids[i][1:]:
                if vp in visited_vp_fts:
                    gmap_img_fts.append(visited_vp_fts[vp])
                else:
                    gmap_img_fts.append(torch.mean(torch.stack(unvisited_vp_fts[vp], 0), 0))
            gmap_img_fts = torch.stack(gmap_img_fts, 0)
            batch_gmap_img_fts.append(gmap_img_fts)

        batch_gmap_img_fts = pad_tensors_wgrad(batch_gmap_img_fts)
        # add a [stop] token at beginning
        batch_gmap_img_fts = torch.cat(
            [torch.zeros(batch_size, 1, batch_gmap_img_fts.size(2)).to(device), batch_gmap_img_fts], 
            dim=1
        )
        return batch_gmap_img_fts
    
    def gmap_input_embedding(
        self, split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
        gmap_step_ids, gmap_pos_fts, gmap_lens
    ):
        gmap_img_fts = self._aggregate_gmap_features(
            split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids
        )
        gmap_embeds = gmap_img_fts + \
                      self.gmap_step_embeddings(gmap_step_ids) + \
                      self.gmap_pos_embeddings(gmap_pos_fts)
        gmap_masks = gen_seq_masks(gmap_lens)
        return gmap_embeds, gmap_masks

    def forward(
        self, txt_embeds, txt_masks,
        split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
        gmap_step_ids, gmap_pos_fts, gmap_lens, graph_sprels=None
    ):
        gmap_embeds, gmap_masks = self.gmap_input_embedding(
            split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
            gmap_step_ids, gmap_pos_fts, gmap_lens
        )
        
        if self.sprel_linear is not None:
            graph_sprels = self.sprel_linear(graph_sprels.unsqueeze(3)).squeeze(3).unsqueeze(1)
        else:
            graph_sprels = None

        gmap_embeds = self.encoder(
            txt_embeds, txt_masks, gmap_embeds, gmap_masks,
            graph_sprels=graph_sprels
        )
        return gmap_embeds
       
    
class ClsPrediction(nn.Module):
    def __init__(self, hidden_size, input_size=None):
        super().__init__()
        if input_size is None:
            input_size = hidden_size
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                #  nn.Linear(hidden_size, 2)
                                 nn.Linear(hidden_size, 1)
                                 )

    def forward(self, x):
        return self.net(x)


class MAPSeg(nn.Module):
    def __init__(self, hidden_size, input_size=None):
        super().__init__()
        if input_size is None:
            input_size = hidden_size
        self.net = nn.Sequential(nn.Linear(input_size, 27),)

    def forward(self, x):
        return self.net(x)


class GlocalTextPathNavCMT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.lang_encoder = LanguageEncoder(config)

        self.img_embeddings = ImageEmbeddings(config)
        
        self.local_encoder = LocalVPEncoder(config)
        self.global_encoder = GlobalMapEncoder(config)

        self.bev_encoder = GlobalMapEncoder(config)
        self.bevlocal_encoder = LocalVPEncoder(config)

        self.global_sap_head = ClsPrediction(self.config.hidden_size)
        self.bevglobal_sap_head = ClsPrediction(self.config.hidden_size)
        self.local_sap_head = ClsPrediction(self.config.hidden_size)
        self.bevlocal_sap_head = ClsPrediction(self.config.hidden_size)
        # self.mapseg_head =  MAPSeg(self.config.hidden_size)
        if args.map_grid == 5:
            self.conv_seg = nn.Conv2d(self.config.hidden_size, 27, 3,2,1)
        elif args.map_grid == 7:
            self.conv_seg = nn.Conv2d(self.config.hidden_size, 27, 3,2,5)
        elif args.map_grid == 11:
            self.conv_seg = nn.Conv2d(self.config.hidden_size, 27, 3,1,1)
        elif args.map_grid == 15:
            self.conv_seg = nn.Conv2d(self.config.hidden_size, 27, 3,1,3)
        elif args.map_grid == 21:
            self.conv_seg = nn.Conv2d(self.config.hidden_size, 27, 3,1,6)
        elif args.map_grid == 31:
            self.conv_seg = nn.Sequential(
                nn.Conv2d(self.config.hidden_size, 27, 5,1,12),
            )            
        
        if config.glocal_fuse:
            self.sap_fuse_linear = ClsPrediction(self.config.hidden_size, input_size=self.config.hidden_size*2)
            self.sap_fuse_linear_bev = ClsPrediction(self.config.hidden_size, input_size=self.config.hidden_size*2)
        else:
            self.sap_fuse_linear = None
            self.sap_fuse_linear_bev = None
            # self.sap_fuse_linear_new = None

        if self.config.obj_feat_size > 0:
            self.og_head = ClsPrediction(self.config.hidden_size)
        
        self.target_predictor = nn.Sequential(
            nn.Linear( self.config.hidden_size, self.config.hidden_size // 2 ),
            nn.ReLU(),
            nn.Linear( self.config.hidden_size // 2, 1 )
        )

        self.pc_range=[-args.bev_range, -args.bev_range, -1.0, args.bev_range, args.bev_range, -1.0+args.bev_height]
        self.bev_h = args.bev_grid
        self.bev_w = args.bev_grid
        self.embed_dims = 768
        self._bevfeature_store = {}
        self.candi2bev = {}

        self.init_weights()
        
        if config.fix_lang_embedding or config.fix_local_branch:
            for k, v in self.embeddings.named_parameters():
                v.requires_grad = False
            for k, v in self.lang_encoder.named_parameters():
                v.requires_grad = False
        if config.fix_pano_embedding or config.fix_local_branch:
            for k, v in self.img_embeddings.named_parameters():
                v.requires_grad = False
        if config.fix_local_branch:
            for k, v in self.local_encoder.named_parameters():
                v.requires_grad = False
            for k, v in self.bevlocal_encoder.named_parameters():
                v.requires_grad = False
            for k, v in self.local_sap_head.named_parameters():
                v.requires_grad = False
            for k, v in self.og_head.named_parameters():
                v.requires_grad = False
    
    def forward_text(self, txt_ids, txt_masks):
        txt_token_type_ids = torch.zeros_like(txt_ids)
        txt_embeds = self.embeddings(txt_ids, token_type_ids=txt_token_type_ids)
        txt_embeds = self.lang_encoder(txt_embeds, txt_masks)
        return txt_embeds

    def forward_panorama_per_step(
        self, view_img_fts, obj_img_fts, loc_fts, nav_types, view_lens, obj_lens
    ):
        device = view_img_fts.device
        has_obj = obj_img_fts is not None

        view_img_embeds = self.img_embeddings.img_layer_norm(
            self.img_embeddings.img_linear(view_img_fts)
        )
        if has_obj:
            if self.img_embeddings.obj_linear is None:
                obj_img_embeds = self.img_embeddings.img_layer_norm(
                    self.img_embeddings.img_linear(obj_img_fts)
                )
            else:
                obj_img_embeds = self.img_embeddings.obj_layer_norm(
                    self.img_embeddings.obj_linear(obj_img_fts)
                )
            img_embeds = []
            for view_embed, obj_embed, view_len, obj_len in zip(
                view_img_embeds, obj_img_embeds, view_lens, obj_lens
            ):
                if obj_len > 0:
                    # import ipdb;ipdb.set_trace()
                    img_embeds.append(torch.cat([view_embed[:view_len], obj_embed[:obj_len]], 0))
                else:
                    img_embeds.append(view_embed[:view_len])
            img_embeds = pad_tensors_wgrad(img_embeds)
            pano_lens = view_lens + obj_lens
        else:
            img_embeds = view_img_embeds
            pano_lens = view_lens
        
        pano_embeds = img_embeds + \
                      self.img_embeddings.loc_layer_norm(self.img_embeddings.loc_linear(loc_fts)) + \
                      self.img_embeddings.nav_type_embedding(nav_types) + \
                      self.embeddings.token_type_embeddings(torch.ones(1, 1).long().to(device))
        pano_embeds = self.img_embeddings.layer_norm(pano_embeds)
        pano_embeds = self.img_embeddings.dropout(pano_embeds)

        pano_masks = gen_seq_masks(pano_lens)
        if self.img_embeddings.pano_encoder is not None:
            pano_embeds = self.img_embeddings.pano_encoder(
                pano_embeds, src_key_padding_mask=pano_masks.logical_not()
            )
        return pano_embeds, pano_masks
    
    def forward_bevvp_per_step(self, batch):
        batch_candi_bev = batch['batch_candi_bev']
        pano_inputs = batch['pano_inputs']
        batch_candi_bev_len = batch['batch_candi_bev_len']
        has_obj =  'obj_img_fts' in pano_inputs.keys()
        batch_candi_bev_embeds = self.img_embeddings.img_layer_norm(
            self.img_embeddings.img_linear(batch_candi_bev)
        )
        if has_obj:
            obj_img_fts = pano_inputs['obj_img_fts']
            if self.img_embeddings.obj_linear is None:
                obj_img_embeds = self.img_embeddings.img_layer_norm(
                    self.img_embeddings.img_linear(obj_img_fts)
                )
            else:
                obj_img_embeds = self.img_embeddings.obj_layer_norm(
                    self.img_embeddings.obj_linear(obj_img_fts)
                )
            bevvp_embeds = []
            for view_embed, obj_embed, obj_len in zip(
                batch_candi_bev_embeds, obj_img_embeds, pano_inputs['obj_lens']
            ):
                if obj_len > 0:
                    bevvp_embeds.append(torch.cat([view_embed[:], obj_embed[:obj_len]], 0))
                else:
                    bevvp_embeds.append(view_embed[:])
            bevvp_embeds = pad_tensors_wgrad(bevvp_embeds)
            bevvp_lens = batch_candi_bev_len + pano_inputs['obj_lens']
        else:
            bevvp_embeds = batch_candi_bev_embeds
        
        bevvp_embeds = self.img_embeddings.layer_norm(bevvp_embeds)
        bevvp_embeds = self.img_embeddings.dropout(bevvp_embeds)
        if has_obj:
            bevvp_masks = torch.cat((gen_seq_masks(batch_candi_bev_len), gen_seq_masks(pano_inputs['obj_lens'])), -1)
        else:
            bevvp_masks = gen_seq_masks(batch_candi_bev_len)
        if self.img_embeddings.pano_encoder is not None:
            bevvp_embeds = self.img_embeddings.pano_encoder(
                bevvp_embeds, src_key_padding_mask=bevvp_masks.logical_not()
            )
        return bevvp_embeds, bevvp_masks

    def forward_navigation_per_step(
        self, txt_embeds, txt_masks, gmap_img_embeds, gmap_step_ids, gmap_pos_fts, 
        gmap_masks, gmap_pair_dists, gmap_visited_masks, gmap_vpids,
        vp_img_embeds, vp_pos_fts, vp_masks, vp_nav_masks, vp_obj_masks, vp_cand_vpids,
        bevmap_img_embeds, bevmap_step_ids, bevmap_pos_fts, bevmap_masks, 
        bevmap_pair_dists, bevmap_visited_masks, bevmap_vpids, return_state, 
        finebevembs, bevpoints, candis, candi2bev_tmp,
        bevvp_embeds, bevvp_masks, bevvp_nav_masks, bevvp_obj_masks,
    ):
        batch_size = txt_embeds.size(0)
        if args.onlybev:
            gmap_embeds = gmap_img_embeds
        else:
            gmap_embeds = gmap_img_embeds + \
                        self.global_encoder.gmap_step_embeddings(gmap_step_ids) + \
                        self.global_encoder.gmap_pos_embeddings(gmap_pos_fts)

            if self.global_encoder.sprel_linear is not None:
                graph_sprels = self.global_encoder.sprel_linear(
                    gmap_pair_dists.unsqueeze(3)).squeeze(3).unsqueeze(1)
            else:
                graph_sprels = None
            gmap_embeds = self.global_encoder.encoder(
                txt_embeds, txt_masks, gmap_embeds, gmap_masks,
                graph_sprels=graph_sprels
            )

        # bev branch
        if args.bevglobal:
            bevmap_embeds = bevmap_img_embeds + \
                        self.bev_encoder.gmap_step_embeddings(bevmap_step_ids) + \
                        self.bev_encoder.gmap_pos_embeddings(bevmap_pos_fts)
            if self.bev_encoder.sprel_linear is not None:
                bevgraph_sprels = self.bev_encoder.sprel_linear(
                    bevmap_pair_dists.unsqueeze(3)).squeeze(3).unsqueeze(1)
            else:
                bevgraph_sprels = None
            bevmap_embeds = self.bev_encoder.encoder(
                txt_embeds, txt_masks, bevmap_embeds, bevmap_masks,
                graph_sprels=bevgraph_sprels
            )
        else:
            bevmap_embeds = None

        # bev local branch
        bevvp_embeds = self.bevlocal_encoder.encoder(txt_embeds, txt_masks, bevvp_embeds, bevvp_masks)
        
        if args.bevlocalsall:
            bevpoints_local = bevpoints
        else:
            bevlocal_index = findindex(args.bev_grid, args.bev_locals) # 11, 7
            bevpoints_local = bevpoints.index_select(1, bevlocal_index)

        # local branch
        if args.onlybev:
            vp_embeds = vp_img_embeds
        else:
            vp_embeds = vp_img_embeds + self.local_encoder.vp_pos_embeddings(vp_pos_fts)
            vp_embeds = self.local_encoder.encoder(txt_embeds, txt_masks, vp_embeds, vp_masks)
            
        # navigation logits
        if self.sap_fuse_linear is None or args.onlybev:
            fuse_weights1 = 0.5
        else:
            fuse_weights1 = torch.sigmoid(self.sap_fuse_linear(
                torch.cat([gmap_embeds[:, 0], vp_embeds[:, 0]], 1)
            ))

        if self.sap_fuse_linear_bev is None:
            fuse_weights2 = 0.5
        else:
            fuse_weights2 = torch.sigmoid(self.sap_fuse_linear_bev(
                torch.cat([bevmap_embeds[:, 0], bevvp_embeds[:, 0]], 1)
            ))
        bevweight = args.bev_weight if not args.onlybev else 1

        if args.onlybev:
            global_logits = None
            local_logits = None
        else:
            oglobal_logits = self.global_sap_head(gmap_embeds).squeeze(2)
            global_logits = oglobal_logits * fuse_weights1 * (1 - bevweight)
            global_logits.masked_fill_(gmap_visited_masks, -float('inf'))
            global_logits.masked_fill_(gmap_masks.logical_not(), -float('inf'))

            olocal_logits = self.local_sap_head(vp_embeds).squeeze(2)
            local_logits = olocal_logits * (1 - fuse_weights1) * (1 - bevweight)
            local_logits.masked_fill_(vp_nav_masks.logical_not(), -float('inf'))

        if args.bevglobal:
            if args.onlybev:
                bev_logits = self.bevglobal_sap_head(bevmap_embeds).squeeze(2) * fuse_weights2
            else:
                bev_logits = self.bevglobal_sap_head(bevmap_embeds).squeeze(2) * fuse_weights2 * bevweight
            bev_logits.masked_fill_(bevmap_visited_masks, -float('inf'))
            bev_logits.masked_fill_(bevmap_masks.logical_not(), -float('inf'))
        else:
            bev_logits = None

        
        bevlocal_logits = self.bevlocal_sap_head(bevvp_embeds).squeeze(2) * (1 - fuse_weights2) * bevweight
        bevlocal_logits.masked_fill_(bevvp_nav_masks.logical_not(), -float('inf'))

        # fusion
        if args.onlybev:
            fused_logits = torch.clone(bev_logits)
            fused_logits[:, 0] = fused_logits[:, 0] + bevlocal_logits[:, 0]  # stop
        else:
            fused_logits = torch.clone(global_logits + bev_logits) if args.bevglobal else torch.clone(global_logits)
            fused_logits[:, 0] = fused_logits[:, 0] + local_logits[:, 0] + bevlocal_logits[:, 0]  # stop
        for i in range(batch_size):
            visited_nodes = set([vp for vp, mask in zip(gmap_vpids[i], gmap_visited_masks[i]) if mask])
            tmp = {}
            bw_logits = 0
            for j, cand_vpid in enumerate(vp_cand_vpids[i]):
                # import ipdb;ipdb.set_trace()
                if j > 0:
                    if cand_vpid in visited_nodes:
                        if args.onlybev:
                            bw_logits += bevlocal_logits[i][j] 
                        else:
                            bw_logits += local_logits[i, j] + bevlocal_logits[i][j] 
                    else:
                        if args.onlybev:
                            tmp[cand_vpid] = bevlocal_logits[i][j]
                        else:
                            tmp[cand_vpid] =  local_logits[i, j] + bevlocal_logits[i][j]
            for j, vp in enumerate(gmap_vpids[i]):
                if j > 0 and vp not in visited_nodes:
                    if vp in tmp:
                        fused_logits[i, j] += tmp[vp]
                    else:
                        fused_logits[i, j] += bw_logits
                        # fused_logits[i, j] += bw_logits[vp]

        # original fusion
        if args.onlybev:
            ofused_logits = None
        else:
            ofused_logits = torch.clone(oglobal_logits)
            ofused_logits[:, 0] += olocal_logits[:, 0]   # stop
            for i in range(batch_size):
                ovisited_nodes = set([vp for vp, mask in zip(gmap_vpids[i], gmap_visited_masks[i]) if mask])
                otmp = {}
                obw_logits = 0
                for j, cand_vpid in enumerate(vp_cand_vpids[i]):
                    if j > 0:
                        if cand_vpid in ovisited_nodes:
                            obw_logits += olocal_logits[i, j]
                        else:
                            otmp[cand_vpid] = olocal_logits[i, j]
                for j, vp in enumerate(gmap_vpids[i]):
                    if j > 0 and vp not in ovisited_nodes:
                        if vp in otmp:
                            ofused_logits[i, j] += otmp[vp]
                        else:
                            ofused_logits[i, j] += obw_logits

        # object grounding logits
        if vp_obj_masks is not None and not args.onlybev:
            obj_logits = self.og_head(vp_embeds).squeeze(2)
            obj_logits.masked_fill_(vp_obj_masks.logical_not(), -float('inf'))
        else:
            obj_logits = None

        if bevvp_obj_masks is not None:
            bevobj_logits = self.og_head(bevvp_embeds).squeeze(2)
            bevobj_logits = torch.cat((torch.zeros(vp_obj_masks.shape[0], vp_obj_masks.shape[1] - bevobj_logits.shape[1]).cuda(), bevobj_logits), -1)
            bevobj_logits.masked_fill_(vp_obj_masks.logical_not(), -float('inf'))
        else:
            bevobj_logits = None
        
        if obj_logits is not None:
            out_obj_logits = obj_logits * (1 - bevweight) + bevobj_logits * bevweight
        else:
            out_obj_logits = bevobj_logits

        if args.bevglobal:
            hidden_state = txt_embeds[:, 0] * ( bevmap_embeds[:, 0] + vp_embeds[:, 0] + gmap_embeds[:, 0] )
        else:
            hidden_state = txt_embeds[:, 0] * ( vp_embeds[:, 0] + gmap_embeds[:, 0] )

        if return_state:
            return hidden_state
        else:
            outs = {
                'gmap_embeds': gmap_embeds,
                'vp_embeds': vp_embeds,
                'bev_logits': bev_logits,
                'global_logits': global_logits,
                'local_logits': local_logits,
                'fused_logits': fused_logits,
                'ofused_logits': ofused_logits,
                'bevlocal_logits': bevlocal_logits, # instead of bevlocal_logits,
                'finebevlocal_logits': bevlocal_logits,
                'waypoint_logits': None,
                'obj_logits': out_obj_logits,
                'hidden_state': hidden_state,
                'bevpoints_local': bevpoints_local,
            }
            return outs

    def forward_bev_per_step(
        self, candis, scans, argscamera, vps, shift=None
    ):
        bevembs = []
        # bs = cam_fts.shape[2]
        bs = args.batch_size
        hdkeys = [scan + '_' + vp for scan,vp in zip(scans, vps)]
        with h5py.File(args.bevfeaturepath, 'r') as bevf:
            for hdkey in hdkeys:
                bevft = bevf[hdkey][...].astype(np.float32)
                bevembs.append(bevft)
        bevembs = torch.from_numpy(np.stack(bevembs)).cuda()
        ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, self.bev_h - 0.5, self.bev_h, dtype=bevembs.dtype, device=bevembs.device),
                torch.linspace(
                    0.5, self.bev_w - 0.5, self.bev_w, dtype=bevembs.dtype, device=bevembs.device),
                indexing='ij'
            )
        ref_y = ref_y.reshape(-1)[None] / self.bev_h
        ref_x = ref_x.reshape(-1)[None] / self.bev_w
        ref_2d = torch.stack((ref_x, ref_y), -1)
        ref_2d = ref_2d.repeat(bs, 1, 1)
        bevpoints = ref_2d.clone()
        
        batch_originshift = torch.from_numpy(np.array([argscamera[i]['originshift'] for i in range(bs)])).cuda()
        batch_originshift = batch_originshift[:, None, :].repeat(1, self.bev_h * self.bev_w, 1)
        bevpoints[..., 0:1] = ref_2d[..., 0:1] * \
            (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0] + batch_originshift[..., 0:1]
        bevpoints[..., 1:2] = ref_2d[..., 1:2] * \
            (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1] + batch_originshift[..., 1:2]
        batch_candi_bev = []
        batch_candi_id = []
        self.candi2bev = {}
        candi2bev_tmp = []
        candi2bev_json = {}
        if os.path.exists(args.candi2bev_dir):
            with open(args.candi2bev_dir,'r') as f_old:
                candi2bev_json = json.load(f_old)

        for i in range(bs):
            candi_bevemb = []
            candi_id = []
            canindice_dict = {}
            allcanindice_dict = {}
            usedpoints = []
            allpoints = [numindex for numindex in range(100)]
            candi2bev_each = []
            self.candi2bev[scans[i]+'_'+vps[i]] = {}
            if scans[i]+'_'+vps[i] in candi2bev_json.keys():
                json_update = False
                for j,cc in enumerate(candis[i]):
                    self.candi2bev[scans[i]+'_'+vps[i]][j] = candi2bev_json[scans[i]+'_'+vps[i]][str(j)]
                    candi_bevemb.append(bevembs[i].index_select(0, torch.tensor(self.candi2bev[scans[i]+'_'+vps[i]][j]).cuda()).mean(0).clone().contiguous())
                    candi2bev_each.append(torch.tensor(self.candi2bev[scans[i]+'_'+vps[i]][j]).cuda())
            else:
                json_update = True
                for j,cc in enumerate(candis[i]):
                    ccpos = torch.from_numpy(np.array(cc['position'][:2]))[None, :].repeat(self.bev_h * self.bev_w, 1).cuda()
                    delta_xy = torch.pow(ccpos - bevpoints[i], 2)
                    allcanindice = delta_xy.sum(-1).sort(-1)[1]
                    allcanindice = allcanindice.tolist()
                    allcanindice_dict[j] = allcanindice
                    canindice_dict[j] = allcanindice[:args.bev_candinum]
                    usedpoints += allcanindice[:args.bev_candinum]
                    candi_id.append(cc['pointId'])
                unusedpoints = set(allpoints) - set(usedpoints)
                for unusedpoint in unusedpoints:
                    indexlist = [allcanindice_dict[j].index(unusedpoint) for j,cc in enumerate(candis[i])]
                    candi_index = indexlist.index(min(indexlist))
                    canindice_dict[candi_index] += [unusedpoint,]
                for j,cc in enumerate(candis[i]):
                    self.candi2bev[scans[i]+'_'+vps[i]][j] = canindice_dict[j]
                    candi_bevemb.append(bevembs[i].index_select(0, torch.tensor(self.candi2bev[scans[i]+'_'+vps[i]][j]).cuda()).mean(0).clone().contiguous())
                    candi2bev_each.append(torch.tensor(self.candi2bev[scans[i]+'_'+vps[i]][j]).cuda())
                
            candi2bev_tmp.append(candi2bev_each)

            candi_bevemb = torch.stack(candi_bevemb, 0)
            batch_candi_bev.append(candi_bevemb)
            batch_candi_id.append(candi_id)
        if json_update:
            candi2bev_json.update(self.candi2bev)
            with open(args.candi2bev_dir,'w') as f_new:
                json.dump(candi2bev_json, f_new)
        

        return bevembs.mean(1).clone(), batch_candi_bev, batch_candi_id, bevembs, bevpoints, candi2bev_tmp


    def forward(self, mode, batch, **kwargs):
        if mode == 'language':
            txt_embeds = self.forward_text(batch['txt_ids'], batch['txt_masks'])
            return txt_embeds

        elif mode == 'panorama':
            pano_embeds, pano_masks = self.forward_panorama_per_step(
                batch['view_img_fts'], batch['obj_img_fts'], batch['loc_fts'],
                batch['nav_types'], batch['view_lens'], batch['obj_lens']
            )
            return pano_embeds, pano_masks

        elif mode == 'cam_fts':
            bev_embeds, candi_bev, candi_id, finebevembs, bevpoints, candi2bev_tmp = self.forward_bev_per_step(
                batch['candis'],
                batch['scan'], batch['argscamera'],
                batch['vp']
                )
            return bev_embeds, candi_bev, candi_id, finebevembs, bevpoints, candi2bev_tmp

        elif mode == 'bev_fts':
            bevvp_embeds, bevvp_masks = self.forward_bevvp_per_step(batch)
            return bevvp_embeds, bevvp_masks
        elif mode == 'navigation':
             return self.forward_navigation_per_step(
                batch['txt_embeds'], batch['txt_masks'], batch['gmap_img_embeds'], 
                batch['gmap_step_ids'], batch['gmap_pos_fts'], batch['gmap_masks'],
                batch['gmap_pair_dists'], batch['gmap_visited_masks'], batch['gmap_vpids'], 
                batch['vp_img_embeds'], batch['vp_pos_fts'], batch['vp_masks'],
                batch['vp_nav_masks'], batch['vp_obj_masks'], batch['vp_cand_vpids'],
                batch['bevmap_img_embeds'], batch['bevmap_step_ids'], batch['bevmap_pos_fts'],
                batch['bevmap_masks'], batch['bevmap_pair_dists'], batch['bevmap_visited_masks'],
                batch['bevmap_vpids'], batch['return_state'], 
                batch['finebevembs'], batch['bevpoints'], batch['candis'], batch['candi2bev_tmp'],
                batch['bevvp_embeds'], batch['bevvp_masks'], batch['bevvp_nav_masks'], batch['bevvp_obj_masks'],
            )
        


            
       