# ------------------------------------------------------------------------------------
# Sparse DETR
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------------------


"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math
import numpy as np
from eval import TorchEval
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from util.dam import idx_to_flat_grid, attn_map_to_flat_grid, compute_corr

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer import build_deforamble_transformer
import copy

from scipy.spatial.transform import Rotation as R
def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False, args=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, output_dim=4, num_layers=3)
        self.orien_embed  = MLP(hidden_dim, hidden_dim, 4, 3)
        self.trans_embed = MLP(hidden_dim, hidden_dim, 3, 3)
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
            # will be splited into query_embed(query_pos) & tgt later
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        self.use_enc_aux_loss = args.use_enc_aux_loss
        self.rho = args.rho

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.orien_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.orien_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.trans_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.trans_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
 
        # hack implementation: a list of embedding heads (see the order)
        # n: dec_layers / m: enc_layers
        # [dec_0, dec_1, ..., dec_n-1, encoder, backbone, enc_0, enc_1, ..., enc_m-2]
        
        # at each layer of decoder (by default)
        num_pred = transformer.decoder.num_layers
        if self.two_stage:
            # at the end of encoder
            num_pred += 1  
        if self.use_enc_aux_loss:
            # at each layer of encoder (excl. the last)
            num_pred += transformer.encoder.num_layers - 1  
        
        if with_box_refine or self.use_enc_aux_loss:
            # individual heads with the same initialization
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            self.orien_embed = _get_clones(self.orien_embed, num_pred)
            self.trans_embed = _get_clones(self.trans_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            nn.init.constant_(self.orien_embed[0].layers[-1].bias.data[2:], -2.0)
        else:
            # shared heads
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            nn.init.constant_(self.orien_embed.layers[-1].bias.data[2:], -2.0)
            nn.init.constant_(self.trans_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.orien_embed = nn.ModuleList([self.orien_embed for _ in range(num_pred)])
            self.trans_embed = nn.ModuleList([self.trans_embed for _ in range(num_pred)])
            
        if two_stage:
            print("Wrong!")
            # hack implementation
            self.transformer.decoder.class_embed = self.class_embed
            self.transformer.decoder.bbox_embed = self.bbox_embed            
            for box_embed in self.transformer.decoder.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
                
        if self.use_enc_aux_loss:
            print("Wrong!")
            # the output from the last layer should be specially treated as an input of decoder
            num_layers_excluding_the_last = transformer.encoder.num_layers - 1
            self.transformer.encoder.aux_heads = True
            self.transformer.encoder.class_embed = self.class_embed[-num_layers_excluding_the_last:]
            self.transformer.encoder.bbox_embed = self.bbox_embed[-num_layers_excluding_the_last:] 
            for box_embed in self.transformer.encoder.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        ###########
        # Backbone
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        
        # multi-scale features projected from ~C5 with 1x1 conv
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
            
        # multi-scale features smaller than C5 projected with 2 strided 3x3 conv
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    # feature scale 1/32 
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    # feature scale <1/64: recursively downsample the last projection
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        ###########
        # Transformer encoder & decoder
        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        (hs, init_reference, inter_references, 
         enc_outputs_class, enc_outputs_coord_unact, 
         backbone_mask_prediction,
         enc_inter_outputs_class, enc_inter_outputs_coord, 
         sampling_locations_enc, attn_weights_enc, 
         sampling_locations_dec, attn_weights_dec,
         backbone_topk_proposals, spatial_shapes, level_start_index) = \
            self.transformer(srcs, masks, pos, query_embeds)

        ###########
        # Detection heads
        outputs_classes = []
        outputs_coords = []
        outputs_oriens = []
        outputs_transes = []
        for lvl in range(len(hs)):
            # lvl: level of decoding layer
            outputs_class = self.class_embed[lvl](hs[lvl])
            outputs_coord = self.bbox_embed[lvl](hs[lvl])
            outputs_orien = self.orien_embed[lvl](hs[lvl])
            outputs_trans = self.trans_embed[lvl](hs[lvl])
            
            assert init_reference is not None and inter_references is not None
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            if reference.shape[-1] == 4:
                outputs_coord += reference
                outputs_orien += reference
            else:
                assert reference.shape[-1] == 2
                outputs_coord[..., :2] += reference
                outputs_orien[..., :2] += reference
            
            outputs_coord = outputs_coord.sigmoid()
            outputs_orien = outputs_orien.sigmoid()
            
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_oriens.append(outputs_orien)
            outputs_transes.append(outputs_trans)

        
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_orien = torch.stack(outputs_oriens)
        outputs_trans = torch.stack(outputs_transes)
        # outputs_orien = self.orien_embed(hs).sigmoid()
        # outputs_trans = self.trans_embed(hs)
        # print("outputs_orien", outputs_orien.shape, outputs_orien[0])

        # the topmost layer output
        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "pred_orien": outputs_orien[-1], 
            "pred_trans": outputs_trans[-1],
            "sampling_locations_enc": sampling_locations_enc,
            "attn_weights_enc": attn_weights_enc,
            "sampling_locations_dec": sampling_locations_dec,
            "attn_weights_dec": attn_weights_dec,
            "spatial_shapes": spatial_shapes,
            "level_start_index": level_start_index,
        }
        if backbone_topk_proposals is not None:
            out["backbone_topk_proposals"] = backbone_topk_proposals
        
        if self.aux_loss:
            # make loss from every intermediate layers (excluding the last one)
            out['aux_outputs'] = self._set_aux_loss(outputs_class[:-1], outputs_coord[:-1], outputs_orien[:-1], outputs_trans[:-1])

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}

        if self.rho:
            out["backbone_mask_prediction"] = backbone_mask_prediction
            
        if self.use_enc_aux_loss:
            out['aux_outputs_enc'] = self._set_aux_loss(enc_inter_outputs_class, enc_inter_outputs_coord)
        
        if self.rho:
            out["sparse_token_nums"] = self.transformer.sparse_token_nums

        out['mask_flatten'] = torch.cat([m.flatten(1) for m in masks], 1)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_orien, outputs_trans):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_orien': c, 'pred_trans': d}
                for a, b, c, d in zip(outputs_class, outputs_coord, outputs_orien, outputs_trans)]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, args, cld):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.cld = cld
        self.teval = TorchEval()
        self.focal_alpha = args.focal_alpha
        self.eff_specific_head = args.eff_specific_head

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2)
        loss_ce = loss_ce * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses
    
    def martix_to_qu(self, matrix):
        r = R.from_matrix(matrix)
        print(r.as_quat())
        
    def cal_add_cuda(self, pred_RT, gt_RT, p3ds):
        pred_p3ds = torch.mm(p3ds, pred_RT.transpose(1, 0).float())
        gt_p3ds = torch.mm(p3ds, gt_RT.transpose(1, 0)) 
        dis = torch.norm(pred_p3ds - gt_p3ds, dim=1)
        return torch.mean(dis)

    def cal_adds_cuda(self, pred_RT, gt_RT, p3ds):
        N, _ = p3ds.size()
        pd = torch.mm(p3ds, pred_RT.transpose(1, 0).float()) 
        pd = pd.view(1, N, 3).repeat(N, 1, 1)
        gt = torch.mm(p3ds, gt_RT.transpose(1, 0)) 
        gt = gt.view(N, 1, 3).repeat(1, N, 1)
        dis = torch.norm(pd - gt, dim=2)
        mdis = torch.min(dis, dim=1)[0]
        return torch.mean(mdis)
    
    # def loss_poses(self, outputs, targets, indices, num_boxes):
    #     idx = self._get_src_permutation_idx(indices)
    #     src_orien = outputs['pred_orien'][idx]
    #     src_trans = outputs['pred_trans'][idx]
    #     # print("trans", type(src_trans), src_trans)
    #     # print("orien", type(src_orien), src_orien)
    #     p3d = []
    #     cnt = 0
    #     # for t, (_, J) in zip(targets, indices):
    #     #     print("target", targets[cnt])
    #     #     print("1", J)
    #     #     print("2", t['labels'])
    #     #     print("3", t["labels"][J])
    #     #     cnt += 1
    #     target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

    #     for id in target_classes_o:
    #         # print("id", id)
    #         ranIdx = np.random.choice(2620, 1500, replace= False)
    #         # print("cld", len(self.cld))
    #         # print("ranIdx", ranIdx)
    #         # break
    #         p3d.append(self.cld[id - 1][ranIdx])
    #     target_orien = torch.cat([t['RTs'][i][:, :, :3] for t, (_, i) in zip(targets, indices)], dim=0)

    #     src_rotMat = []
    #     # print(src_orien.shape)
    #     for orien in src_orien:
    #         # print("3333333333333333333")
    #         # print("orien", type(src_orien), src_orien)
    #         # print("orien before", type(orien), orien)
    #         # print(l)
    #         # edit
    #         # a = orien.clone().detach()
    #         # orien_np = a.cpu().numpy()
    #         orien_np = orien.detach().cpu().numpy()
    #         # print("orien after", type(orien), orien)
    #         r = R.from_quat(orien_np).as_matrix()
    #         src_rotMat.append(r)
    #     p3d = np.array(p3d)
    #     p3d = torch.Tensor(p3d).cuda()
    #     src_rotMat = np.array(src_rotMat)
    #     src_rotMat = torch.Tensor(src_rotMat).cuda()
    #     #print(target_classes_o.shape)
    #     # target_quan = []
    #     # for orien in target_orien:
    #     #     orien_np = orien.detach().cpu().numpy()
    #     #     r = R.from_matrix(orien_np).as_quat()
    #     #     target_quan.append(r)
    #     # target_quan = np.array(target_quan)
    #     # target_quan = torch.Tensor(target_quan).cuda()
    #     target_trans = torch.cat([t['RTs'][i][:, :, 3] for t, (_, i) in zip(targets, indices)], dim=0)
    #     loss_orien = 0
    #     for i in range(src_orien.shape[0]):
    #         # if target_classes_o[i] == 0 or target_classes_o[i] == 22:
    #         #     print("wrong")
    #         if(target_classes_o[i] in [13, 16, 19, 20, 21]):
    #             loss_orien += self.cal_adds_cuda(src_rotMat[i], target_orien[i], p3d[i])
    #         else:
    #             loss_orien += self.cal_add_cuda(src_rotMat[i], target_orien[i], p3d[i])
    #         #loss_orien += 2 - 2 *(src_orien[i].dot(target_quan[i])) ** 2
   
    #     loss_fn = torch.nn.MSELoss(reduction='sum')
    #     # dim:26*3
    #     loss_trans = loss_fn(src_trans, target_trans)
    #     # loss_trans = F.l1_loss(src_trans, target_trans, reduction='none')
    #     losses = {}
    #     losses['loss_orien'] = loss_orien.sum() / src_trans.shape[0]
    #     losses['loss_trans'] = loss_trans.sum() / src_trans.shape[0]
        
    #     print("orien", type(src_orien), src_orien)
    #     return losses
    
    def loss_poses(self, outputs, targets, indices, num_boxes):
 
        idx = self._get_src_permutation_idx(indices)
    
        src_orien = outputs['pred_orien'][idx]
        src_trans = outputs['pred_trans'][idx]
        
        # p3d = []
        # # print("targets", targets[0])
        # # print("indices", indices[0])
        # target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        # # print("target_classes_o", target_classes_o)
        # for id in target_classes_o:
        #     ranIdx = np.random.choice(2620, 1500,replace= False)
        #     p3d.append(self.cld[id - 1][ranIdx])
        ###### edit
        target_quan = []
        target_orien = torch.cat([t['RTs'][i][:, :, :3] for t, (_, i) in zip(targets, indices)], dim=0)
        # print('target_orien', target_orien.shape)
        for orien in target_orien:
            orien_np = orien.detach().cpu().numpy()
            # print("orien_np", orien_np.shape)
            r = R.from_matrix(orien_np).as_quat()
            target_quan.append(r)
        target_quan = np.array(target_quan)
        target_quan = torch.Tensor(target_quan).cuda()
        eps = 1e-4
        # n_obj = len(target_orien)
        # bs, q_dim = target_orien.shape
        # print("src_orien", src_orien.shape)
        # print("target_orien", target_quan.shape)
        dot_product = torch.mul(src_orien, target_quan)
        dp_sum = torch.sum(dot_product, 1)
        dp_square = torch.square(dp_sum)
        loss_quat = - torch.log(dp_square + eps)
        # print("loss_quat", loss_quat)
        # losses["loss_rot"] = loss_quat.sum() / n_obj
        ###### edit
        # src_rotMat = []
        # #print(src_orien.shape)
        # for orien in src_orien:
        #     orien_np = orien.detach().cpu().numpy()
        #     r = R.from_quat(orien_np).as_matrix()
        #     src_rotMat.append(r)
        # p3d = np.array(p3d)
        # p3d = torch.Tensor(p3d).cuda()
        # src_rotMat = np.array(src_rotMat)
        # src_rotMat = torch.Tensor(src_rotMat).cuda()
        #print(target_classes_o.shape)
        # target_quan = []
        # for orien in target_orien:
        #     orien_np = orien.detach().cpu().numpy()
        #     r = R.from_matrix(orien_np).as_quat()
        #     target_quan.append(r)
        # target_quan = np.array(target_quan)
        # target_quan = torch.Tensor(target_quan).cuda()

        target_trans = torch.cat([t['RTs'][i][:, :, 3] for t, (_, i) in zip(targets, indices)], dim=0)
        # loss_orien = 0
        # for i in range(src_orien.shape[0]):
        #     if(target_classes_o[i] in [13, 16, 19, 20, 21]):
        #         loss_orien +=self.cal_adds_cuda(src_rotMat[i], target_orien[i], p3d[i])
        #     else:
        #         loss_orien += self.cal_add_cuda(src_rotMat[i], target_orien[i], p3d[i])
        #     #loss_orien += 2 - 2 *(src_orien[i].dot(target_quan[i])) ** 2
        # loss_fn = torch.nn.MSELoss(reduction='sum')
        # # dim:26*3
        # loss_trans = loss_fn(src_trans, target_trans)
        
        loss_trans = F.mse_loss(src_trans, target_trans, reduction='none')
        loss_trans = torch.sum(loss_trans, dim=1)
        loss_trans = torch.sqrt(loss_trans)
        
        # loss_trans = F.l1_loss(src_trans, target_trans, reduction='none')
        losses = {}

        losses['loss_orien'] = loss_quat.sum() / src_orien.shape[0]
        losses['loss_trans'] = loss_trans.sum() / src_trans.shape[0]
        # losses['loss_orien'] = 0
        # losses['loss_trans'] = 0
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses
    
    def loss_mask_prediction(self, outputs, targets, indices, num_boxes, layer=None):
        assert "backbone_mask_prediction" in outputs
        assert "sampling_locations_dec" in outputs
        assert "attn_weights_dec" in outputs
        assert "spatial_shapes" in outputs
        assert "level_start_index" in outputs

        mask_prediction = outputs["backbone_mask_prediction"] 
        loss_key = "loss_mask_prediction"

        sampling_locations_dec = outputs["sampling_locations_dec"]
        attn_weights_dec = outputs["attn_weights_dec"]
        spatial_shapes = outputs["spatial_shapes"]
        level_start_index = outputs["level_start_index"]

        flat_grid_attn_map_dec = attn_map_to_flat_grid(
            spatial_shapes, level_start_index, sampling_locations_dec, attn_weights_dec).sum(dim=(1,2))

        losses = {}

        if 'mask_flatten' in outputs:
            flat_grid_attn_map_dec = flat_grid_attn_map_dec.masked_fill(
                outputs['mask_flatten'], flat_grid_attn_map_dec.min()-1)
                
        sparse_token_nums = outputs["sparse_token_nums"]
        num_topk = sparse_token_nums.max()

        topk_idx_tgt = torch.topk(flat_grid_attn_map_dec, num_topk)[1]
        target = torch.zeros_like(mask_prediction)
        for i in range(target.shape[0]):
            target[i].scatter_(0, topk_idx_tgt[i][:sparse_token_nums[i]], 1)

        losses.update({loss_key: F.multilabel_soft_margin_loss(mask_prediction, target)})

        return losses

    @torch.no_grad()
    def corr(self, outputs, targets, indices, num_boxes):
        if "backbone_topk_proposals" not in outputs.keys():
            return {}

        assert "backbone_topk_proposals" in outputs
        assert "sampling_locations_dec" in outputs
        assert "attn_weights_dec" in outputs
        assert "spatial_shapes" in outputs
        assert "level_start_index" in outputs

        backbone_topk_proposals = outputs["backbone_topk_proposals"]
        sampling_locations_dec = outputs["sampling_locations_dec"]
        attn_weights_dec = outputs["attn_weights_dec"]
        spatial_shapes = outputs["spatial_shapes"]
        level_start_index = outputs["level_start_index"]

        flat_grid_topk = idx_to_flat_grid(spatial_shapes, backbone_topk_proposals)
        flat_grid_attn_map_dec = attn_map_to_flat_grid(
            spatial_shapes, level_start_index, sampling_locations_dec, attn_weights_dec).sum(dim=(1,2))
        corr = compute_corr(flat_grid_topk, flat_grid_attn_map_dec, spatial_shapes)

        losses = {}
        losses["corr_mask_attn_map_dec_all"] = corr[0].mean()
        for i, _corr in enumerate(corr[1:]):
            losses[f"corr_mask_attn_map_dec_{i}"] = _corr.mean()
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'poses': self.loss_poses,
            "mask_prediction": self.loss_mask_prediction,
            "corr": self.corr,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
    
    def evals(self, outputs, targets, indices):
        idx = self._get_src_permutation_idx(indices)
        
        src_orien = outputs['pred_orien'][idx]
        src_orien = quaternion_to_matrix(src_orien)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        
        
        src_trans = outputs['pred_trans'][idx]
        source_RTs = torch.cat([ torch.cat((orien,trans.view(3,-1)), dim= 1 )for (orien, trans) in zip(src_orien, src_trans)], dim= 0).view(-1,3,4)
        target_RTs = torch.cat([t['RTs'][i][:, :, :] for t, (_, i) in zip(targets, indices)], dim=0)
        
        #target_orien = torch.cat([t['RTs'][i][:, :, :3] for t, (_, i) in zip(targets, indices)], dim=0)
        #target_trans = torch.cat([t['RTs'][i][:, :, 3] for t, (_, i) in zip(targets, indices)], dim=0)
        
        self.teval.eval_pose_parallel(target_classes_o, source_RTs, target_classes_o, target_RTs, )
        
    def forward(self, outputs, targets, eval=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() 
                               if k not in ['aux_outputs', 'enc_outputs', 'backbone_outputs', 'mask_flatten']}

        # Retrieve the matching between the outputs of the last layer and the targets
        # print("outputs_without_aux", outputs_without_aux)
        indices = self.matcher(outputs_without_aux, targets)
        # print("indices", indices)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss in ['masks', "mask_prediction", "corr"]:
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            if not self.eff_specific_head:
                for bt in bin_targets:
                    bt['labels'] = torch.zeros_like(bt['labels'])  # all labels are zero (meaning foreground)
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss in ['masks', "mask_prediction", "corr"]:
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        if 'backbone_outputs' in outputs:
            backbone_outputs = outputs['backbone_outputs']
            bin_targets = copy.deepcopy(targets)
            if not self.eff_specific_head:
                for bt in bin_targets:
                    bt['labels'] = torch.zeros_like(bt['labels'])  # all labels are zero (meaning foreground)
            indices = self.matcher(backbone_outputs, bin_targets)
            for loss in self.losses:
                if loss in ['masks', "mask_prediction", "corr"]:
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, backbone_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_backbone': v for k, v in l_dict.items()}
                losses.update(l_dict)
                
        if 'aux_outputs_enc' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs_enc']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss in ['masks', "mask_prediction", "corr"]:
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                    
        if eval == True:
            self.evals(outputs, targets, indices)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
def load_model_pcd():
    cld = []
    f = open('./YCB_Video_Dataset/image_sets/classes.txt')
    for line in f.readlines():
        if '\n' in line:
            line = line[:-1]
        pointxyz = np.loadtxt('./YCB_Video_Dataset/models/'+ line +'/points.xyz', dtype=np.float32)
        cld.append(pointxyz)
    return cld

def build(args):
    num_classes = 20 if args.dataset_file != 'coco' else 91
    num_classes = 22 if args.dataset_file == 'YCB_V' else num_classes
    if args.dataset_file == "coco_panoptic":
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)
    cld = load_model_pcd()
    
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        args=args,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    weight_dict['loss_orien'] = 1 # 1
    weight_dict['loss_trans'] = 1 # 1

    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
        
    # TODO this is a hack
    aux_weight_dict = {}
    
    if args.aux_loss:
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            
    if args.two_stage:
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        
    if args.use_enc_aux_loss:
        for i in range(args.enc_layers - 1):
            aux_weight_dict.update({k + f'_enc_{i}': v for k, v in weight_dict.items()})
            
    if args.rho:
        aux_weight_dict.update({k + f'_backbone': v for k, v in weight_dict.items()})
        
    if aux_weight_dict:
        weight_dict.update(aux_weight_dict)

    weight_dict['loss_mask_prediction'] = args.mask_prediction_coef

    losses = ['labels', 'boxes', 'cardinality', "corr", 'poses']
    if args.masks:
        losses += ["masks"]
    if args.rho:
        losses += ["mask_prediction"]
    
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, args, cld)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
