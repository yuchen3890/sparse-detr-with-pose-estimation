# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import torch.nn.functional as F
import math
import os
import cv2
import sys
import argparse
from pathlib import Path
from typing import Iterable
from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

import util.misc as utils
from cv2 import circle
from models import build_model
from datasets.ycbv import make_face_transforms

import matplotlib.pyplot as plt
import time
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
p3d = []
for cls in open('./YCB_Video_Dataset/image_sets/classes.txt'):
    if '\n' in cls:
        cls = cls[:-1]
    pointxyz = np.loadtxt('./YCB_Video_Dataset/models/'+cls+'/points.xyz', dtype=np.float32) 
    p3d.append(pointxyz)

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h,
                          img_w, img_h
                          ], dtype=torch.float32)
    return b

def get_images(in_path):
    img_files = []
    # for (dirpath, dirnames, filenames) in os.walk(in_path):
    #     for file in filenames:
    #         filename, ext = os.path.splitext(file)
    #         ext = str.lower(ext)z
    #         if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
    #             img_files.append(os.path.join(dirpath, file))
    # img_files = ['./YCB_Video_Dataset/data/0011/000001-color.png']
    # img_files = ['./ycbv_BOP/test/000058/rgb/000030.png']
    # img_files = ['./000001-color.png']
    img_files = ['./test.png']
    return img_files


# def get_args_parser():
#     parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
#     parser.add_argument('--lr', default=1e-4, type=float)
#     parser.add_argument('--lr_backbone', default=1e-5, type=float)
#     parser.add_argument('--batch_size', default=6, type=int)
#     parser.add_argument('--weight_decay', default=1e-4, type=float)
#     parser.add_argument('--epochs', default=300, type=int)
#     parser.add_argument('--lr_drop', default=200, type=int)
#     parser.add_argument('--clip_max_norm', default=0.1, type=float,
#                         help='gradient clipping max norm')

#     # Variants of Deformable DETR
#     parser.add_argument('--with_box_refine', default=False, action='store_true')
#     parser.add_argument('--two_stage', default=False, action='store_true')

#     # Model parameters
#     parser.add_argument('--frozen_weights', type=str, default=None,
#                         help="Path to the pretrained model. If set, only the mask head will be trained")
#     # * Backbone
#     parser.add_argument('--backbone', default='resnet50', type=str,
#                         help="Name of the convolutional backbone to use")
#     parser.add_argument('--dilation', action='store_true',
#                         help="If true, we replace stride with dilation in the last convolutional block (DC5)")
#     parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
#                         help="Type of positional embedding to use on top of the image features")
#     parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

#     # * Modified architecture
#     parser.add_argument('--backbone_from_scratch', default=False, action='store_true')
#     parser.add_argument('--finetune_early_layers', default=False, action='store_true')
#     parser.add_argument('--scrl_pretrained_path', default='', type=str)

#     # * Transformer
#     parser.add_argument('--enc_layers', default=6, type=int,
#                         help="Number of encoding layers in the transformer")
#     parser.add_argument('--dec_layers', default=6, type=int,
#                         help="Number of decoding layers in the transformer")
#     parser.add_argument('--dim_feedforward', default=1024, type=int,
#                         help="Intermediate size of the feedforward layers in the transformer blocks")
#     parser.add_argument('--hidden_dim', default=256, type=int,
#                         help="Size of the embeddings (dimension of the transformer)")
#     parser.add_argument('--dropout', default=0.1, type=float,
#                         help="Dropout applied in the transformer")
#     parser.add_argument('--nheads', default=8, type=int,
#                         help="Number of attention heads inside the transformer's attentions")
#     parser.add_argument('--num_queries', default=20, type=int,
#                         help="Number of query slots")
#     parser.add_argument('--pre_norm', action='store_true')
#     parser.add_argument('--dec_n_points', default=4, type=int)
#     parser.add_argument('--enc_n_points', default=4, type=int)

#     # * Efficient DETR
#     parser.add_argument('--eff_query_init', default=False, action='store_true')
#     parser.add_argument('--eff_specific_head', default=False, action='store_true')

#     # * Sparse DETR
#     parser.add_argument('--use_enc_aux_loss', default=False, action='store_true')
#     parser.add_argument('--rho', default=0., type=float)


#     # * Segmentation
#     parser.add_argument('--masks', action='store_true',
#                         help="Train segmentation head if the flag is provided")

#     # # Loss
#     parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
#                         help="Disables auxiliary decoding losses (loss at each layer)")
#     # * Matcher
#     parser.add_argument('--set_cost_class', default=1, type=float,
#                         help="Class coefficient in the matching cost")
#     parser.add_argument('--set_cost_bbox', default=5, type=float,
#                         help="L1 box coefficient in the matching cost")
#     parser.add_argument('--set_cost_giou', default=2, type=float,
#                         help="giou box coefficient in the matching cost")
    
#     # * Loss coefficients
#     parser.add_argument('--mask_loss_coef', default=1, type=float)
#     parser.add_argument('--dice_loss_coef', default=1, type=float)
#     parser.add_argument('--cls_loss_coef', default=2, type=float)
#     parser.add_argument('--bbox_loss_coef', default=5, type=float)
#     parser.add_argument('--giou_loss_coef', default=2, type=float)
#     parser.add_argument('--mask_prediction_coef', default=1, type=float)
#     parser.add_argument('--focal_alpha', default=0.25, type=float)
#     parser.add_argument('--eos_coef', default=0.1, type=float,
#                         help="Relative classification weight of the no-object class")


#     # dataset parameters
#     parser.add_argument('--dataset_file', default='YCB_V')
#     parser.add_argument('--data_path', type=str)
#     parser.add_argument('--data_panoptic_path', type=str)
#     parser.add_argument('--remove_difficult', action='store_true')

#     parser.add_argument('--output_dir', default='',
#                         help='path where to save the results, empty for no saving')
#     parser.add_argument('--device', default='cuda',
#                         help='device to use for training / testing')
#     parser.add_argument('--resume', default='./output/checkpoint0014.pth', help='resume from checkpoint')

#     parser.add_argument('--thresh', default=0.5, type=float)

#     return parser

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Modified architecture
    parser.add_argument('--backbone_from_scratch', default=False, action='store_true')
    parser.add_argument('--finetune_early_layers', default=False, action='store_true')
    parser.add_argument('--scrl_pretrained_path', default='', type=str)

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=10, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    
    # * Efficient DETR
    parser.add_argument('--eff_query_init', default=False, action='store_true')
    parser.add_argument('--eff_specific_head', default=False, action='store_true')

    # * Sparse DETR
    parser.add_argument('--use_enc_aux_loss', default=False, action='store_true')
    parser.add_argument('--rho', default=0., type=float)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float, # 2
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=1, type=float, # 5
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float, # 2
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=1, type=float) # 2
    parser.add_argument('--bbox_loss_coef', default=1, type=float) # 5
    parser.add_argument('--giou_loss_coef', default=2, type=float) # 2
    parser.add_argument('--mask_prediction_coef', default=1, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # * dataset parameters
    parser.add_argument('--dataset_file', default='YCB_V')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--coco_path', default='./data/coco', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='./output/checkpoint0079.pth', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    
    # * benchmark
    parser.add_argument('--approx_benchmark_only', default=False, action='store_true')
    parser.add_argument('--benchmark_only', default=False, action='store_true')
    parser.add_argument('--no_benchmark', dest='benchmark', action='store_false')

    parser.add_argument('--thresh', default=0.5, type=float) # edit

    return parser

intrinsic_matrix = {
    'linemod': np.array([[572.4114, 0.,         325.2611],
                        [0.,        573.57043,  242.04899],
                        [0.,        0.,         1.]]),
    'blender': np.array([[700.,     0.,     320.],
                         [0.,       700.,   240.],
                         [0.,       0.,     1.]]),
    'pascal': np.asarray([[-3000.0, 0.0,    0.0],
                         [0.0,      3000.0, 0.0],
                         [0.0,      0.0,    1.0]]),
    'ycb_K1': np.array([[1066.778, 0.        , 312.9869],
                        [0.      , 1067.487  , 241.3109],
                        [0.      , 0.        , 1.0]], np.float32),
    'ycb_K2': np.array([[1077.836, 0.        , 323.7872],
                        [0.      , 1078.189  , 279.6921],
                        [0.      , 0.        , 1.0]], np.float32)
}

def project_p3d(p3d, cam_scale, K=intrinsic_matrix['ycb_K1']):
    if type(K) == str:
        K = intrinsic_matrix[K]
    p3d = p3d * cam_scale
    p2d = np.dot(p3d, K.T)
    p2d_3 = p2d[:, 2]
    p2d_3[np.where(p2d_3 < 1e-8)] = 1.0
    p2d[:, 2] = p2d_3
    p2d = np.around((p2d[:, :2] / p2d[:, 2:])).astype(np.int32)
    return p2d

def rotation_6d_to_matrix(rot_6d):
        """
        Given a 6D rotation output, calculate the 3D rotation matrix in SO(3) using the Gramm Schmit process
        For details: https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.pdf
        """
        bs, n_q, _ = rot_6d.shape
        rot_6d = rot_6d.view(-1, 6)
        m1 = rot_6d[:, 0:3]
        m2 = rot_6d[:, 3:6]

        x = F.normalize(m1, p=2, dim=1)
        z = torch.cross(x, m2, dim=1)
        z = F.normalize(z, p=2, dim=1)
        y = torch.cross(z, x, dim=1)
        rot_matrix = torch.cat((x.view(-1, 3, 1), y.view(-1, 3, 1), z.view(-1, 3, 1)), 2)  # Rotation Matrix lying in the SO(3)
        rot_matrix = rot_matrix.view(bs, n_q, 3, 3)  #.transpose(2, 3)
        return rot_matrix

def draw_p2ds( img, p2ds, r=1, color=[(255, 0, 0)]):
        # print("p2ds", p2ds)
        if type(color) == tuple:
            color = [color]
        if len(color) != p2ds.shape[0]:
            color = [color[0] for i in range(p2ds.shape[0])]
        h, w = img.shape[0], img.shape[1]
        for pt_2d, c in zip(p2ds, color):
            # print("C", c)
            pt_2d[0] = np.clip(pt_2d[0], 0, w)
            pt_2d[1] = np.clip(pt_2d[1], 0, h)
            img = circle(
                img, (pt_2d[0], pt_2d[1]), r, c, -1
            )
        return img

@torch.no_grad()
def infer(images_path, model, postprocessors, device, output_path):
    model.eval()
    duration = 0
    for img_sample in images_path:
        filename = os.path.basename(img_sample)
        print("processing...{}".format(filename))
        orig_image = Image.open(img_sample).convert('RGB') # edit
        w, h = orig_image.size
        transform = make_face_transforms("test")
        dummy_target = {
            "size": torch.as_tensor([int(h), int(w)]),
            "orig_size": torch.as_tensor([int(h), int(w)])
        }
        # print("orig_image", h, w, orig_image)
        image, targets = transform(orig_image, dummy_target)
        image = image.unsqueeze(0)
        image = image.to(device)

        conv_features, enc_attn_weights, dec_attn_weights = [], [], []
        hooks = [
            model.backbone[-2].register_forward_hook(
                        lambda self, input, output: conv_features.append(output)

            ),
            model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                        lambda self, input, output: enc_attn_weights.append(output[1])

            ),
            model.transformer.decoder.layers[-1].cross_attn.register_forward_hook( # edit
                        lambda self, input, output: dec_attn_weights.append(output[1])

            ),
        ]
        start_t = time.perf_counter()
        outputs = model(image)
        end_t = time.perf_counter()
        outputs["pred_logits"] = outputs["pred_logits"].cpu()
        outputs["pred_boxes"] = outputs["pred_boxes"].cpu()
       
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
      
        # keep = probas.max(-1).values > 0.85
        keep = probas.max(-1).values > args.thresh
       
        idxs = probas[keep].argmax(-1).tolist()
        #print(probas[keep].argmax(-1))
        
        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], orig_image.size)
        probas = probas[keep].cpu().data.numpy()
       
        for hook in hooks:
            hook.remove()

        conv_features = conv_features[0]
        enc_attn_weights = enc_attn_weights[0]
        dec_attn_weights = dec_attn_weights[0].cpu()

        # get the feature map shape
        h, w = conv_features['0'].tensors.shape[-2:]

        if len(bboxes_scaled) == 0:
            continue
        img = np.array(orig_image)
        #outputs['pred_orien'] = outputs['pred_orien'] / outputs['pred_orien'].norm(dim=1, keepdim=True).clamp(min=1e-7)
       
        # edit
        # orien_mat = quaternion_to_matrix(outputs['pred_orien'])
        print("outputs['pred_orien']", outputs['pred_orien'].shape)
        orien_mat = rotation_6d_to_matrix(outputs['pred_orien'])
        print("orien_mat", orien_mat.shape)
        
        trans_cpu = outputs['pred_trans'].cpu()
        # print("trans_cpu", trans_cpu)
        #print(s[0, keep].shape)
        #show_kp_img = np.zeros((480, 640, 3), np.uint8)
        for i , idx in enumerate(idxs):
            pointxyz  = p3d[idx - 1].copy()
            rotMat = orien_mat[0, keep][i].cpu()
            
            pointxyz = pointxyz.dot(rotMat) + trans_cpu[0, keep][i].numpy()
            # pointxyz = np.dot(rotMat, pointxyz.transpose(1, 0)) + trans_cpu[0, keep][i].numpy()
            
            kp_2ds = project_p3d(pointxyz, 1000)
        #color = bs_utils.get_label_color(cls_id.item())
            img = draw_p2ds(img, kp_2ds, r=3)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for idx, box in enumerate(bboxes_scaled):
            bbox = box.cpu().data.numpy()
            bbox = bbox.astype(np.int32)
            bbox = np.array([
                [bbox[0], bbox[1]],
                [bbox[2], bbox[1]],
                [bbox[2], bbox[3]],
                [bbox[0], bbox[3]],
                ])
    
            bbox = bbox.reshape((4, 2))
            cv2.polylines(img, [bbox], True, (0, 255, 0), 2)
        
        img_save_path = os.path.join(output_path, filename)
        cv2.imwrite(img_save_path, img)
        cv2.imshow("img", img)
        cv2.waitKey()
        infer_time = end_t - start_t
        duration += infer_time
        print("Processing...{} ({:.3f}s)".format(filename, infer_time))

    avg_duration = duration / len(images_path)
    print("Avg. Time: {:.3f}s".format(avg_duration))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    model, _, postprocessors = build_model(args)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    model.to(device)
    image_paths = get_images(args.data_path)

    infer(image_paths, model, postprocessors, device, args.output_dir)