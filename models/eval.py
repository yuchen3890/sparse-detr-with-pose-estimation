#!/usr/bin/env python3
import os
import time
import torch
import numpy as np
import random
import pickle as pkl
import concurrent.futures

p3d = []
for cls in open('./YCB_Video_Dataset/classes.txt'):
    if '\n' in cls:
        cls = cls[:-1]
    pointxyz = np.loadtxt('./YCB_Video_Dataset/models/'+cls+'/points.xyz', dtype=np.float32) 
    p3d.append(pointxyz)

def get_pointxyz_cuda(
    cls, ds_type='ycb'
):
    if ds_type == "ycb":
        
        ptsxyz_cu = torch.from_numpy(p3d[cls].astype(np.float32)).cuda()
        
        return ptsxyz_cu.clone()
    # else:
    #     if cls in self.lm_cls_ptsxyz_cuda_dict.keys():
    #         return self.lm_cls_ptsxyz_cuda_dict[cls].clone()
    #     ptsxyz = self.get_pointxyz(cls, ds_type)
    #     ptsxyz_cu = torch.from_numpy(ptsxyz.astype(np.float32)).cuda()
    #     self.lm_cls_ptsxyz_cuda_dict[cls] = ptsxyz_cu
    #     return ptsxyz_cu.clone()
def VOCap(rec, prec):
    idx = np.where(rec != np.inf)
    if len(idx[0]) == 0:
        return 0
    rec = rec[idx]
    prec = prec[idx]
    mrec = np.array([0.0]+list(rec)+[0.1])
    mpre = np.array([0.0]+list(prec)+[prec[-1]])
    for i in range(1, prec.shape[0]):
        mpre[i] = max(mpre[i], mpre[i-1])
    i = np.where(mrec[1:] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[i] - mrec[i-1]) * mpre[i]) * 10
    return ap

def cal_auc(add_dis, max_dis=0.1):
        D = np.array(add_dis)
        D[np.where(D > max_dis)] = np.inf;
        D = np.sort(D)
        n = len(add_dis)
        acc = np.cumsum(np.ones((1,n)), dtype=np.float32) / n
        aps = VOCap(D, acc)
        return aps * 100

def cal_add_cuda(pred_RT, gt_RT, p3ds
):
    pred_p3ds = torch.mm(p3ds, pred_RT[:, :3].transpose(1, 0)) + pred_RT[:, 3]
    gt_p3ds = torch.mm(p3ds, gt_RT[:, :3].transpose(1, 0)) + gt_RT[:, 3]
    dis = torch.norm(pred_p3ds - gt_p3ds, dim=1)
    return torch.mean(dis)

def cal_adds_cuda(
    pred_RT, gt_RT, p3ds
):
    N, _ = p3ds.size()
    pd = torch.mm(p3ds, pred_RT[:, :3].transpose(1, 0)) + pred_RT[:, 3]
    pd = pd.view(1, N, 3).repeat(N, 1, 1)
    gt = torch.mm(p3ds, gt_RT[:, :3].transpose(1, 0)) + gt_RT[:, 3]
    gt = gt.view(N, 1, 3).repeat(1, N, 1)
    dis = torch.norm(pd - gt, dim=2)
    mdis = torch.min(dis, dim=1)[0]
    return torch.mean(mdis)

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
        A: Nxm numpy array of corresponding points, usually points on mdl
        B: Nxm numpy array of corresponding points, usually points on camera axis
    Returns:
    T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
    R: mxm rotation matrix
    t: mx1 translation vector
    '''

    assert A.shape == B.shape
    # get number of dimensions
    m = A.shape[1]
    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    # rotation matirx
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)
    T = np.zeros((3, 4))
    T[:, :3] = R
    T[:, 3] = t
    return T


# ###############################YCB Evaluation###############################



def eval_metric(
    cls_ids, pred_pose_lst, pred_cls_ids, RTs
):
    n_cls = 22
    cls_add_dis = [list() for i in range(n_cls)]
    cls_adds_dis = [list() for i in range(n_cls)]
    for icls, cls_id in enumerate(cls_ids):
        if cls_id == 0:
            break


        cls_idx = np.where(pred_cls_ids == cls_id[0].item())[0]
        if len(cls_idx) == 0:
            pred_RT = torch.zeros(3, 4).cuda()
           
        else:
            pred_RT = pred_pose_lst[cls_idx[0]]
            pred_RT = torch.from_numpy(pred_RT.astype(np.float32)).cuda()
        
        gt_RT = RTs[icls]
        mesh_pts = get_pointxyz_cuda(cls_id - 1).clone()
        add = cal_add_cuda(pred_RT, gt_RT, mesh_pts)
        adds = cal_adds_cuda(pred_RT, gt_RT, mesh_pts)
        cls_add_dis[cls_id].append(add.item())
        cls_adds_dis[cls_id].append(adds.item())
        cls_add_dis[0].append(add.item())
        cls_adds_dis[0].append(adds.item())

    return (cls_add_dis, cls_adds_dis)


def eval_one_frame_pose(
    item
):
    cls_ids, pred_pose_lst, pred_cls_ids, RTs = item

    cls_add_dis, cls_adds_dis = eval_metric(
        cls_ids, pred_pose_lst, pred_cls_ids, RTs  )
    return (cls_add_dis, cls_adds_dis)

# ###############################End YCB Evaluation###############################


# ###############################LineMOD Evaluation###############################

def cal_frame_poses_lm(
    pcld, mask, ctr_of, pred_kp_of, use_ctr, n_cls, use_ctr_clus_flter, obj_id,
    debug=False
):
    """
    Calculates pose parameters by 3D keypoints & center points voting to build
    the 3D-3D corresponding then use least-squares fitting to get the pose parameters.
    """
    n_kps, n_pts, _ = pred_kp_of.size()
    pred_ctr = pcld - ctr_of[0]
    pred_kp = pcld.view(1, n_pts, 3).repeat(n_kps, 1, 1) - pred_kp_of

    radius = 0.04
    if use_ctr:
        cls_kps = torch.zeros(n_cls, n_kps+1, 3).cuda()
    else:
        cls_kps = torch.zeros(n_cls, n_kps, 3).cuda()

    pred_pose_lst = []
    cls_id = 1
    cls_msk = mask == cls_id
    if cls_msk.sum() < 1:
        pred_pose_lst.append(np.identity(4)[:3, :])
    else:
        cls_voted_kps = pred_kp[:, cls_msk, :]
        ms = MeanShiftTorch(bandwidth=radius)
        ctr, ctr_labels = ms.fit(pred_ctr[cls_msk, :])
        if ctr_labels.sum() < 1:
            ctr_labels[0] = 1
        if use_ctr:
            cls_kps[cls_id, n_kps, :] = ctr

        if use_ctr_clus_flter:
            in_pred_kp = cls_voted_kps[:, ctr_labels, :]
        else:
            in_pred_kp = cls_voted_kps

        for ikp, kps3d in enumerate(in_pred_kp):
            cls_kps[cls_id, ikp, :], _ = ms.fit(kps3d)

        # visualize
        if debug:
            show_kp_img = np.zeros((480, 640, 3), np.uint8)
            kp_2ds = bs_utils.project_p3d(
                cls_kps[cls_id].cpu().numpy(), 1000.0, K='linemod'
            )
            # print("cls_id = ", cls_id)
            # print("kp3d:", cls_kps[cls_id])
            # print("kp2d:", kp_2ds, "\n")
            color = (0, 0, 255)  # bs_utils.get_label_color(cls_id.item())
            show_kp_img = bs_utils.draw_p2ds(show_kp_img, kp_2ds, r=3, color=color)
            imshow("kp: cls_id=%d" % cls_id, show_kp_img)
            waitKey(0)

        mesh_kps = bs_utils_lm.get_kps(obj_id, ds_type="linemod")
        if use_ctr:
            mesh_ctr = bs_utils_lm.get_ctr(obj_id, ds_type="linemod").reshape(1, 3)
            mesh_kps = np.concatenate((mesh_kps, mesh_ctr), axis=0)
        # mesh_kps = torch.from_numpy(mesh_kps.astype(np.float32)).cuda()
        pred_RT = best_fit_transform(
            mesh_kps,
            cls_kps[cls_id].squeeze().contiguous().cpu().numpy()
        )
        pred_pose_lst.append(pred_RT)
    return pred_pose_lst


def eval_metric_lm(cls_ids, pred_pose_lst, RTs, mask, label, obj_id):
    n_cls = config.n_classes
    cls_add_dis = [list() for i in range(n_cls)]
    cls_adds_dis = [list() for i in range(n_cls)]

    pred_RT = pred_pose_lst[0]
    pred_RT = torch.from_numpy(pred_RT.astype(np.float32)).cuda()
    gt_RT = RTs[0]
    mesh_pts = bs_utils_lm.get_pointxyz_cuda(obj_id, ds_type="linemod").clone()
    add = bs_utils_lm.cal_add_cuda(pred_RT, gt_RT, mesh_pts)
    adds = bs_utils_lm.cal_adds_cuda(pred_RT, gt_RT, mesh_pts)
    # print("obj_id:", obj_id, add, adds)
    cls_add_dis[obj_id].append(add.item())
    cls_adds_dis[obj_id].append(adds.item())
    cls_add_dis[0].append(add.item())
    cls_adds_dis[0].append(adds.item())

    return (cls_add_dis, cls_adds_dis)


def eval_one_frame_pose_lm(
    item
):
    pcld, mask, ctr_of, pred_kp_of, RTs, cls_ids, use_ctr, n_cls, \
        min_cnt, use_ctr_clus_flter, label, epoch, ibs, obj_id = item
    pred_pose_lst = cal_frame_poses_lm(
        pcld, mask, ctr_of, pred_kp_of, use_ctr, n_cls, use_ctr_clus_flter,
        obj_id
    )

    cls_add_dis, cls_adds_dis = eval_metric_lm(
        cls_ids, pred_pose_lst, RTs, mask, label, obj_id
    )
    return (cls_add_dis, cls_adds_dis)

# ###############################End LineMOD Evaluation###############################


# ###############################Shared Evaluation Entry###############################
class TorchEval():

    def __init__(self):
        n_cls = 22
        self.n_cls = 22
        self.cls_add_dis = [list() for i in range(n_cls)]
        self.cls_adds_dis = [list() for i in range(n_cls)]
        self.cls_add_s_dis = [list() for i in range(n_cls)]
        self.pred_kp_errs = [list() for i in range(n_cls)]
        self.pred_id2pose_lst = []
        self.sym_cls_ids = []
       
        

    def cal_auc(self):
        add_auc_lst = []
        adds_auc_lst = []
        add_s_auc_lst = []
        for cls_id in range(1, self.n_cls):
            if (cls_id) in [13, 16, 19, 20, 21]:
                self.cls_add_s_dis[cls_id] = self.cls_adds_dis[cls_id]
            else:
                self.cls_add_s_dis[cls_id] = self.cls_add_dis[cls_id]
            self.cls_add_s_dis[0] += self.cls_add_s_dis[cls_id]
        for i in range(self.n_cls):
            add_auc = cal_auc(self.cls_add_dis[i])
            adds_auc = cal_auc(self.cls_adds_dis[i])
            add_s_auc = cal_auc(self.cls_add_s_dis[i])
            add_auc_lst.append(add_auc)
            adds_auc_lst.append(adds_auc)
            add_s_auc_lst.append(add_s_auc)
            if i == 0:
                continue
            print(i - 1)
            print("***************add:\t", add_auc)
            print("***************adds:\t", adds_auc)
            print("***************add(-s):\t", add_s_auc)
        # kp errs:
        n_objs = sum([len(l) for l in self.pred_kp_errs])
        all_errs = 0.0
        for cls_id in range(1, self.n_cls):
            all_errs += sum(self.pred_kp_errs[cls_id])
        print("mean kps errs:", all_errs / n_objs)

        print("Average of all object:")
        print("***************add:\t", np.mean(add_auc_lst[1:]))
        print("***************adds:\t", np.mean(adds_auc_lst[1:]))
        print("***************add(-s):\t", np.mean(add_s_auc_lst[1:]))

        print("All object (following PoseCNN):")
        print("***************add:\t", add_auc_lst[0])
        print("***************adds:\t", adds_auc_lst[0])
        print("***************add(-s):\t", add_s_auc_lst[0])


    # def cal_lm_add(self, obj_id, test_occ=False):
    #     add_auc_lst = []
    #     adds_auc_lst = []
    #     add_s_auc_lst = []
    #     cls_id = obj_id
    #     if (obj_id) in config_lm.lm_sym_cls_ids:
    #         self.cls_add_s_dis[cls_id] = self.cls_adds_dis[cls_id]
    #     else:
    #         self.cls_add_s_dis[cls_id] = self.cls_add_dis[cls_id]
    #     self.cls_add_s_dis[0] += self.cls_add_s_dis[cls_id]
    #     add_auc = bs_utils_lm.cal_auc(self.cls_add_dis[cls_id])
    #     adds_auc = bs_utils_lm.cal_auc(self.cls_adds_dis[cls_id])
    #     add_s_auc = bs_utils_lm.cal_auc(self.cls_add_s_dis[cls_id])
    #     add_auc_lst.append(add_auc)
    #     adds_auc_lst.append(adds_auc)
    #     add_s_auc_lst.append(add_s_auc)
    #     d = config_lm.lm_r_lst[obj_id]['diameter'] / 1000.0 * 0.1
    #     print("obj_id: ", obj_id, "0.1 diameter: ", d)
    #     add = np.mean(np.array(self.cls_add_dis[cls_id]) < d) * 100
    #     adds = np.mean(np.array(self.cls_adds_dis[cls_id]) < d) * 100

    #     cls_type = config_lm.lm_id2obj_dict[obj_id]
    #     print(obj_id, cls_type)
    #     print("***************add auc:\t", add_auc)
    #     print("***************adds auc:\t", adds_auc)
    #     print("***************add(-s) auc:\t", add_s_auc)
    #     print("***************add < 0.1 diameter:\t", add)
    #     print("***************adds < 0.1 diameter:\t", adds)

    #     sv_info = dict(
    #         add_dis_lst=self.cls_add_dis,
    #         adds_dis_lst=self.cls_adds_dis,
    #         add_auc_lst=add_auc_lst,
    #         adds_auc_lst=adds_auc_lst,
    #         add_s_auc_lst=add_s_auc_lst,
    #         add=add,
    #         adds=adds,
    #     )
    #     occ = "occlusion" if test_occ else ""
    #     sv_pth = os.path.join(
    #         config_lm.log_eval_dir,
    #         'pvn3d_eval_cuda_{}_{}_{}_{}.pkl'.format(
    #             cls_type, occ, add, adds
    #         )
    #     )
    #     pkl.dump(sv_info, open(sv_pth, 'wb'))

    def eval_pose_parallel(
        self,cls_ids, pred_pose_lst, pred_cls_ids,RTs, ds='ycb'
    ):
        
        cls_ids = cls_ids.long()
       
        
    
        if ds == "ycb":
            data_gen = zip(
                cls_ids, pred_pose_lst, pred_cls_ids, RTs
            )
        # else:
        #     data_gen = zip(
        #         pclds, masks, pred_ctr_ofs, pred_kp_ofs, RTs,
        #         cls_ids, use_ctr_lst, n_cls_lst, min_cnt_lst, use_ctr_clus_flter_lst,
        #         labels, epoch_lst, bs_lst, obj_id_lst
        #     )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=4
        ) as executor:
            if ds == "ycb":
                eval_func = eval_one_frame_pose
            # else:
            #     eval_func = eval_one_frame_pose_lm
            for res in executor.map(eval_func, data_gen):
                if ds == 'ycb':
                    cls_add_dis_lst, cls_adds_dis_lst = res
                else:
                    cls_add_dis_lst, cls_adds_dis_lst = res
                self.cls_add_dis = self.merge_lst(
                    self.cls_add_dis, cls_add_dis_lst
                )
                self.cls_adds_dis = self.merge_lst(
                    self.cls_adds_dis, cls_adds_dis_lst
                )

    def merge_lst(self, targ, src):
        for i in range(len(targ)):
            targ[i] += src[i]
        return targ

# vim: ts=4 sw=4 sts=4 expandtab
