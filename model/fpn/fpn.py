import datetime
import math
import os
import random
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

from torchvision.ops.roi_pool import RoIPool as _RoIPooling
from torchvision.ops.roi_align import RoIAlign as RoIAlignAvg


from model.fpn.resnet import resnet101
from model.nms.nms import non_max_suppression as nms
from model.rpn.rpn_fpn import _RPN_FPN, _smooth_l1_loss
from model.rpn.proposal_target_layer import _ProposalTargetLayer



class SamePad2d(nn.Module):
    """Mimics tensorflow's 'SAME' padding.
    """

    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = ((out_width - 1) * self.stride[0] +
                           self.kernel_size[0] - in_width)
        pad_along_height = ((out_height - 1) * self.stride[1] +
                            self.kernel_size[1] - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

    def __repr__(self):
        return self.__class__.__name__
    
    
    
    
class TopDownLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super.__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same')
        
    def forward(self, x, y):
        y = F.upsample(y, scale_factor=2)
        x = self.conv1(x)
        x = self.conv2(x+y)
        
        
        
class FPN(nn.Module):
    def __init__(self, C1, C2, C3, C4, C5, out_channels=256):
        super().__init__()
        self.out_channels = out_channels
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.C4 = C4
        self.C5 = C5
        
        self.P6 = nn.MaxPool2d(kernel_size=1, stride=2)
        
        
        self.C5_conv1 = nn.Conv2d(2048, self.out_channels, kernel_size=1, stride=1)
        self.P5_conv3 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        
        
        self.C4_conv1 =  nn.Conv2d(1024, self.out_channels, kernel_size=1, stride=1)
        self.P4_conv3 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        self.C3_conv1 = nn.Conv2d(512, self.out_channels, kernel_size=1, stride=1)
        self.P3_conv3 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        self.C2_conv1 = nn.Conv2d(256, self.out_channels, kernel_size=1, stride=1)
        self.P2_conv3 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )

    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)
        c2_out = x
        x = self.C3(x)
        c3_out = x
        x = self.C4(x)
        c4_out = x
        x = self.C5(x) # c5_outy
        
        m5 = self.C5_conv1(x)
        m4 = self.C4_conv1(c4_out) + F.upsample(m5, scale_factor=2)
        m3 = self.C3_conv1(c3_out) + F.upsample(m4, scale_factor=2)
        m2 = self.C2_conv1(c2_out) + F.upsample(m3, scale_factor=2)

        p5_out = self.P5_conv3(m5)
        p4_out = self.P4_conv3(m4)
        p3_out = self.P3_conv3(m3)
        p2_out = self.P2_conv3(m2)

        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        p6_out = self.P6(p5_out)

        return [p2_out, p3_out, p4_out, p5_out, p6_out]
    
    
    




class _FPN(nn.Module):
    """ FPN """
    def __init__(self, classes, class_agnostic = False):
        super().__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        self.maxpool2d = nn.MaxPool2d(1, stride=2)
        # define rpn
        self.RCNN_rpn = _RPN_FPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        # NOTE: the original paper used pool_size = 7 for cls branch, and 14 for mask branch, to save the
        # computation time, we first use 14 as the pool_size, and then do stride=2 pooling for cls branch.
        self.RCNN_roi_pool = _RoIPooling(7,  1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(7,  1.0/16.0)
        self.grid_size = 7 * 2 if True else 7
        self.RCNN_roi_crop = _RoICrop()
        
        self.resnet = resnet101()
        
    def _get_fpn_features(self, x, model_path):
        resnet = self.resnet
        in_features = resnet.fc.in_features
        resnet.fc = nn.Linear(in_features=in_features, out_features=2)
        param = torch.load(model_path)
        resnet.load_state_dict(param)
        resnet.eval()
        c1,c2,c3,c4,c5 = resnet.stages()
        
        fpn = FPN(c1,c2,c3,c4,c5)
        p2, p3, p4, p5, p6 = fpn(x)
        
        rpn_feature_maps = [p2, p3, p4, p5, p6]
        mrcnn_feature_maps = [p2, p3, p4, p5]
        return rpn_feature_maps, mrcnn_feature_maps
    
        
    def _PyramidRoI_Feat(self, feat_maps, rois, im_info):
        POOLING_MODE = 'pool'
        
        ''' roi pool on pyramid feature maps'''
        # do roi pooling based on predicted rois
        img_area = im_info[0][0] * im_info[0][1]
        h = rois.data[:, 4] - rois.data[:, 2] + 1
        w = rois.data[:, 3] - rois.data[:, 1] + 1
        roi_level = torch.log(torch.sqrt(h * w) / 224.0)
        roi_level = torch.round(roi_level + 4)
        roi_level[roi_level < 2] = 2
        roi_level[roi_level > 5] = 5
        # roi_level.fill_(5)
        if POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            # NOTE: need to add pyrmaid
            grid_xy = _affine_grid_gen(rois, base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            roi_pool_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if True:
                roi_pool_feat = F.max_pool2d(roi_pool_feat, 2, 2)

        elif POOLING_MODE == 'align':
            roi_pool_feats = []
            box_to_levels = []
            for i, l in enumerate(range(2, 6)):
                if (roi_level == l).sum() == 0:
                    continue
                idx_l = (roi_level == l).nonzero().squeeze()
                box_to_levels.append(idx_l)
                scale = feat_maps[i].size(2) / im_info[0][0]
                feat = self.RCNN_roi_align(feat_maps[i], rois[idx_l], scale)
                roi_pool_feats.append(feat)
            roi_pool_feat = torch.cat(roi_pool_feats, 0)
            box_to_level = torch.cat(box_to_levels, 0)
            idx_sorted, order = torch.sort(box_to_level)
            roi_pool_feat = roi_pool_feat[order]

        elif POOLING_MODE == 'pool':
            roi_pool_feats = []
            box_to_levels = []
            for i, l in enumerate(range(2, 6)):
                if (roi_level == l).sum() == 0:
                    continue
                idx_l = (roi_level == l).nonzero().squeeze()
                box_to_levels.append(idx_l)
                scale = feat_maps[i].size(2) / im_info[0][0]
                feat = self.RCNN_roi_pool(feat_maps[i], rois[idx_l], scale)
                roi_pool_feats.append(feat)
            roi_pool_feat = torch.cat(roi_pool_feats, 0)
            box_to_level = torch.cat(box_to_levels, 0)
            idx_sorted, order = torch.sort(box_to_level)
            roi_pool_feat = roi_pool_feat[order]
            
        return roi_pool_feat
    
    def _head_to_tail(self, pool5):
        block5 = self.RCNN_top(pool5)
        fc7 = block5.mean(3).mean(2)
        return fc7
    
    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        rpn_feature_maps, mrcnn_feature_maps = _get_fpn_features(im_info, '/home/kuchoco97/work/brain/data/backbone/model/epoch20_valloss0.33.pth')
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(rpn_feature_maps, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, gt_assign, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            ## NOTE: additionally, normalize proposals to range [0, 1],
            #        this is necessary so that the following roi pooling
            #        is correct on different feature maps
            # rois[:, :, 1::2] /= im_info[0][1]
            # rois[:, :, 2::2] /= im_info[0][0]

            rois = rois.view(-1, 5)
            rois_label = rois_label.view(-1).long()
            gt_assign = gt_assign.view(-1).long()
            pos_id = rois_label.nonzero().squeeze()
            gt_assign_pos = gt_assign[pos_id]
            rois_label_pos = rois_label[pos_id]
            rois_label_pos_ids = pos_id

            rois_pos = Variable(rois[pos_id])
            rois = Variable(rois)
            rois_label = Variable(rois_label)

            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            ## NOTE: additionally, normalize proposals to range [0, 1],
            #        this is necessary so that the following roi pooling
            #        is correct on different feature maps
            # rois[:, :, 1::2] /= im_info[0][1]
            # rois[:, :, 2::2] /= im_info[0][0]

            rois_label = None
            gt_assign = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0
            rois = rois.view(-1, 5)
            pos_id = torch.arange(0, rois.size(0)).long().type_as(rois).long()
            rois_label_pos_ids = pos_id
            rois_pos = Variable(rois[pos_id])
            rois = Variable(rois)

        # pooling features based on rois, output 14x14 map
        roi_pool_feat = self._PyramidRoI_Feat(mrcnn_feature_maps, rois, im_info)

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(roi_pool_feat)


        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.long().view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # loss (cross entropy) for object classification
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            # loss (l1-norm) for bounding box regression
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        rois = rois.view(batch_size, -1, rois.size(1))
        cls_prob = cls_prob.view(batch_size, -1, cls_prob.size(1))
        bbox_pred = bbox_pred.view(batch_size, -1, bbox_pred.size(1))

        if self.training:
            rois_label = rois_label.view(batch_size, -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label