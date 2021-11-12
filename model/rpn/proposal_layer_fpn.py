import torch
import torch.nn as nn
import numpy as np

from model.rpn.generate_anchors import generate_anchors, generate_anchors_all_pyramids
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from model.nms.nms import non_max_suppression as nms


class _ProposalLayer_FPN(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self, feat_stride, scales, ratios):
        super(_ProposalLayer_FPN, self).__init__()
        self._anchor_ratios = ratios
        self._feat_stride = feat_stride
        self._fpn_scales = np.array([8, 16, 32, 64, 128])
        self._fpn_feature_strides = np.array([4, 8, 16, 32, 64])
        self._fpn_anchor_stride = 1


    def forward(self, input):

        scores = input[0][:, :, 1]  # batch_size x num_rois x 1
        bbox_deltas = input[1]      # batch_size x num_rois x 4
        im_info = input[2]
        cfg_key = input[3]
        feat_shapes = input[4]        

        pre_nms_topN  = 12000
        post_nms_topN = 2000
        nms_thresh    = 0.7
        min_size      = 8
        batch_size = bbox_deltas.size(0)

        anchors = torch.from_numpy(generate_anchors_all_pyramids(self._fpn_scales, self._anchor_ratios, 
                feat_shapes, self._fpn_feature_strides, self._fpn_anchor_stride)).type_as(scores)
        num_anchors = anchors.size(0)

        anchors = anchors.view(1, num_anchors, 4).expand(batch_size, num_anchors, 4)
     
        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas, batch_size)
     
        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info, batch_size)
        # keep_idx = self._filter_boxes(proposals, min_size).squeeze().long().nonzero().squeeze()
                
        scores_keep = scores
        proposals_keep = proposals

        _, order = torch.sort(scores_keep, 1, True)

        output = scores.new(batch_size, post_nms_topN, 5).zero_()
        for i in range(batch_size):
            # # 3. remove predicted boxes with either height or width < threshold
            # # (NOTE: convert min_size to input image scale stored in im_info[2])
            proposals_single = proposals_keep[i]
            scores_single = scores_keep[i]

            # # 4. sort all (proposal, score) pairs by score from highest to lowest
            # # 5. take top pre_nms_topN (e.g. 6000)
            order_single = order[i]

            if pre_nms_topN > 0 and pre_nms_topN < scores_keep.numel():
                order_single = order_single[:pre_nms_topN]

            proposals_single = proposals_single[order_single, :]
            scores_single = scores_single[order_single].view(-1,1)

            # 6. apply nms (e.g. threshold = 0.7)
            # 7. take after_nms_topN (e.g. 300)
            # 8. return the top proposals (-> RoIs top)

            # keep_idx_i = nms(torch.cat((proposals_single, scores_single), 1), nms_thresh)
            keep_idx_i = nms(proposals_single, scores_single ,nms_thresh)
            keep_idx_i = keep_idx_i.long().view(-1)

            if post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:post_nms_topN]
            proposals_single = proposals_single[keep_idx_i, :]
            scores_single = scores_single[keep_idx_i, :]

            # padding 0 at the end.
            num_proposal = proposals_single.size(0)
            output[i,:,0] = i
            output[i,:num_proposal,1:] = proposals_single

        return output

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _filter_boxes(self, boxes, min_size):
        """Remove all boxes with any side smaller than min_size."""
        ws = boxes[:, :, 2] - boxes[:, :, 0] + 1
        hs = boxes[:, :, 3] - boxes[:, :, 1] + 1
        keep = ((ws >= min_size) & (hs >= min_size))
        return keep