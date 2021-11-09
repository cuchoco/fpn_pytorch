import numpy as np
import torch

def compute_iou(box, boxes, box_area, boxes_area):

    # Calculate intersection areas
    x1 = np.maximum(box[0], boxes[:, 0])
    x2 = np.minimum(box[2], boxes[:, 2])
    y1 = np.maximum(box[1], boxes[:, 1])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def non_max_suppression(boxes, scores, threshold):
    assert boxes.shape[0] > 0
    
    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()

    # Compute box areas
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (y2 - y1) * (x2 - x1)

    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort(axis=0)[::-1].squeeze()

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indices into ixs[1:], so add 1 to get
        # indices into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indices of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
        
    return torch.FloatTensor(pick)


def test():
    # Let P be the following
    P = torch.tensor([
        [1, 1, 3, 3, 0.95],
        [1, 1, 3, 4, 0.93],
        [1, 0.9, 3.6, 3, 0.98],
        [1, 0.9, 3.5, 3, 0.97]
    ])
    bbox = P[:,:4]
    scores = P[:,4]
    return nms_pytorch(bbox, scores, 0.5)

if __name__=='__main__':
    test = test()
    print(test)

    
