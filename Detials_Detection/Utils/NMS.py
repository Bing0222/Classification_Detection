
import torch

"""
Implementation NMS

different box  according to overloop part select the best scores
"""
def nms(self,bboxes,scores,thresh=0.5):
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    # get areas of boxes
    areas = (x2-x1+1)*(y2-y1+1)
    _,order = scores.sort(0,descending=True)
    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()
            keep.append(i)

        # using tensor.clamp get each box max and min
        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(min=x2[i])
        yy2 = y2[order[1:]].clamp(min=y2[i])

        # get overloop part
        inter = (xx2-xx1).clamp(min=0) * (yy2-yy1).clamp(min=0)
        iou = inter/(areas[i]+areas[order[1:]]-inter)
        idx = (iou <= thresh).nonzero().squeeze()
        if idx.numel() == 0:
            break
        order = order[idx+1]

    return torch.LongTensor(keep)
