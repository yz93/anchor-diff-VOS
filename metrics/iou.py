def get_iou(preds, labels, thresh=0.5):
    preds, labels = preds.squeeze(), labels.squeeze()
    preds = preds > thresh
    mask_sum = (preds == 1) + (labels > 0)
    intersection = (mask_sum == 2).sum().float()
    union = (mask_sum > 0).sum().float()

    if union > 0:
        return intersection / union

    return 1.
