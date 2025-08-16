import torch
import torch.nn.functional as F

@torch.no_grad()
def predict_proba(model, x):  # x: Bx3xHxW
    model.eval()
    logits = model(x)
    return F.softmax(logits, dim=1)

def topk_from_proba(proba, class_names, k=3):
    vals, idxs = proba.topk(k, dim=1)
    vals = vals[0].cpu().tolist()
    idxs = idxs[0].cpu().tolist()
    out = []
    for p, i in zip(vals, idxs):
        label = class_names[i] if (class_names and i < len(class_names)) else f"class_{i}"
        out.append((label, float(p)))
    return out
