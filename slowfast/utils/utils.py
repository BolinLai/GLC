import torch.nn.functional as F


def frame_softmax(logits, temperature):
    batch_size, T = logits.shape[0], logits.shape[2]
    H, W = logits.shape[3], logits.shape[4]
    # reshape -> softmax (dim=-1) -> reshape back
    logits = logits.view(batch_size, -1, T, H * W)
    atten_map = F.softmax(logits / temperature, dim=-1)
    atten_map = atten_map.view(batch_size, -1, T, H, W)
    return atten_map