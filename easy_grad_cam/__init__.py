import contextlib

import numpy as np
import cv2
import torch
import torch.nn.functional as F


@contextlib.contextmanager
def grad_cam(module, size=None):
    """
    Args:
        module: pytorch conv layer.
        size: the size that the final cam will be resized to.
    """
    # forward/backward list
    fl, bl = [], []

    # forward/backward hook handle
    fh = module.register_forward_hook(lambda _, i, o: fl.append(o))
    bh = module.register_backward_hook(lambda _, i, o: bl.append(o[0]))

    def compute():
        assert fl and bl, 'please run backward first'
        output, gradient = fl[-1], bl[-1]
        alpha = gradient.flatten(-2, -1).mean(-1)
        cam = torch.einsum('ij,ij...->i...', alpha, output)
        cam = cam[:, None]
        cam = torch.relu(cam)
        if size is not None:
            cam = F.interpolate(cam, size,
                                mode='bilinear',
                                align_corners=True)
        high = cam.flatten(-2).max(dim=-1)[0]
        cam /= high[..., None, None]
        return cam

    try:
        yield compute
    finally:
        fh.remove()
        bh.remove()


def tensor_to_numpy(image):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        if image.dim() == 3:
            image = image.permute(1, 2, 0)
        image = image.numpy()
    return image


def blend(image, cam):
    image = tensor_to_numpy(image)
    cam = tensor_to_numpy(cam)
    image = np.uint8(image * 255)
    cam = np.uint8(cam * 255)
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_RAINBOW)
    image = cv2.addWeighted(image, 0.6, cam, 0.4, 0)
    return image
