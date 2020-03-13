# Easy Grad-CAM: A simple Grad-CAM tool for PyTorch

## Install

```
pip install .
```

## Example

```python
import torch
from torchvision import models
import matplotlib.pyplot as plt

from easy_grad_cam import grad_cam, blend

model = models.resnet18(True)
model.eval()

image = plt.imread('cat_dog.png')
image = torch.tensor(image.transpose(2, 0, 1))

# 1. select the conv layer to visualize
conv = list(model.children())[-3]

# 2. create a with statement as this
with grad_cam(conv, size=image.shape[-2:]) as compute:
    # 3. forward
    logits = model(image[None])
    # 4. backward
    model.zero_grad()
    loss = logits.log_softmax(-1).max(-1)[0]
    loss.backward()
    # 5. get the cam!
    cam = compute()[0]

# 6. blend the cam with the original image
blended = blend(image, cam)

# 7. plot it
plt.imshow(blended)
plt.show()
```

