# FCOS: Fully Convolutional One-Stage Object Detector

This repository contains a PyTorch implementation of **FCOS** (Fully Convolutional One-Stage Object Detection).  
The project is built from scratch for educational purposes, following the original paper:  
👉 *FCOS: Fully Convolutional One-Stage Object Detection* ([Tian et al., 2019](https://arxiv.org/abs/1904.01355)).

---

## 🚀 Features
- **Backbone + FPN**
  - Uses a RegNet backbone with an FPN (Feature Pyramid Network).
  - Outputs multi-scale features at strides 8, 16, and 32 (`p3`, `p4`, `p5`).
- **Prediction Head**
  - Separate convolutional stems for classification and regression.
  - Predicts:
    - **Class logits**
    - **Box regression deltas (LTRB)**
    - **Centerness score**
- **Training**
  - Location-to-GT assignment based on FCOS rules.
  - Focal loss for classification, L1 loss for box regression, BCE loss for centerness.
  - Online loss normalization with EMA foreground counter.
- **Inference**
  - Box decoding with predicted deltas + centerness weighting.
  - Class-specific **NMS (Non-Maximum Suppression)**.
  - Final output: bounding boxes, predicted classes, and confidence scores.

---

## 📂 Project Structure
```
fcos.py              # Main implementation
cs639/loading.py     # Dataset loading helpers (course-specific)
notebooks/           # Jupyter notebooks for experiments
```

Key components inside `fcos.py`:
- `DetectorBackboneWithFPN`: RegNet backbone + FPN
- `FCOSPredictionNetwork`: Multi-branch prediction head
- `FCOS`: End-to-end detection model (training + inference)
- Utility functions:
  - `nms`, `class_spec_nms`
  - `fcos_match_locations_to_gt`
  - `fcos_get_deltas_from_locations`
  - `fcos_apply_deltas_to_locations`
  - `fcos_make_centerness_targets`

---

## 🛠 Installation
```bash
git clone https://github.com/yourusername/fcos-pytorch.git
cd fcos-pytorch
pip install -r requirements.txt
```

Dependencies:
- `torch >= 1.12`
- `torchvision`
- `numpy`
- `matplotlib`

---

## 📖 Usage

### 1. Quick sanity check
```python
from fcos import FCOS
import torch

model = FCOS(num_classes=20, fpn_channels=64, stem_channels=[64,64,64,64])
dummy_input = torch.randn(1, 3, 224, 224)

model.train()
outputs = model(dummy_input, gt_boxes=torch.randn(1, 5, 5))  # fake GT boxes
print(outputs)  # training losses
```

### 2. Inference
```python
model.eval()
with torch.no_grad():
    pred_boxes, pred_classes, pred_scores = model(
        dummy_input,
        test_score_thresh=0.3,
        test_nms_thresh=0.5
    )

print(pred_boxes.shape, pred_classes.shape, pred_scores.shape)
```

---

## 🏋️ Training & Evaluation

You can run experiments with the provided **notebooks** or by writing a simple training loop.

### Training Example
```python
from fcos import FCOS
import torch
import torch.optim as optim

# Initialize model
model = FCOS(num_classes=20, fpn_channels=64, stem_channels=[64,64,64,64])
model = model.cuda()

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

# Dummy dataloader (replace with real dataset)
for epoch in range(2):
    for images, gt_boxes in dataloader:   # images: (B,3,H,W), gt_boxes: (B,N,5)
        images, gt_boxes = images.cuda(), gt_boxes.cuda()
        loss_dict = model(images, gt_boxes=gt_boxes)

        loss = loss_dict["loss_cls"] + loss_dict["loss_box"] + loss_dict["loss_ctr"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: loss = {loss.item():.4f}")
```

### Evaluation Example
```python
model.eval()
with torch.no_grad():
    for images, _ in dataloader_val:
        images = images.cuda()
        pred_boxes, pred_classes, pred_scores = model(
            images,
            test_score_thresh=0.3,
            test_nms_thresh=0.5
        )
        # TODO: compute mAP or visualize results
```

---

## 📊 Example Results
- Loss curves stabilize after ~20k iterations on COCO-mini.
- Model outputs reasonable bounding boxes after a few epochs.

*(Add screenshots or sample output visualizations here if available)*

---

## 🔍 References
- Tian et al., *FCOS: Fully Convolutional One-Stage Object Detection*, ICCV 2019.  
- [torchvision.ops](https://pytorch.org/vision/stable/ops.html) for NMS reference.

---

## 📜 License
This repository is released under the MIT License.
