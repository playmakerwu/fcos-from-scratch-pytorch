"""
This module contains classes and functions that are used for FCOS, a one-stage
object detector. You have to implement the functions here -
walk through the notebooks and you will find instructions on *when* to implement
*what* in this module.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
from cs639.loading import *
from torch import nn
from torch.nn import functional as F
from torch.utils.data._utils.collate import default_collate

from torchvision import models
from torchvision.models import feature_extraction
from torchvision.ops import sigmoid_focal_loss


def hello_fcos():
    print("Hello from fcos.py!")


class DetectorBackboneWithFPN(nn.Module):
    r"""
    Detection backbone network: A tiny RegNet model coupled with a Feature
    Pyramid Network (FPN). This model takes in batches of input images with
    shape `(B, 3, H, W)` and gives features from three different FPN levels
    with shapes and total strides upto that level:

        - level p3: (out_channels, H /  8, W /  8)      stride =  8
        - level p4: (out_channels, H / 16, W / 16)      stride = 16
        - level p5: (out_channels, H / 32, W / 32)      stride = 32

    NOTE: We could use any convolutional network architecture that progressively
    downsamples the input image and couple it with FPN. We use a small enough
    backbone that can work with Colab GPU and get decent enough performance.
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

        # Initialize with ImageNet pre-trained weights for faster convergence.
        _cnn = models.regnet_x_400mf(pretrained=True)

        # Wrap with a feature extractor to obtain intermediate features.
        self.backbone = feature_extraction.create_feature_extractor(
            _cnn,
            return_nodes={
                "trunk_output.block2": "c3",
                "trunk_output.block3": "c4",
                "trunk_output.block4": "c5",
            },
        )

        # Dry run to infer shapes.
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))
        dummy_out_shapes = [(key, value.shape) for key, value in dummy_out.items()]

        print("For dummy input images with shape: (2, 3, 224, 224)")
        for level_name, feature_shape in dummy_out_shapes:
            print(f"Shape of {level_name} features: {feature_shape}")

        ######################################################################
        # TODO: Initialize additional Conv layers for FPN.                   #
        ######################################################################
        self.fpn_params = nn.ModuleDict()

        # Read channels of c3/c4/c5 from the dummy pass
        c3_ch, c4_ch, c5_ch = (
            dummy_out["c3"].shape[1],
            dummy_out["c4"].shape[1],
            dummy_out["c5"].shape[1],
        )

        # Lateral 1x1 convs (unify channels to out_channels)
        self.fpn_params["lat_c3"] = nn.Conv2d(c3_ch, self.out_channels, kernel_size=1, bias=False)
        self.fpn_params["lat_c4"] = nn.Conv2d(c4_ch, self.out_channels, kernel_size=1, bias=False)
        self.fpn_params["lat_c5"] = nn.Conv2d(c5_ch, self.out_channels, kernel_size=1, bias=False)

        # Output 3x3 heads: p4 and p5 use single conv; p3 uses Conv-BN-ReLU-Conv
        self.fpn_params["o_p5"] = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1, bias=True)
        self.fpn_params["o_p4"] = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1, bias=True)

        self.fpn_params["o_p3_block1"] = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1, bias=False)
        self.fpn_params["o_p3_bn"] = nn.BatchNorm2d(self.out_channels)
        self.fpn_params["o_p3_relu"] = nn.ReLU(inplace=True)
        self.fpn_params["o_p3_block2"] = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1, bias=True)

        # Initialize weights
        for m in self.fpn_params.values():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    @property
    def fpn_strides(self):
        """
        Total stride up to the FPN level. For a fixed ConvNet, these values
        are invariant to input image size. You may access these values freely
        to implement your logic in FCOS / Faster R-CNN.
        """
        return {"p3": 8, "p4": 16, "p5": 32}

    def forward(self, images: torch.Tensor):
        # Multi-scale features from the backbone: dict with keys {"c3","c4","c5"}.
        backbone_feats = self.backbone(images)

        fpn_feats = {"p3": None, "p4": None, "p5": None}
        ######################################################################
        # TODO: Fill output FPN features (p3, p4, p5) using Conv features    #
        # (c3, c4, c5) and FPN conv layers created above.                    #
        # HINT: Use `F.interpolate` to upsample FPN features.                #
        ######################################################################
        # Lateral projections
        l3 = self.fpn_params["lat_c3"](backbone_feats["c3"])
        l4 = self.fpn_params["lat_c4"](backbone_feats["c4"])
        l5 = self.fpn_params["lat_c5"](backbone_feats["c5"])

        # Top-down pathway with nearest upsampling
        up5 = F.interpolate(l5, scale_factor=2.0, mode="nearest")
        m4 = l4 + up5
        up4 = F.interpolate(m4, scale_factor=2.0, mode="nearest")
        m3 = l3 + up4

        # Output heads
        p5 = self.fpn_params["o_p5"](l5)
        p4 = self.fpn_params["o_p4"](m4)

        p3_mid = self.fpn_params["o_p3_block1"](m3)
        p3_mid = self.fpn_params["o_p3_bn"](p3_mid)
        p3_mid = self.fpn_params["o_p3_relu"](p3_mid)
        p3 = self.fpn_params["o_p3_block2"](p3_mid)

        fpn_feats["p3"], fpn_feats["p4"], fpn_feats["p5"] = p3, p4, p5
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

        return fpn_feats


def get_fpn_location_coords(
    shape_per_fpn_level: Dict[str, Tuple],
    strides_per_fpn_level: Dict[str, int],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Map every location in FPN feature map to a point on the image. This point
    represents the center of the receptive field of this location. We need to
    do this for having a uniform co-ordinate representation of all the locations
    across FPN levels, and GT boxes.
    """

    location_coords = {level_name: None for level_name, _ in shape_per_fpn_level.items()}

    for level_name, feat_shape in shape_per_fpn_level.items():
        level_stride = strides_per_fpn_level[level_name]

        ######################################################################
        # TODO: Implement logic to get location co-ordinates below.          #
        ######################################################################
        _, _, H, W = feat_shape

        # Build a meshgrid in (y, x) order, then stack as (x, y)
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing="ij",
        )
        xc = (xx + 0.5) * level_stride
        yc = (yy + 0.5) * level_stride

        coords = torch.stack([xc.reshape(-1), yc.reshape(-1)], dim=1).to(dtype)
        location_coords[level_name] = coords
        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################
    return location_coords


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5):
    """
    Non-maximum suppression removes overlapping bounding boxes.

    Args:
        boxes: Tensor of shape (N, 4) giving top-left and bottom-right coordinates
            of the bounding boxes to perform NMS on.
        scores: Tensor of shape (N,) giving scores for each of the boxes.
        iou_threshold: Discard all overlapping boxes with IoU > iou_threshold

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """

    if (not boxes.numel()) or (not scores.numel()):
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    keep = None
    #############################################################################
    # TODO: Implement non-maximum suppression which iterates the following:     #
    #       1. Select the highest-scoring box among the remaining ones,         #
    #          which has not been chosen in this step before                    #
    #       2. Eliminate boxes with IoU > threshold                             #
    #       3. If any boxes remain, GOTO 1                                      #
    #       Your implementation should not depend on a specific device type;    #
    #       you can use the device of the input if necessary.                   #
    #############################################################################
    # Sort boxes by score descending and process greedily with vectorized IoU.
    order = torch.argsort(scores, descending=True)
    boxes_sorted = boxes[order]

    x1, y1, x2, y2 = boxes_sorted[:, 0], boxes_sorted[:, 1], boxes_sorted[:, 2], boxes_sorted[:, 3]
    areas = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)

    keep_local = []
    idxs = torch.arange(boxes_sorted.size(0), device=boxes.device, dtype=torch.long)

    while idxs.numel() > 0:
        i = idxs[0].item()
        keep_local.append(i)

        if idxs.numel() == 1:
            break

        rest = idxs[1:]

        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])

        w = (xx2 - xx1).clamp_min(0)
        h = (yy2 - yy1).clamp_min(0)
        inter = w * h
        union = areas[i] + areas[rest] - inter
        iou = inter / (union + 1e-12)

        remain = iou <= iou_threshold
        idxs = rest[remain]

    keep = order[torch.tensor(keep_local, device=boxes.device, dtype=torch.long)]
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return keep


def class_spec_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float = 0.5,
):
    """
    Wrap `nms` to make it class-specific. Pass class IDs as `class_ids`.
    STUDENT: This depends on your `nms` implementation.

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = class_ids.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


# Short hand type notation:
TensorDict = Dict[str, torch.Tensor]


class FCOSPredictionNetwork(nn.Module):
    """
    FCOS prediction network that accepts FPN feature maps from different levels
    and makes three predictions at every location: bounding boxes, class ID and
    centerness. This module contains a "stem" of convolution layers, along with
    one final layer per prediction. For a visual depiction, see Figure 2 (right
    side) in FCOS paper: https://arxiv.org/abs/1904.01355

    We will use feature maps from FPN levels (P3, P4, P5) and exclude (P6, P7).
    """

    def __init__(
        self, num_classes: int, in_channels: int, stem_channels: List[int]
    ):
        """
        Args:
            num_classes: Number of object classes for classification.
            in_channels: Number of channels in input feature maps. This value
                is same as the output channels of FPN, since the head directly
                operates on them.
            stem_channels: List of integers giving the number of output channels
                in each convolution layer of stem layers.
        """
        super().__init__()

        ######################################################################
        # TODO: Create a stem of alternating 3x3 convolution layers and RELU  #
        # activation modules. Two separate stems: one for classification and  #
        # one for box/centerness.                                            #
        ######################################################################
        def _make_stem(cin: int, channels: List[int]) -> nn.Sequential:
            layers: List[nn.Module] = []
            in_ch = cin
            for out_ch in channels:
                conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)
                nn.init.normal_(conv.weight, mean=0.0, std=0.01)
                nn.init.zeros_(conv.bias)
                layers.append(conv)
                layers.append(nn.ReLU(inplace=True))
                in_ch = out_ch
            return nn.Sequential(*layers)

        self.stem_cls = _make_stem(in_channels, stem_channels)
        self.stem_box = _make_stem(in_channels, stem_channels)
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        ######################################################################
        # TODO: Create THREE 3x3 conv layers for predicting:                 #
        # 1) class logits, 2) box deltas (LTRB), 3) centerness logits.       #
        ######################################################################
        out_ch = stem_channels[-1]

        self.pred_cls = nn.Conv2d(out_ch, num_classes, kernel_size=3, stride=1, padding=1, bias=True)
        nn.init.normal_(self.pred_cls.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.pred_cls.bias)

        self.pred_box = nn.Conv2d(out_ch, 4, kernel_size=3, stride=1, padding=1, bias=True)
        nn.init.normal_(self.pred_box.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.pred_box.bias)

        self.pred_ctr = nn.Conv2d(out_ch, 1, kernel_size=3, stride=1, padding=1, bias=True)
        nn.init.normal_(self.pred_ctr.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.pred_ctr.bias)
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        # Override: negative bias for class logits to stabilize training
        torch.nn.init.constant_(self.pred_cls.bias, -math.log(99))

    def forward(self, feats_per_fpn_level: TensorDict) -> List[TensorDict]:
        """
        Accept FPN feature maps and predict the desired outputs at every location
        (as described above). Format them such that channels are placed at the
        last dimension, and (H, W) are flattened.
        """
        ######################################################################
        # TODO: Iterate over every FPN feature map and obtain predictions.   #
        # Do not apply sigmoid here.                                         #
        ######################################################################
        class_logits: Dict[str, torch.Tensor] = {}
        boxreg_deltas: Dict[str, torch.Tensor] = {}
        centerness_logits: Dict[str, torch.Tensor] = {}

        for lvl, feat in feats_per_fpn_level.items():
            B, _, H, W = feat.shape

            cls_feat = self.stem_cls(feat)
            box_feat = self.stem_box(feat)

            cls_logits = self.pred_cls(cls_feat)        # (B, Cc, H, W)
            box_delta = self.pred_box(box_feat)         # (B, 4,  H, W)
            ctr_logits = self.pred_ctr(box_feat)        # (B, 1,  H, W)

            class_logits[lvl] = cls_logits.permute(0, 2, 3, 1).reshape(B, H * W, -1)
            boxreg_deltas[lvl] = box_delta.permute(0, 2, 3, 1).reshape(B, H * W, 4)
            centerness_logits[lvl] = ctr_logits.permute(0, 2, 3, 1).reshape(B, H * W, 1)
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        return [class_logits, boxreg_deltas, centerness_logits]


@torch.no_grad()
def fcos_match_locations_to_gt(
    locations_per_fpn_level: TensorDict,
    strides_per_fpn_level: Dict[str, int],
    gt_boxes: torch.Tensor,
) -> TensorDict:
    """
    Match FPN feature locations to GT boxes with FCOS heuristics.
    """
    matched_gt_boxes = {level_name: None for level_name in locations_per_fpn_level.keys()}

    for level_name, centers in locations_per_fpn_level.items():
        stride = strides_per_fpn_level[level_name]

        x, y = centers.unsqueeze(dim=2).unbind(dim=1)
        x0, y0, x1, y1 = gt_boxes[:, :4].unsqueeze(dim=0).unbind(dim=2)
        pairwise_dist = torch.stack([x - x0, y - y0, x1 - x, y1 - y], dim=2)

        # shape: (num_gt_boxes, num_centers, 4)
        pairwise_dist = pairwise_dist.permute(1, 0, 2)

        # Location must be inside GT
        match_matrix = pairwise_dist.min(dim=2).values > 0

        # Scale range restriction per FPN level
        pairwise_max = pairwise_dist.max(dim=2).values
        lower_bound = stride * 4 if level_name != "p3" else 0
        upper_bound = stride * 8 if level_name != "p5" else float("inf")
        match_matrix &= (pairwise_max > lower_bound) & (pairwise_max < upper_bound)

        # Prefer GT with minimum area
        gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
        match_matrix = match_matrix.to(torch.float32)
        match_matrix *= 1e8 - gt_areas[:, None]

        match_quality, matched_idxs = match_matrix.max(dim=0)
        matched_idxs[match_quality < 1e-5] = -1

        matched_boxes_this_level = gt_boxes[matched_idxs.clip(min=0)]
        matched_boxes_this_level[matched_idxs < 0, :] = -1
        matched_gt_boxes[level_name] = matched_boxes_this_level

    return matched_gt_boxes


def fcos_get_deltas_from_locations(
    locations: torch.Tensor, gt_boxes: torch.Tensor, stride: int
) -> torch.Tensor:
    """
    Compute normalized LTRB deltas from feature locations to GT box edges.
    Background rows (all -1) should produce (-1, -1, -1, -1).
    """
    ##########################################################################
    # TODO: Implement the logic to get deltas from feature locations.        #
    ##########################################################################
    x_c = locations[:, 0]
    y_c = locations[:, 1]

    x1 = gt_boxes[:, 0]
    y1 = gt_boxes[:, 1]
    x2 = gt_boxes[:, 2]
    y2 = gt_boxes[:, 3]

    L = x_c - x1
    T = y_c - y1
    R = x2 - x_c
    B = y2 - y_c

    deltas_raw = torch.stack([L, T, R, B], dim=1) / float(stride)

    bg = (x1 == -1) & (y1 == -1) & (x2 == -1) & (y2 == -1)
    deltas = torch.where(bg.unsqueeze(1), torch.full_like(deltas_raw, -1.0), deltas_raw)
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################
    return deltas


def fcos_apply_deltas_to_locations(
    deltas: torch.Tensor, locations: torch.Tensor, stride: int
) -> torch.Tensor:
    """
    Inverse transform: apply LTRB deltas at locations to get XYXY boxes.
    Negative deltas are clipped to zero before applying.
    """
    ##########################################################################
    # TODO: Implement the transformation logic to get boxes.                 #
    ##########################################################################
    d = (deltas * float(stride)).clone()
    d = torch.clamp_min(d, 0.0)

    lx = locations[:, 0]
    ly = locations[:, 1]

    x1 = lx - d[:, 0]
    y1 = ly - d[:, 1]
    x2 = lx + d[:, 2]
    y2 = ly + d[:, 3]

    output_boxes = torch.stack([x1, y1, x2, y2], dim=1)
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################
    return output_boxes


def fcos_make_centerness_targets(deltas: torch.Tensor):
    """
    Compute centerness targets from LTRB deltas (see FCOS Eq. 3).
    Background rows (all -1) should output -1.
    """
    ##########################################################################
    # TODO: Implement the centerness calculation logic.                      #
    ##########################################################################
    l, t, r, b = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]
    bg = (l == -1) & (t == -1) & (r == -1) & (b == -1)

    eps = 1e-12
    lr_min, lr_max = torch.minimum(l, r), torch.maximum(l, r)
    tb_min, tb_max = torch.minimum(t, b), torch.maximum(t, b)

    cen = torch.sqrt(((lr_min + eps) * (tb_min + eps)) / ((lr_max + eps) * (tb_max + eps)))
    centerness = torch.where(bg, torch.full_like(cen, -1.0), cen)
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################
    return centerness


class FCOS(nn.Module):
    """
    FCOS: Fully-Convolutional One-Stage Detector
    """

    def __init__(
        self, num_classes: int, fpn_channels: int, stem_channels: List[int]
    ):
        super().__init__()
        self.num_classes = num_classes

        ######################################################################
        # TODO: Initialize backbone and prediction network using arguments.   #
        ######################################################################
        self.backbone = DetectorBackboneWithFPN(fpn_channels)
        self.pred_net = FCOSPredictionNetwork(
            num_classes=num_classes,
            in_channels=fpn_channels,
            stem_channels=stem_channels,
        )
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        # EMA normalizer (per-image) for training losses
        self._normalizer = 150  # per image

    def forward(
        self,
        images: torch.Tensor,
        gt_boxes: Optional[torch.Tensor] = None,
        test_score_thresh: Optional[float] = None,
        test_nms_thresh: Optional[float] = None,
    ):
        """
        Args:
            images: (B, 3, H, W)
            gt_boxes: (B, N, 5) with (x1, y1, x2, y2, C), only during training.
        """
        ######################################################################
        # TODO: Process the image through backbone, FPN, and prediction head #
        # to obtain model predictions at every FPN location.                 #
        ######################################################################
        fpn_feats = self.backbone(images)
        pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits = self.pred_net(fpn_feats)
        ######################################################################
        # TODO: Get absolute co-ordinates `(xc, yc)` for every location in   #
        # FPN levels.                                                         #
        ######################################################################
        fpn_shapes = {k: v.shape for k, v in fpn_feats.items()}
        locations_per_fpn_level = get_fpn_location_coords(
            shape_per_fpn_level=fpn_shapes,
            strides_per_fpn_level=self.backbone.fpn_strides,
            device=images.device,
        )
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        if not self.training:
            # Inference path
            return self.inference(
                images,
                locations_per_fpn_level,
                pred_cls_logits,
                pred_boxreg_deltas,
                pred_ctr_logits,
                test_score_thresh=test_score_thresh,
                test_nms_thresh=test_nms_thresh,
            )

        ######################################################################
        # TODO: Assign ground-truth boxes to feature locations.               #
        ######################################################################
        matched_gt_boxes = []
        B = images.shape[0]
        for b in range(B):
            matched = fcos_match_locations_to_gt(
                locations_per_fpn_level=locations_per_fpn_level,
                strides_per_fpn_level=self.backbone.fpn_strides,
                gt_boxes=gt_boxes[b],
            )
            matched_gt_boxes.append(matched)

        matched_gt_deltas = []
        for b in range(B):
            per_img = {}
            for lvl_name, locs in locations_per_fpn_level.items():
                per_img[lvl_name] = fcos_get_deltas_from_locations(
                    locations=locs,
                    gt_boxes=matched_gt_boxes[b][lvl_name],
                    stride=self.backbone.fpn_strides[lvl_name],
                )
            matched_gt_deltas.append(per_img)
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        # Collate lists of dicts -> dict of batched tensors
        matched_gt_boxes = default_collate(matched_gt_boxes)
        matched_gt_deltas = default_collate(matched_gt_deltas)

        # Concatenate across FPN levels
        matched_gt_boxes = self._cat_across_fpn_levels(matched_gt_boxes)
        matched_gt_deltas = self._cat_across_fpn_levels(matched_gt_deltas)
        pred_cls_logits = self._cat_across_fpn_levels(pred_cls_logits)
        pred_boxreg_deltas = self._cat_across_fpn_levels(pred_boxreg_deltas)
        pred_ctr_logits = self._cat_across_fpn_levels(pred_ctr_logits)

        # EMA update of normalizer by number of positive locations
        num_pos_locations = (matched_gt_boxes[:, :, 4] != -1).sum()
        pos_loc_per_image = num_pos_locations.item() / images.shape[0]
        self._normalizer = 0.9 * self._normalizer + 0.1 * pos_loc_per_image

        #######################################################################
        # TODO: Calculate losses per location for classification, box reg and #
        # centerness. Background positions should contribute zero for         #
        # box/centerness losses.                                              #
        #######################################################################
        # Classification targets -> one-hot (background rows are zeros)
        tgt_cls = matched_gt_boxes[:, :, 4].clone()
        bg_mask = tgt_cls.eq(-1)
        tgt_cls = tgt_cls.masked_fill(bg_mask, 0).long()
        one_hot = F.one_hot(tgt_cls, num_classes=self.num_classes).to(pred_cls_logits.dtype)
        one_hot[bg_mask] = 0

        # Focal loss on logits
        loss_cls = sigmoid_focal_loss(pred_cls_logits, one_hot)

        # L1 loss for box deltas with background masked out (set to zero)
        l1_raw = F.l1_loss(pred_boxreg_deltas, matched_gt_deltas, reduction="none")
        l1_raw = 0.25 * l1_raw
        loss_box = torch.where(matched_gt_deltas < 0, torch.zeros_like(l1_raw), l1_raw)

        # Centerness BCE with logits, background masked to zero contribution
        flat_deltas = matched_gt_deltas.reshape(-1, 4)
        ctr_tgt = fcos_make_centerness_targets(flat_deltas)  # (B*N,)
        ctr_pred = pred_ctr_logits.reshape_as(ctr_tgt)
        bce_raw = F.binary_cross_entropy_with_logits(ctr_pred, ctr_tgt, reduction="none")
        loss_ctr = torch.where(ctr_tgt.lt(0), torch.zeros_like(bce_raw), bce_raw)
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

        # Sum over all locations and normalize
        return {
            "loss_cls": loss_cls.sum() / (self._normalizer * images.shape[0]),
            "loss_box": loss_box.sum() / (self._normalizer * images.shape[0]),
            "loss_ctr": loss_ctr.sum() / (self._normalizer * images.shape[0]),
        }

    @staticmethod
    def _cat_across_fpn_levels(
        dict_with_fpn_levels: Dict[str, torch.Tensor], dim: int = 1
    ):
        """
        Convert a dict of tensors across FPN levels {"p3", "p4", "p5"} to a
        single tensor. Values could be anything - batches of image features,
        GT targets, etc.
        """
        return torch.cat(list(dict_with_fpn_levels.values()), dim=dim)

    def inference(
        self,
        images: torch.Tensor,
        locations_per_fpn_level: Dict[str, torch.Tensor],
        pred_cls_logits: Dict[str, torch.Tensor],
        pred_boxreg_deltas: Dict[str, torch.Tensor],
        pred_ctr_logits: Dict[str, torch.Tensor],
        test_score_thresh: float = 0.3,
        test_nms_thresh: float = 0.5,
    ):
        """
        Run inference on a single input image (batch size = 1). Other input
        arguments are same as those computed in `forward` method.
        Returns predicted boxes, classes, and scores after NMS.
        """

        pred_boxes_all_levels = []
        pred_classes_all_levels = []
        pred_scores_all_levels = []

        for level_name in locations_per_fpn_level.keys():
            # Gather per-level tensors (remove batch dim)
            level_locations = locations_per_fpn_level[level_name]
            level_cls_logits = pred_cls_logits[level_name][0]  # (N, C)
            level_deltas = pred_boxreg_deltas[level_name][0]   # (N, 4)
            level_ctr_logits = pred_ctr_logits[level_name][0]  # (N, 1)

            ##################################################################
            # TODO: FCOS uses the geometric mean of class prob and centerness #
            # as confidence. Then: 1) take max over classes; 2) threshold;    #
            # 3) decode deltas; 4) clip to image.                             #
            ##################################################################
            cls_prob = level_cls_logits.sigmoid()                 # (N, C)
            ctr_prob = level_ctr_logits.sigmoid().squeeze(1)      # (N,)
            joint = cls_prob * ctr_prob.unsqueeze(1)              # (N, C)

            level_pred_scores, level_pred_classes = joint.max(dim=1)

            keep_mask = level_pred_scores > test_score_thresh
            if keep_mask.sum() == 0:
                pred_boxes_all_levels.append(level_locations.new_zeros((0, 4)))
                pred_classes_all_levels.append(level_locations.new_zeros((0,), dtype=torch.long))
                pred_scores_all_levels.append(level_locations.new_zeros((0,)))
                continue

            level_pred_scores = level_pred_scores[keep_mask]
            level_pred_classes = level_pred_classes[keep_mask]
            sel_locs = level_locations[keep_mask]
            sel_deltas = level_deltas[keep_mask]

            # Decode
            level_pred_boxes = fcos_apply_deltas_to_locations(
                deltas=sel_deltas,
                locations=sel_locs,
                stride=self.backbone.fpn_strides[level_name],
            )

            # Clip to image bounds
            H, W = images.shape[2], images.shape[3]
            level_pred_boxes[:, 0].clamp_(min=0, max=W)
            level_pred_boxes[:, 2].clamp_(min=0, max=W)
            level_pred_boxes[:, 1].clamp_(min=0, max=H)
            level_pred_boxes[:, 3].clamp_(min=0, max=H)
            ##################################################################
            #                          END OF YOUR CODE                      #
            ##################################################################

            pred_boxes_all_levels.append(level_pred_boxes)
            pred_classes_all_levels.append(level_pred_classes)
            pred_scores_all_levels.append(level_pred_scores)

        # Concatenate and run class-specific NMS
        pred_boxes_all_levels = torch.cat(pred_boxes_all_levels)
        pred_classes_all_levels = torch.cat(pred_classes_all_levels)
        pred_scores_all_levels = torch.cat(pred_scores_all_levels)

        keep = class_spec_nms(
            pred_boxes_all_levels,
            pred_scores_all_levels,
            pred_classes_all_levels,
            iou_threshold=test_nms_thresh,
        )

        pred_boxes_all_levels = pred_boxes_all_levels[keep]
        pred_classes_all_levels = pred_classes_all_levels[keep]
        pred_scores_all_levels = pred_scores_all_levels[keep]
        return (
            pred_boxes_all_levels,
            pred_classes_all_levels,
            pred_scores_all_levels,
        )
