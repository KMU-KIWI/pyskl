# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import mmcv
import torch
from torch import nn

from pyskl.apis import init_recognizer


class MMCVWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, keypoint):
        x = self.model.extract_feat(keypoint)
        x = self.model.cls_head(x)
        return x


def main(args):
    config = mmcv.Config.fromfile(args.config)
    config.data.test.pipeline = [
        x for x in config.data.test.pipeline if x["type"] != "DecompressPose"
    ]
    # Are we using GCN for Infernece?
    GCN_flag = "GCN" in config.model.type
    GCN_nperson = None
    if GCN_flag:
        format_op = [
            op for op in config.data.test.pipeline if op["type"] == "FormatGCNInput"
        ][0]
        GCN_nperson = format_op["num_person"]

    model = init_recognizer(config, args.checkpoint, device="cpu")
    model = MMCVWrapper(model)

    dummy_input = torch.randn(1, GCN_nperson, args.num_frame, args.num_joints, 3)
    torch.onnx.export(model, dummy_input, "model.onnx", opset_version=12)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/posec3d/slowonly_r50_ntu120_xsub/joint.py",
        help="skeleton action recognition config file path",
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "https://download.openmmlab.com/mmaction/pyskl/ckpt/"
            "posec3d/slowonly_r50_ntu120_xsub/joint.pth"
        ),
        help="skeleton action recognition checkpoint file/url",
    )
    parser.add_argument("--num_frame", type=int, default=30)
    parser.add_argument("--num_joints", type=int, default=17)

    args = parser.parse_args()

    main(args)
