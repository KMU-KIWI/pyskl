model = dict(
    type="RecognizerGCN",
    backbone=dict(
        type="SGN",
        in_channels=3,
        base_channels=64,
        num_joints=17,
        T=30,
        bias=True,
    ),
    cls_head=dict(type="GCNHead", num_classes=120, in_channels=512),
)

dataset_type = "PoseDataset"
ann_file = "data/nturgbd/ntu120_hrnet.pkl"
train_pipeline = [
    dict(type="PreNormalize2D"),
    dict(type="GenSkeFeat", dataset="coco", feats=["j"]),
    dict(type="UniformSample", clip_len=30),
    dict(type="PoseDecode"),
    dict(type="FormatGCNInput", num_person=2),
    dict(type="Collect", keys=["keypoint", "label"], meta_keys=[]),
    dict(type="ToTensor", keys=["keypoint"]),
]
val_pipeline = [
    dict(type="PreNormalize2D"),
    dict(type="GenSkeFeat", dataset="coco", feats=["j"]),
    dict(type="UniformSample", clip_len=30, num_clips=1, test_mode=True),
    dict(type="PoseDecode"),
    dict(type="FormatGCNInput", num_person=2),
    dict(type="Collect", keys=["keypoint", "label"], meta_keys=[]),
    dict(type="ToTensor", keys=["keypoint"]),
]
test_pipeline = [
    dict(type="PreNormalize2D"),
    dict(type="GenSkeFeat", dataset="coco", feats=["j"]),
    dict(type="UniformSample", clip_len=30, num_clips=10, test_mode=True),
    dict(type="PoseDecode"),
    dict(type="FormatGCNInput", num_person=2),
    dict(type="Collect", keys=["keypoint", "label"], meta_keys=[]),
    dict(type="ToTensor", keys=["keypoint"]),
]
data = dict(
    videos_per_gpu=128,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type="RepeatDataset",
        times=5,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file,
            pipeline=train_pipeline,
            split="xsub_train",
        ),
    ),
    val=dict(
        type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split="xsub_val"
    ),
    test=dict(
        type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split="xsub_val"
    ),
)

# optimizer
optimizer = dict(type="SGD", lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy="CosineAnnealing", min_lr=0, by_epoch=False)
total_epochs = 16
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=["top_k_accuracy"])
log_config = dict(interval=30, hooks=[dict(type="WandbLoggerHook")])

# runtime settings
log_level = "INFO"
work_dir = "./work_dirs/sgn_30/sgn_pyskl_ntu120_xsub_hrnet/j"
