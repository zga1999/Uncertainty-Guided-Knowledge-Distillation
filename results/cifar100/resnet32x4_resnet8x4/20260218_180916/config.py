default_scope = 'mmcls'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=0,
        save_best='accuracy/top1',
        rule='greater',
        max_keep_ckpts=1,
        save_last=False),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='ClsVisualizer', vis_backends=[dict(type='LocalVisBackend')])
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=None, deterministic=False)
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005))
param_scheduler = [
    dict(
        type='MultiStepLR',
        by_epoch=True,
        milestones=[150, 180, 210],
        gamma=0.1)
]
train_cfg = dict(by_epoch=True, max_epochs=240, val_interval=1)
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(base_batch_size=256)
dataset_type = 'CIFAR100'
data_preprocessor = dict(
    num_classes=100,
    mean=[129.304, 124.07, 112.434],
    std=[68.17, 65.392, 70.418],
    to_rgb=False,
    batch_augments=dict(
        augments=[
            dict(type='Mixup', alpha=0.4),
            dict(type='CutMix', alpha=1.0)
        ],
        probs=[0.6, 0.2]))
train_pipeline = [
    dict(type='RandomCrop', crop_size=32, padding=4),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackClsInputs')
]
test_pipeline = [dict(type='PackClsInputs')]
train_dataloader = dict(
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    batch_size=64,
    num_workers=4,
    dataset=dict(
        type='CIFAR100',
        data_prefix='../datasets/mmcls',
        test_mode=False,
        pipeline=[
            dict(type='RandomCrop', crop_size=32, padding=4),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='PackClsInputs')
        ]),
    sampler=dict(type='DefaultSampler', shuffle=True))
val_dataloader = dict(
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    batch_size=64,
    num_workers=4,
    dataset=dict(
        type='CIFAR100',
        data_prefix='../datasets/mmcls',
        test_mode=True,
        pipeline=[dict(type='PackClsInputs')]),
    sampler=dict(type='DefaultSampler', shuffle=False))
val_evaluator = dict(type='Accuracy', topk=(1, ))
test_dataloader = dict(
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    batch_size=64,
    num_workers=4,
    dataset=dict(
        type='CIFAR100',
        data_prefix='../datasets/mmcls',
        test_mode=True,
        pipeline=[dict(type='PackClsInputs')]),
    sampler=dict(type='DefaultSampler', shuffle=False))
test_evaluator = dict(type='Accuracy', topk=(1, ))
model = dict(
    type='ClassificationDistiller',
    teacher_pretrained='weights/resnet32x4_81.63.pth',
    teacher_cfg='projects/_models_/cifar100_resnet32x4.py',
    student_cfg='projects/_models_/cifar100_resnet8x4.py',
    distill_cfg=dict(
        loss_ori=dict(type='ORILoss', use_this=True, loss_weight=1.0),
        loss_uatm=dict(
            type='UATMLoss', use_this=True, temp=4.0, loss_weight=15.0),
        loss_ugsa=dict(
            type='UGSALoss', use_this=True, loss_type='kl', loss_weight=1.0)))
launcher = 'none'
work_dir = 'work_dirs/cifar100/resnet32x4_resnet8x4/UATM+UGSA_temp=4.0_loss_weight=15.0+loss_weight=1.0_1'
