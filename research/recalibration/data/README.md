# CIFAR-100 test logits

Model: cifar100_resnet56 from github.com/chenyaofo/pytorch-cifar-models (torch.hub, pretrained).
Measured top-1 on the 10k test set: 0.7262 (card claims ~0.7263).
Contents: logits (10000x100 f32), labels, head_weight (100xd), head_bias,
coarse_of_fine (20 superclasses), model_name, top1.
Regenerate: python scratchpad/cifar_pipeline.py (see session transcript).
