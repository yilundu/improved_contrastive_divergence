Pytorch code for the paper, [Improved Contrastive Divergence Training of Energy Based Models](https://arxiv.org/abs/2012.01316)

# Installation

Create a new environment and install the requirements file:

```
pip install -r requirements.txt
```


# Training CIFAR-10 models

The following command trains a basic cifar10 model.
```
python train.py --exp=cifar10_model --step_lr=100.0 --num_steps=40 --cuda --ensembles=1 --kl_coeff=1.0 --kl=True --multiscale --self_attn --reservoir
```


# Training CelebA models

The following command trains a CelebA model.

```
 python train.py --dataset=celeba --exp=celeba_model --step_lr=500.0 --num_steps=40 --kl=True --gpus=8 --filter_dim=128 --multiscale --self_attn --reservoir
```


# Code for composing CelebA models

The following command combines models trained on CelebA together.

```
 python celeba_combine.py
```

Given a list of model names, resume iterations, and conditioned values, the code composes each model together


# Code for composing CIFAR-10 models

The following command combines models trained on CIFAR-10 together.

```
 python cifar10_combine.py
```

Given a list of model names, resume iterations, and conditioned values, the code composes each model together

# Generation and sandbox evaluation

The ebm_sandbox.py file consists of functions for evaluating EBMs (such as out-of-distribution detection). The test_inception.py contains code to evaluate generations of the model.

# Citation

If you find the code or paper helpful, please consider citing:

```
@article{du2020improved,
  title={Improved Contrastive Divergence Training of Energy Based Models},
  author={Du, Yilun and Li, Shuang and Tenenbaum, Joshua and Mordatch, Igor},
  journal={arXiv preprint arXiv:2012.01316},
  year={2020}
}
```
