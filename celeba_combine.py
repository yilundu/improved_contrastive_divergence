import tensorflow as tf
import math
from tqdm import tqdm
from tensorflow.python.platform import flags
from torch.utils.data import DataLoader, Dataset
from models import ResNetModel, CelebAModel
from utils import ReplayBuffer
import os.path as osp
import numpy as np
from logger import TensorBoardOutputFormat
from scipy.misc import imsave
from torchvision import transforms
import os
from itertools import product
from PIL import Image
import torch
from data import CelebAHQOverfit
from fid import get_fid_score

flags.DEFINE_integer('batch_size', 256, 'Size of inputs')
flags.DEFINE_integer('data_workers', 4, 'Number of workers to do things')
flags.DEFINE_string('logdir', 'cachedir', 'directory for logging')
flags.DEFINE_string('savedir', 'cachedir', 'location where log of experiments will be stored')
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omniglot.')
flags.DEFINE_float('step_lr', 500.0, 'size of gradient descent size')
flags.DEFINE_bool('cclass', True, 'not cclass')
flags.DEFINE_bool('proj_cclass', False, 'use for backwards compatibility reasons')
flags.DEFINE_bool('spec_norm', True, 'Whether to use spectral normalization on weights')
flags.DEFINE_bool('use_bias', True, 'Whether to use bias in convolution')
flags.DEFINE_bool('use_attention', False, 'Whether to use self attention in network')
flags.DEFINE_integer('num_steps', 80, 'number of steps to optimize the label')
flags.DEFINE_string('task', 'negation_figure', 'conceptcombine, combination_figure, negation_figure, or_figure, negation_eval')

flags.DEFINE_bool('eval', False, 'Whether to quantitively evaluate models')
flags.DEFINE_bool('latent_energy', False, 'latent energy in model')
flags.DEFINE_bool('proj_latent', False, 'Projection of latents')


# Whether to train for gentest
flags.DEFINE_bool('train', False, 'whether to train on generalization into multiple different predictions')

FLAGS = flags.FLAGS


def conceptcombineeval(model_list, select_idx):
    dataset = CelebAHQOverfit()
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=True, num_workers=4)

    n = 64
    labels = []

    for six in select_idx:
        label_ix = np.eye(2)[six]
        label_batch = np.tile(label_ix[None, :], (n, 1))
        label = torch.Tensor(label_batch).cuda()
        labels.append(label)


    def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.4*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([
            rnd_color_jitter,
            rnd_gray])
        return color_distort

    color_transform = get_color_distortion()

    im_size = 128
    transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.08, 1.0)), transforms.RandomHorizontalFlip(), color_transform, transforms.ToTensor()])

    gt_ims = []
    fake_ims = []

    for data, label in tqdm(dataloader):
        gt_ims.extend(list((data.numpy().transpose((0, 2, 3, 1)) * 255).astype(np.uint8)))

        im = torch.rand(n, 3, 128, 128).cuda()
        im_noise = torch.randn_like(im).detach()
        # First get good initializations for sampling
        for i in range(10):
            for i in range(20):
                im_noise.normal_()
                im = im + 0.001 * im_noise
                # im.requires_grad = True
                im.requires_grad_(requires_grad=True)
                energy = 0

                for model, label in zip(model_list, labels):
                    energy = model.forward(im, label) +  energy

                # print("step: ", i, energy.mean())
                im_grad = torch.autograd.grad([energy.sum()], [im])[0]

                im = im - FLAGS.step_lr *  im_grad
                im = im.detach()

                im = torch.clamp(im, 0, 1)

            im = im.detach().cpu().numpy().transpose((0, 2, 3, 1))
            im = (im * 255).astype(np.uint8)

            ims = []
            for i in range(im.shape[0]):
                im_i = np.array(transform(Image.fromarray(np.array(im[i]))))
                ims.append(im_i)

            im = torch.Tensor(np.array(ims)).cuda()

        # Then refine the images

        for i in range(FLAGS.num_steps):
            im_noise.normal_()
            im = im + 0.001 * im_noise
            # im.requires_grad = True
            im.requires_grad_(requires_grad=True)
            energy = 0

            for model, label in zip(model_list, labels):
                energy = model.forward(im, label) +  energy

            print("step: ", i, energy.mean())
            im_grad = torch.autograd.grad([energy.sum()], [im])[0]

            im = im - FLAGS.step_lr * im_grad
            im = im.detach()

            im = torch.clamp(im, 0, 1)

        im = im.detach().cpu()
        fake_ims.extend(list((im.numpy().transpose((0, 2, 3, 1)) * 255).astype(np.uint8)))
        if len(gt_ims) > 10000:
            break



    get_fid_score(gt_ims, fake_ims)
    fake_ims = np.array(fake_ims)
    fake_ims_flat = fake_ims.reshape(fake_ims.shape[0], -1)
    std_im = np.std(fake_ims, axis=0).mean()
    print("standard deviation of image", std_im)
    import pdb
    pdb.set_trace()
    print("here")



def conceptcombine(model_list, select_idx):

    n = 64
    labels = []

    for six in select_idx:
        label_ix = np.eye(2)[six]
        label_batch = np.tile(label_ix[None, :], (n, 1))
        label = torch.Tensor(label_batch).cuda()
        labels.append(label)

    im = torch.rand(n, 3, 128, 128).cuda()
    im_noise = torch.randn_like(im).detach()

    def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.4*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([
            rnd_color_jitter,
            rnd_gray])
        return color_distort

    color_transform = get_color_distortion()

    im_size = 128
    transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.08, 1.0)), transforms.RandomHorizontalFlip(), color_transform, transforms.ToTensor()])

    # First get good initializations for sampling
    for i in range(10):
        for i in range(20):
            im_noise.normal_()
            im = im + 0.001 * im_noise
            # im.requires_grad = True
            im.requires_grad_(requires_grad=True)
            energy = 0

            for model, label in zip(model_list, labels):
                energy = model.forward(im, label) +  energy

            # print("step: ", i, energy.mean())
            im_grad = torch.autograd.grad([energy.sum()], [im])[0]

            im = im - FLAGS.step_lr * im_grad
            im = im.detach()

            im = torch.clamp(im, 0, 1)

        im = im.detach().cpu().numpy().transpose((0, 2, 3, 1))
        im = (im * 255).astype(np.uint8)

        ims = []
        for i in range(im.shape[0]):
            im_i = np.array(transform(Image.fromarray(np.array(im[i]))))
            ims.append(im_i)

        im = torch.Tensor(np.array(ims)).cuda()

    # Then refine the images

    for i in range(FLAGS.num_steps):
        im_noise.normal_()
        im = im + 0.001 * im_noise
        # im.requires_grad = True
        im.requires_grad_(requires_grad=True)
        energy = 0

        for model, label in zip(model_list, labels):
            energy = model.forward(im, label) +  energy

        print("step: ", i, energy.mean())
        im_grad = torch.autograd.grad([energy.sum()], [im])[0]

        im = im - FLAGS.step_lr * im_grad
        im = im.detach()

        im = torch.clamp(im, 0, 1)

    output = im.detach().cpu().numpy()
    output = output.transpose((0, 2, 3, 1))
    output = output.reshape((-1, 8, 128, 128, 3)).transpose((0, 2, 1, 3, 4)).reshape((-1, 128 * 8, 3))
    imsave("debug.png", output)


def conceptnegate(model_list, select_idx):

    n = 64
    labels = []

    for six in select_idx:
        label_ix = np.eye(2)[six]
        label_batch = np.tile(label_ix[None, :], (n, 1))
        label = torch.Tensor(label_batch).cuda()
        labels.append(label)

    im = torch.rand(n, 3, 128, 128).cuda()
    im_noise = torch.randn_like(im).detach()

    def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.4*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([
            rnd_color_jitter,
            rnd_gray])
        return color_distort

    color_transform = get_color_distortion()

    im_size = 128
    transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.08, 1.0)), transforms.RandomHorizontalFlip(), color_transform, transforms.ToTensor()])

    # First get good initializations for sampling
    for i in range(10):
        for i in range(20):
            im_noise.normal_()
            im = im + 0.001 * im_noise
            # im.requires_grad = True
            im.requires_grad_(requires_grad=True)
            energy = 0

            for i, (model, label) in enumerate(zip(model_list, labels)):
                # Negate the 2nd model
                if i == 1:
                    energy = -0.001 * model.forward(im, label) +  energy
                else:
                    energy = model.forward(im, label) +  energy

            # print("step: ", i, energy.mean())
            im_grad = torch.autograd.grad([energy.sum()], [im])[0]

            im = im - FLAGS.step_lr * im_grad
            im = im.detach()

            im = torch.clamp(im, 0, 1)

        im = im.detach().cpu().numpy().transpose((0, 2, 3, 1))
        im = (im * 255).astype(np.uint8)

        ims = []
        for i in range(im.shape[0]):
            im_i = np.array(transform(Image.fromarray(np.array(im[i]))))
            ims.append(im_i)

        im = torch.Tensor(np.array(ims)).cuda()

    # Then refine the images

    for i in range(FLAGS.num_steps):
        im_noise.normal_()
        im = im + 0.001 * im_noise
        # im.requires_grad = True
        im.requires_grad_(requires_grad=True)
        energy = 0

        for i, (model, label) in enumerate(zip(model_list, labels)):
            # Negate the 2nd model
            if i == 1:
                energy = -0.001 * model.forward(im, label) +  energy
            else:
                energy = model.forward(im, label) +  energy

        print("step: ", i, energy.mean())
        im_grad = torch.autograd.grad([energy.sum()], [im])[0]

        im = im - FLAGS.step_lr * im_grad
        im = im.detach()

        im = torch.clamp(im, 0, 1)

    output = im.detach().cpu().numpy()
    output = output.transpose((0, 2, 3, 1))
    output = output.reshape((-1, 8, 128, 128, 3)).transpose((0, 2, 1, 3, 4)).reshape((-1, 128 * 8, 3))
    imsave("debug.png", output)


def conceptdisjunction(model_list, select_idx):

    n = 64
    labels = []

    for six in select_idx:
        label_ix = np.eye(2)[six]
        label_batch = np.tile(label_ix[None, :], (n, 1))
        label = torch.Tensor(label_batch).cuda()
        labels.append(label)

    im = torch.rand(n, 3, 128, 128).cuda()
    im_noise = torch.randn_like(im).detach()

    def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.4*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([
            rnd_color_jitter,
            rnd_gray])
        return color_distort

    color_transform = get_color_distortion()

    im_size = 128
    transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.08, 1.0)), transforms.RandomHorizontalFlip(), color_transform, transforms.ToTensor()])

    # First get good initializations for sampling
    for i in range(10):
        for i in range(20):
            im_noise.normal_()
            im = im + 0.001 * im_noise
            # im.requires_grad = True
            im.requires_grad_(requires_grad=True)
            energy = 0

            # Scale to determine temperature on distributions
            scale = 100
            energies = []
            for model, label in zip(model_list, labels):
                energy = model.forward(im, label)
                energies.append(energy)

            energy = torch.stack(energies, dim=1)
            energy = -torch.logsumexp(scale * (-1 * energy), dim=1) / scale

            # print("step: ", i, energy.mean())
            im_grad = torch.autograd.grad([energy.sum()], [im])[0]

            im = im - FLAGS.step_lr * im_grad
            im = im.detach()

            im = torch.clamp(im, 0, 1)

        im = im.detach().cpu().numpy().transpose((0, 2, 3, 1))
        im = (im * 255).astype(np.uint8)

        ims = []
        for i in range(im.shape[0]):
            im_i = np.array(transform(Image.fromarray(np.array(im[i]))))
            ims.append(im_i)

        im = torch.Tensor(np.array(ims)).cuda()

    # Then refine the images

    for i in range(FLAGS.num_steps):
        im_noise.normal_()
        im = im + 0.001 * im_noise
        # im.requires_grad = True
        im.requires_grad_(requires_grad=True)
        energy = 0

        scale = 100
        energies = []
        for model, label in zip(model_list, labels):
            energy = model.forward(im, label)
            energies.append(energy)

        energy = torch.stack(energies, dim=1)
        energy = -torch.logsumexp(scale * (-1 * energy), dim=1) / scale

        print("step: ", i, energy.mean())
        im_grad = torch.autograd.grad([energy.sum()], [im])[0]

        im = im - FLAGS.step_lr * im_grad
        im = im.detach()

        im = torch.clamp(im, 0, 1)

    output = im.detach().cpu().numpy()
    output = output.transpose((0, 2, 3, 1))
    output = output.reshape((-1, 8, 128, 128, 3)).transpose((0, 2, 1, 3, 4)).reshape((-1, 128 * 8, 3))
    imsave("debug.png", output)


def combine_main(models, resume_iters, select_idx):

    model_list = []

    for model, resume_iter in zip(models, resume_iters):
        model_path = osp.join("cachedir", model, "model_{}.pth".format(resume_iter))
        checkpoint = torch.load(model_path)
        FLAGS_model = checkpoint['FLAGS']
        model_base = CelebAModel(FLAGS_model)
        model_base.load_state_dict(checkpoint['ema_model_state_dict_0'])
        model_base = model_base.cuda()
        model_list.append(model_base)

    conceptcombine(model_list, select_idx)


if __name__ == "__main__":
    models_orig = ['celeba_128_male_2', 'celeba_128_old_2', 'celeba_128_smiling_2', 'celeba_128_wavy_hair_2']
    resume_iters_orig = ["latest", "10000", "13000", "9000"]

    ##################################
    # Settings for the composition_figure
    models = [models_orig[1]]
    resume_iters = [resume_iters_orig[1]]
    select_idx = [1]

    # models = models + [models_orig[0]]
    # resume_iters = resume_iters + [resume_iters_orig[0]]
    # select_idx = select_idx + [0]

    # models = models + [models_orig[2]]
    # resume_iters = resume_iters + [resume_iters_orig[2]]
    # select_idx = select_idx + [1]

    # models = models + [models_orig[3]]
    # resume_iters = resume_iters + [resume_iters_orig[3]]
    # select_idx = select_idx + [1]

    FLAGS.step_lr = FLAGS.step_lr / len(models)

    # List of 4 attributes that might be good
    # Young -> Female -> Smiling -> Wavy
    combine_main(models, resume_iters, select_idx)

