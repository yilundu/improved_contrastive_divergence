import tensorflow as tf
import numpy as np
import timeit
from tensorflow.python.platform import flags
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
import time
from multiprocessing import Process

from data import Cifar10, CelebAHQ, Mnist, ImageNet, LSUNBed, STLDataset
from models import ResNetModel, CelebAModel, MNISTModel, ImagenetModel
import os.path as osp
import os
from logger import TensorBoardOutputFormat
from utils import ReplayBuffer, ReservoirBuffer
from tqdm import tqdm
import random
from torch.utils.data import DataLoader
import time as time
from io import StringIO
from tensorflow.core.util import event_pb2
import torch
import numpy as np
from scipy.misc import imsave
import matplotlib.pyplot as plt
from easydict import EasyDict

from utils import ReplayBuffer
from torch.optim import Adam, SGD
import torch.multiprocessing as mp
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP

FLAGS = flags.FLAGS

# Distributed training hyperparameters
flags.DEFINE_integer('nodes', 1,
    'number of nodes for training')
flags.DEFINE_integer('gpus', 1,
    'number of gpus per nodes')
flags.DEFINE_integer('node_rank', 0,
    'rank of node')

# Configurations for distributed training
flags.DEFINE_string('master_addr', '8.8.8.8',
    'address of communicating server')
flags.DEFINE_string('port', '10002',
    'port of training')
flags.DEFINE_bool('slurm', False,
    'whether we are on slurm')
flags.DEFINE_bool('repel_im', True,
    'maximize entropy by repeling images from each other')
flags.DEFINE_bool('hmc', False,
    'use the hamiltonian monte carlo sampler')
flags.DEFINE_bool('square_energy', False,
    'make the energy square')
flags.DEFINE_bool('alias', False,
    'make the energy square')

flags.DEFINE_string('dataset','cifar10',
    'cifar10 or celeba')
flags.DEFINE_integer('batch_size', 128, 'batch size during training')
flags.DEFINE_bool('multiscale', True, 'A multiscale EBM')
flags.DEFINE_bool('self_attn', True, 'Use self attention in models')
flags.DEFINE_bool('sigmoid', False, 'Apply sigmoid on energy (can improve the stability)')
flags.DEFINE_bool('anneal', False, 'Decrease noise over Langevin steps')
flags.DEFINE_integer('data_workers', 4,
    'Number of different data workers to load data in parallel')
flags.DEFINE_integer('buffer_size', 10000, 'Size of inputs')

# General Experiment Settings
flags.DEFINE_string('logdir', 'cachedir',
    'location where log of experiments will be stored')
flags.DEFINE_string('exp', 'default', 'name of experiments')
flags.DEFINE_integer('log_interval', 10, 'log outputs every so many batches')
flags.DEFINE_integer('save_interval', 1000,'save outputs every so many batches')
flags.DEFINE_integer('test_interval', 1000,'evaluate outputs every so many batches')
flags.DEFINE_integer('resume_iter', 0, 'iteration to resume training from')
flags.DEFINE_bool('train', True, 'whether to train or test')
flags.DEFINE_bool('transform', True, 'apply data augmentation when sampling from the replay buffer')
flags.DEFINE_bool('kl', True, 'apply a KL term to loss')
flags.DEFINE_bool('cuda', True, 'move device on cuda')
flags.DEFINE_integer('epoch_num', 10000, 'Number of Epochs to train on')
flags.DEFINE_integer('ensembles', 1, 'Number of ensembles to train models with')
flags.DEFINE_float('lr', 2e-4, 'Learning for training')
flags.DEFINE_float('kl_coeff', 1.0, 'coefficient for kl')

# EBM Specific Experiments Settings
flags.DEFINE_string('objective', 'cd', 'use the cd objective')

# Setting for MCMC sampling
flags.DEFINE_integer('num_steps', 40, 'Steps of gradient descent for training')
flags.DEFINE_float('step_lr', 100.0, 'Size of steps for gradient descent')
flags.DEFINE_bool('replay_batch', True, 'Use MCMC chains initialized from a replay buffer.')
flags.DEFINE_bool('reservoir', True, 'Use a reservoir of past entires')
flags.DEFINE_float('noise_scale', 1.,'Relative amount of noise for MCMC')

# Architecture Settings
flags.DEFINE_integer('filter_dim', 64, 'number of filters for conv nets')
flags.DEFINE_integer('im_size', 32, 'size of images')
flags.DEFINE_bool('spec_norm', False, 'Whether to use spectral normalization on weights')
flags.DEFINE_bool('norm', True, 'Use group norm in models norm in models')

# Conditional settings
flags.DEFINE_bool('cond', False, 'conditional generation with the model')
flags.DEFINE_bool('all_step', False, 'backprop through all langevin steps')
flags.DEFINE_bool('log_grad', False, 'log the gradient norm of the kl term')
flags.DEFINE_integer('cond_idx', 0, 'conditioned index')


def compress_x_mod(x_mod):
    x_mod = (255 * np.clip(x_mod, 0, 1)).astype(np.uint8)
    return x_mod


def decompress_x_mod(x_mod):
    x_mod = x_mod / 256  + \
        np.random.uniform(0, 1 / 256, x_mod.shape)
    return x_mod


def make_image(tensor):
    """Convert an numpy representation image to Image protobuf"""
    from PIL import Image
    if len(tensor.shape) == 4:
        _, height, width, channel = tensor.shape
    elif len(tensor.shape) == 3:
        height, width, channel = tensor.shape
    elif len(tensor.shape) == 2:
        height, width = tensor.shape
        channel = 1
    tensor = tensor.astype(np.uint8).squeeze()
    image = Image.fromarray(tensor)
    import io
    output = io.BytesIO()
    image.save(output, format='png')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                            width=width,
                            colorspace=channel,
                            encoded_image_string=image_string)

def sync_model(models):
    size = float(dist.get_world_size())

    for model in models:
        for param in model.parameters():
            dist.broadcast(param.data, 0)


def ema_model(models, models_ema, mu=0.99):
    for model, model_ema in zip(models, models_ema):
        for param, param_ema in zip(model.parameters(), model_ema.parameters()):
            param_ema.data[:] = mu * param_ema.data + (1 - mu) * param.data


def average_gradients(models):
    size = float(dist.get_world_size())

    for model in models:
        for param in model.parameters():
            if param.grad is None:
                continue

            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
            param.grad.data /= size



def log_image(im, logger, tag, step=0):
    im = make_image(im)

    summary = [tf.Summary.Value(tag=tag, image=im)]
    summary = tf.Summary(value=summary)
    event = event_pb2.Event(summary=summary)
    event.step = step
    logger.writer.WriteEvent(event)
    logger.writer.Flush()

def rescale_im(image):
    image = np.clip(image, 0, 1)
    return (np.clip(image * 256, 0, 255)).astype(np.uint8)


def hamiltonian(x, v, model, label):
    energy = 0.5 * torch.pow(v, 2).sum(dim=1).sum(dim=1).sum(dim=1) + model.forward(x, label).squeeze()
    return energy


def leapfrog_step(x, v, model, step_size, num_steps, label, sample=False):
    x.requires_grad_(requires_grad=True)
    energy = model.forward(x, label)
    im_grad = torch.autograd.grad([energy.sum()], [x])[0]
    v = v - 0.5 * step_size * im_grad
    im_negs = []

    for i in range(num_steps):
        x.requires_grad_(requires_grad=True)
        energy = model.forward(x, label)

        if i == num_steps - 1:
            im_grad = torch.autograd.grad([energy.sum()], [x], create_graph=True)[0]
            v = v - step_size * im_grad
            x = x + step_size * v
            v = v.detach()
        else:
            im_grad = torch.autograd.grad([energy.sum()], [x])[0]
            v = v - step_size * im_grad
            x = x + step_size * v
            x = x.detach()
            v = v.detach()


        if sample:
            im_negs.append(x)

        if i % 10 == 0:
            print(i, hamiltonian(torch.sigmoid(x), v, model, label).mean(), torch.abs(im_grad).mean())

    if sample:
        return x, im_negs, v, im_grad
    else:
        return x, v, im_grad


def gen_hmc_image(label, FLAGS, model, im_neg, num_steps, sample=False):
    step_size = FLAGS.step_lr

    v = 0.001 * torch.randn_like(im_neg)

    if sample:
        im_neg, im_negs, v, im_grad = leapfrog_step(im_neg, v, model, step_size, num_steps, label, sample=sample)
        return im_neg, im_negs, im_grad, v
    else:
        im_neg, v, im_grad = leapfrog_step(im_neg, v, model, step_size, num_steps, label, sample=sample)
        return im_neg, im_grad, v


def gen_image(label, FLAGS, model, im_neg, num_steps, sample=False):
    im_noise = torch.randn_like(im_neg).detach()

    im_negs_samples = []

    for i in range(num_steps):
        im_noise.normal_()

        if FLAGS.anneal:
            im_neg = im_neg + 0.001 * (num_steps - i - 1) / num_steps * im_noise
        else:
            im_neg = im_neg + 0.001 * im_noise

        im_neg.requires_grad_(requires_grad=True)
        energy = model.forward(im_neg, label)

        if FLAGS.all_step:
            im_grad = torch.autograd.grad([energy.sum()], [im_neg], create_graph=True)[0]
        else:
            im_grad = torch.autograd.grad([energy.sum()], [im_neg])[0]

        if i == num_steps - 1:
            im_neg_orig = im_neg
            im_neg = im_neg - FLAGS.step_lr * im_grad

            if FLAGS.dataset == "cifar10":
                n = 128
            elif FLAGS.dataset == "celeba":
                # Save space
                n = 128
            elif FLAGS.dataset == "lsun":
                # Save space
                n = 32
            elif FLAGS.dataset == "object":
                # Save space
                n = 32
            elif FLAGS.dataset == "mnist":
                n = 32
            elif FLAGS.dataset == "imagenet":
                n = 32
            elif FLAGS.dataset == "stl":
                n = 32

            im_neg_kl = im_neg_orig[:n]
            if sample:
                pass
            else:
                energy = model.forward(im_neg_kl, label)
                im_grad = torch.autograd.grad([energy.sum()], [im_neg_kl], create_graph=True)[0]

            im_neg_kl = im_neg_kl - FLAGS.step_lr * im_grad[:n]
            im_neg_kl = torch.clamp(im_neg_kl, 0, 1)
        else:
            im_neg = im_neg - FLAGS.step_lr * im_grad

        im_neg = im_neg.detach()

        if sample:
            im_negs_samples.append(im_neg)

        im_neg = torch.clamp(im_neg, 0, 1)

    if sample:
        return im_neg, im_neg_kl, im_negs_samples, im_grad
    else:
        return im_neg, im_neg_kl, im_grad


def test(model, logger, dataloader):
    pass

def train(models, models_ema, optimizer, logger, dataloader, resume_iter, logdir, FLAGS, rank_idx, best_inception):

    torch.cuda.set_device(rank_idx)

    if FLAGS.replay_batch:
        if FLAGS.reservoir:
            replay_buffer = ReservoirBuffer(FLAGS.buffer_size, FLAGS.transform, FLAGS.dataset)
        else:
            replay_buffer = ReplayBuffer(FLAGS.buffer_size, FLAGS.transform, FLAGS.dataset)

    if rank_idx == 0:
        from inception import get_inception_score

    itr = resume_iter
    im_neg = None
    gd_steps = 1

    optimizer.zero_grad()

    num_steps = FLAGS.num_steps

    if FLAGS.cuda:
        dev = torch.device("cuda:{}".format(rank_idx))
    else:
        dev = torch.device("cpu")

    for epoch in range(FLAGS.epoch_num):
        tock = time.time()
        for data_corrupt, data, label in dataloader:
            label = label.float().cuda(rank_idx)
            data = data.permute(0, 3, 1, 2).float().contiguous()

            # Generate samples to evaluate inception score
            if itr % FLAGS.save_interval == 0:
                if FLAGS.dataset == "cifar10":
                    data_corrupt = torch.Tensor(np.random.uniform(0.0, 1.0, (128, 32, 32, 3)))
                    repeat = 128 // FLAGS.batch_size + 1
                    label = torch.cat([label] * repeat, axis=0)
                    label = label[:128]
                elif FLAGS.dataset == "celeba":
                    data_corrupt = torch.Tensor(np.random.uniform(0.0, 1.0, (data.shape[0], 128, 128, 3)))
                    label = label[:data.shape[0]]
                    data_corrupt = data_corrupt[:label.shape[0]]
                elif FLAGS.dataset == "stl":
                    data_corrupt = torch.Tensor(np.random.uniform(0.0, 1.0, (32, 48, 48, 3)))
                    label = label[:32]
                    data_corrupt = data_corrupt[:label.shape[0]]
                elif FLAGS.dataset == "lsun":
                    data_corrupt = torch.Tensor(np.random.uniform(0.0, 1.0, (32, 128, 128, 3)))
                    label = label[:32]
                    data_corrupt = data_corrupt[:label.shape[0]]
                elif FLAGS.dataset == "imagenet":
                    data_corrupt = torch.Tensor(np.random.uniform(0.0, 1.0, (32, 128, 128, 3)))
                    label = label[:32]
                    data_corrupt = data_corrupt[:label.shape[0]]
                elif FLAGS.dataset == "object":
                    data_corrupt = torch.Tensor(np.random.uniform(0.0, 1.0, (32, 128, 128, 3)))
                    label = label[:32]
                    data_corrupt = data_corrupt[:label.shape[0]]
                elif FLAGS.dataset == "mnist":
                    data_corrupt = torch.Tensor(np.random.uniform(0.0, 1.0, (32, 28, 28, 1)))
                    label = label[:32]
                    data_corrupt = data_corrupt[:label.shape[0]]
                else:
                    assert False

            data_corrupt = torch.Tensor(data_corrupt.float()).permute(0, 3, 1, 2).float().contiguous()
            data = data.cuda(rank_idx)
            data_corrupt = data_corrupt.cuda(rank_idx)

            if FLAGS.replay_batch and len(replay_buffer) >= FLAGS.batch_size:
                replay_batch, idxs = replay_buffer.sample(data_corrupt.size(0))
                replay_batch = decompress_x_mod(replay_batch)
                replay_mask = (
                    np.random.uniform(
                        0,
                        1,
                        data_corrupt.size(0)) > 0.001)
                data_corrupt[replay_mask] = torch.Tensor(replay_batch[replay_mask]).cuda(rank_idx)
            else:
                idxs = None

            ix = random.randint(0, len(models) - 1)
            model = models[ix]

            if FLAGS.hmc:
                if itr % FLAGS.save_interval == 0:
                    im_neg, im_samples, x_grad, v = gen_hmc_image(label, FLAGS, model, data_corrupt, num_steps, sample=True)
                else:
                    im_neg, x_grad, v = gen_hmc_image(label, FLAGS, model, data_corrupt, num_steps)
            else:
                if itr % FLAGS.save_interval == 0:
                    im_neg, im_neg_kl, im_samples, x_grad = gen_image(label, FLAGS, model, data_corrupt, num_steps, sample=True)
                else:
                    im_neg, im_neg_kl, x_grad = gen_image(label, FLAGS, model, data_corrupt, num_steps)

            energy_pos = model.forward(data, label[:data.size(0)])
            energy_neg = model.forward(im_neg.clone(), label)

            if FLAGS.replay_batch and (im_neg is not None):
                replay_buffer.add(compress_x_mod(im_neg.detach().cpu().numpy()))

            loss = energy_pos.mean() - energy_neg.mean() #
            loss = loss  + (torch.pow(energy_pos, 2).mean() + torch.pow(energy_neg, 2).mean())

            if FLAGS.kl:
                model.requires_grad_(False)
                loss_kl = model.forward(im_neg_kl, label)
                model.requires_grad_(True)
                loss = loss + FLAGS.kl_coeff * loss_kl.mean()

                if FLAGS.repel_im:
                    start = timeit.timeit()
                    bs = im_neg_kl.size(0)

                    if FLAGS.dataset in ["celeba", "imagenet", "object", "lsun", "stl"]:
                        im_neg_kl = im_neg_kl[:, :, :, :].contiguous()

                    im_flat = torch.clamp(im_neg_kl.view(bs, -1), 0, 1)

                    if FLAGS.dataset == "cifar10":
                        if len(replay_buffer) > 1000:
                            compare_batch, idxs = replay_buffer.sample(100, no_transform=False)
                            compare_batch = decompress_x_mod(compare_batch)
                            compare_batch = torch.Tensor(compare_batch).cuda(rank_idx)
                            compare_flat = compare_batch.view(100, -1)

                            dist_matrix = torch.norm(im_flat[:, None, :] - compare_flat[None, :, :], p=2, dim=-1)
                            loss_repel = torch.log(dist_matrix.min(dim=1)[0]).mean()
                            loss = loss - 0.3 * loss_repel
                        else:
                            loss_repel = torch.zeros(1)
                    else:
                        if len(replay_buffer) > 1000:
                            compare_batch, idxs = replay_buffer.sample(100, no_transform=False, downsample=True)
                            compare_batch = decompress_x_mod(compare_batch)
                            compare_batch = torch.Tensor(compare_batch).cuda(rank_idx)
                            compare_flat = compare_batch.view(100, -1)
                            dist_matrix = torch.norm(im_flat[:, None, :] - compare_flat[None, :, :], p=2, dim=-1)
                            loss_repel = torch.log(dist_matrix.min(dim=1)[0]).mean()
                        else:
                            loss_repel = torch.zeros(1).cuda(rank_idx)

                        loss = loss - 0.3 * loss_repel

                    end = timeit.timeit()
                else:
                    loss_repel = torch.zeros(1)

            else:
                loss_kl = torch.zeros(1)
                loss_repel = torch.zeros(1)

            if FLAGS.hmc:
                v_flat = v.view(v.size(0), -1)
                im_grad_flat = x_grad.view(x_grad.size(0), -1)
                dot_product = F.normalize(v_flat, dim=1) * F.normalize(im_grad_flat, dim=1)
                hmc_loss = torch.abs(dot_product.sum(dim=1)).mean()
                loss = loss + 0.01 * hmc_loss
            else:
                hmc_loss = torch.zeros(1)

            if FLAGS.log_grad and len(replay_buffer) > 1000:
                loss_kl = loss_kl - 0.1 * loss_repel
                loss_kl = loss_kl.mean()
                loss_ml = energy_pos.mean() - energy_neg.mean()

                loss_ml.backward(retain_graph=True)
                ele = []

                for param in model.parameters():
                    if param.grad is not None:
                        ele.append(torch.norm(param.grad.data))

                ele = torch.stack(ele, dim=0)
                ml_grad = torch.mean(ele)
                model.zero_grad()

                loss_kl.backward(retain_graph=True)
                ele = []

                for param in model.parameters():
                    if param.grad is not None:
                        ele.append(torch.norm(param.grad.data))

                ele = torch.stack(ele, dim=0)
                kl_grad = torch.mean(ele)
                model.zero_grad()

            else:
                ml_grad = None
                kl_grad = None

            loss.backward()

            if FLAGS.gpus > 1:
                average_gradients(models)

            [clip_grad_norm(model.parameters(), 0.5) for model in models]

            optimizer.step()
            optimizer.zero_grad()

            ema_model(models, models_ema)

            if torch.isnan(energy_pos.mean()):
                assert False

            if torch.abs(energy_pos.mean()) > 10.0:
                assert False

            if itr % FLAGS.log_interval == 0 and rank_idx==0:
                tick = time.time()
                kvs = {}
                kvs['e_pos'] = energy_pos.mean().item()
                kvs['e_pos_std'] = energy_pos.std().item()
                kvs['e_neg'] = energy_neg.mean().item()
                kvs['kl_mean'] = loss_kl.mean().item()
                kvs['loss_repel'] = loss_repel.mean().item()
                kvs['e_neg_std'] = energy_neg.std().item()
                kvs['e_diff'] = kvs['e_pos'] - kvs['e_neg']
                kvs['x_grad'] = np.abs(x_grad.detach().cpu().numpy()).mean()
                kvs['iter'] = itr
                kvs['hmc_loss'] = hmc_loss.item()
                kvs['num_steps'] = num_steps
                kvs['t_diff'] = tick - tock

                if FLAGS.replay_batch:
                    kvs['length'] = len(replay_buffer)

                if (ml_grad is not None):
                    kvs['kl_grad'] = kl_grad
                    kvs['ml_grad'] = ml_grad

                string = "Obtained a total of "
                for key, value in kvs.items():
                    string += "{}: {}, ".format(key, value)

                print(string)
                logger.writekvs(kvs)
                tock = tick

            if itr % FLAGS.save_interval == 0 and rank_idx == 0 and (FLAGS.save_interval != 0):
                model_path = osp.join(logdir, "model_{}.pth".format(itr))
                ckpt = {'optimizer_state_dict': optimizer.state_dict(),
                            'FLAGS': FLAGS, 'best_inception': best_inception}

                for i in range(FLAGS.ensembles):
                    ckpt['model_state_dict_{}'.format(i)] = models[i].state_dict()
                    ckpt['ema_model_state_dict_{}'.format(i)] = models_ema[i].state_dict()

                torch.save(ckpt, model_path)

            if itr % FLAGS.save_interval == 0 and rank_idx == 0:
                im_samples = im_samples[::10]
                im_samples_total = torch.stack(im_samples, dim=1).detach().cpu().permute(0, 1, 3, 4, 2).numpy()
                try_im = im_neg
                orig_im = data_corrupt
                actual_im = rescale_im(data.detach().permute(0, 2, 3, 1).cpu().numpy())

                orig_im = rescale_im(orig_im.detach().permute(0, 2, 3, 1).cpu().numpy())
                try_im = rescale_im(try_im.detach().permute(0, 2, 3, 1).cpu().numpy()).squeeze()
                im_samples_total = rescale_im(im_samples_total)

                for i, (im, sample_im, actual_im_i) in enumerate(
                        zip(orig_im[:20], im_samples_total[:20], actual_im)):
                    shape = orig_im.shape[1:]
                    new_im = np.zeros((shape[0], shape[1] * (2 + sample_im.shape[0]), *shape[2:]))
                    size = shape[1]
                    new_im[:, :size] = im

                    for i, sample_i in enumerate(sample_im):
                        new_im[:, (i+1) * size:(i+2) * size] = sample_i

                    new_im[:, -size:] = actual_im_i

                    log_image(
                        new_im, logger, 'train_gen_{}'.format(itr), step=i)


                if rank_idx == 0:
                    score, std = get_inception_score(list(try_im), splits=1)
                    print("Inception score of {} with std of {}".format(
                            score, std))
                    kvs = {}
                    kvs['inception_score'] = score
                    kvs['inception_score_std'] = std
                    logger.writekvs(kvs)

                    if score > best_inception:
                        model_path = osp.join(logdir, "model_best.pth")
                        torch.save(ckpt, model_path)
                        best_inception = score

            itr += 1


def main_single(gpu, FLAGS):
    if FLAGS.slurm:
        init_distributed_mode(FLAGS)

    os.environ['MASTER_ADDR'] = FLAGS.master_addr
    os.environ['MASTER_PORT'] = FLAGS.port

    rank_idx = FLAGS.node_rank * FLAGS.gpus + gpu
    world_size = FLAGS.nodes * FLAGS.gpus
    print("Values of args: ", FLAGS)

    if world_size > 1:
        if FLAGS.slurm:
            dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank_idx)
        else:
            dist.init_process_group(backend='nccl', init_method='tcp://localhost:1700', world_size=world_size, rank=rank_idx)

    if FLAGS.dataset == "cifar10":
        train_dataset = Cifar10(FLAGS)
        valid_dataset = Cifar10(FLAGS, train=False, augment=False)
        test_dataset = Cifar10(FLAGS, train=False, augment=False)
    elif FLAGS.dataset == "stl":
        train_dataset = STLDataset(FLAGS)
        valid_dataset = STLDataset(FLAGS, train=False)
        test_dataset = STLDataset(FLAGS, train=False)
    elif FLAGS.dataset == "object":
        train_dataset = ObjectDataset(FLAGS.cond_idx)
        valid_dataset = ObjectDataset(FLAGS.cond_idx)
        test_dataset = ObjectDataset(FLAGS.cond_idx)
    elif FLAGS.dataset == "imagenet":
        train_dataset = ImageNet()
        valid_dataset = ImageNet()
        test_dataset = ImageNet()
    elif FLAGS.dataset == "mnist":
        train_dataset = Mnist(train=True)
        valid_dataset = Mnist(train=False)
        test_dataset = Mnist(train=False)
    elif FLAGS.dataset == "celeba":
        train_dataset = CelebAHQ(cond_idx=FLAGS.cond_idx)
        valid_dataset = CelebAHQ(cond_idx=FLAGS.cond_idx)
        test_dataset = CelebAHQ(cond_idx=FLAGS.cond_idx)
    elif FLAGS.dataset == "lsun":
        train_dataset = LSUNBed(cond_idx=FLAGS.cond_idx)
        valid_dataset = LSUNBed(cond_idx=FLAGS.cond_idx)
        test_dataset = LSUNBed(cond_idx=FLAGS.cond_idx)
    else:
        assert False

    train_dataloader = DataLoader(train_dataset, num_workers=FLAGS.data_workers, batch_size=FLAGS.batch_size, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, num_workers=FLAGS.data_workers, batch_size=FLAGS.batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, num_workers=FLAGS.data_workers, batch_size=FLAGS.batch_size, shuffle=True, drop_last=True)

    FLAGS_OLD = FLAGS

    logdir = osp.join(FLAGS.logdir, FLAGS.exp)
    best_inception = 0.0

    if FLAGS.resume_iter != 0:
        model_path = osp.join(logdir, "model_{}.pth".format(FLAGS.resume_iter))
        checkpoint = torch.load(model_path)
        best_inception = checkpoint['best_inception']
        FLAGS = checkpoint['FLAGS']

        FLAGS.resume_iter = FLAGS_OLD.resume_iter
        FLAGS.nodes = FLAGS_OLD.nodes
        FLAGS.gpus = FLAGS_OLD.gpus
        FLAGS.node_rank = FLAGS_OLD.node_rank
        FLAGS.master_addr = FLAGS_OLD.master_addr
        FLAGS.train = FLAGS_OLD.train
        FLAGS.num_steps = FLAGS_OLD.num_steps
        FLAGS.step_lr = FLAGS_OLD.step_lr
        FLAGS.batch_size = FLAGS_OLD.batch_size
        FLAGS.ensembles = FLAGS_OLD.ensembles
        FLAGS.kl_coeff = FLAGS_OLD.kl_coeff
        FLAGS.repel_im = FLAGS_OLD.repel_im
        FLAGS.save_interval = FLAGS_OLD.save_interval

        for key in dir(FLAGS):
            if "__" not in key:
                FLAGS_OLD[key] = getattr(FLAGS, key)

        FLAGS = FLAGS_OLD

    if FLAGS.dataset == "cifar10":
        model_fn = ResNetModel
    elif FLAGS.dataset == "stl":
        model_fn = ResNetModel
    elif FLAGS.dataset == "object":
        model_fn = CelebAModel
    elif FLAGS.dataset == "mnist":
        model_fn = MNISTModel
    elif FLAGS.dataset == "celeba":
        model_fn = CelebAModel
    elif FLAGS.dataset == "lsun":
        model_fn = CelebAModel
    elif FLAGS.dataset == "imagenet":
        model_fn = ImagenetModel
    else:
        assert False

    models = [model_fn(FLAGS).train() for i in range(FLAGS.ensembles)]
    models_ema = [model_fn(FLAGS).train() for i in range(FLAGS.ensembles)]

    torch.cuda.set_device(gpu)
    if FLAGS.cuda:
        models = [model.cuda(gpu) for model in models]
        model_ema = [model_ema.cuda(gpu) for model_ema in models_ema]

    if FLAGS.gpus > 1:
        sync_model(models)

    parameters = []
    for model in models:
        parameters.extend(list(model.parameters()))

    optimizer = Adam(parameters, lr=FLAGS.lr, betas=(0.0, 0.9), eps=1e-8)

    ema_model(models, models_ema, mu=0.0)

    logger = TensorBoardOutputFormat(logdir)

    it = FLAGS.resume_iter

    if not osp.exists(logdir):
        os.makedirs(logdir)

    checkpoint = None
    if FLAGS.resume_iter != 0:
        model_path = osp.join(logdir, "model_{}.pth".format(FLAGS.resume_iter))
        checkpoint = torch.load(model_path)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        for i, (model, model_ema) in enumerate(zip(models, models_ema)):
            model.load_state_dict(checkpoint['model_state_dict_{}'.format(i)])
            model_ema.load_state_dict(checkpoint['ema_model_state_dict_{}'.format(i)])


    print("New Values of args: ", FLAGS)

    pytorch_total_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print("Number of parameters for models", pytorch_total_params)

    train(models, models_ema, optimizer, logger, train_dataloader, FLAGS.resume_iter, logdir, FLAGS, gpu, best_inception)

def main():
    flags_dict = EasyDict()

    for key in dir(FLAGS):
        flags_dict[key] = getattr(FLAGS, key)

    if FLAGS.gpus > 1:
        mp.spawn(main_single, nprocs=FLAGS.gpus, args=(flags_dict,))
    else:
        main_single(0, flags_dict)


if __name__ == "__main__":
    main()
