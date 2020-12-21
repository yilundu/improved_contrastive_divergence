import tensorflow as tf
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from imageio import imwrite
import math
from tqdm import tqdm
from tensorflow.python.platform import flags
from torch.utils.data import DataLoader
import torch
from models import ResNetModel, CelebAModel, ModelLinear
from data import Cifar10, CelebAHQ, Svhn, Cifar100, Textures, CelebaSmall
import os.path as osp
import numpy as np
from baselines.logger import TensorBoardOutputFormat
from scipy.misc import imsave
import os
from baselines.common.tf_util import initialize
import matplotlib.pyplot as plt
import sklearn.metrics as sk
from utils import accuracy

# set_seed(1)
flags.DEFINE_string('logdir', 'cachedir', 'location where log of experiments will be stored')
flags.DEFINE_string('exp', 'default', 'name of experiments')

# Architecture settings
flags.DEFINE_bool('multiscale', False, 'A multiscale EBM')
flags.DEFINE_float('step_lr', 10.0, 'Size of steps for gradient descent')
flags.DEFINE_integer('num_steps', 10, 'number of steps to optimize the label')
flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_integer('resume_iter', -1, 'resume iteration')
flags.DEFINE_integer('repeat_scale', 50, 'number of repeat iterations')
flags.DEFINE_bool('ema', True, 'whether to scale noise added')
flags.DEFINE_bool('random_init', False, 'initialize models from model')
flags.DEFINE_string('dataset', 'cifar10', 'cifar10 or imagenet or imagenetfull')

flags.DEFINE_string('task', 'unsup_finetune', 'unsup_finetune: unsupervised finetuning of models),'
                    'or mixenergy to evaluate out of distribution generalization compared to other datasets')
flags.DEFINE_integer('data_workers', 5, 'Number of different data workers to load data in parallel')


FLAGS = flags.FLAGS

def rescale_im(im):
    im = np.clip(im, 0, 1)
    return np.round(im * 255).astype(np.uint8)

def label(dataloader, test_dataloader, target_vars, sess, l1val=8, l2val=40):
    X = target_vars['X']
    Y = target_vars['Y']
    Y_GT = target_vars['Y_GT']
    accuracy = target_vars['accuracy']
    train_op = target_vars['train_op']
    l1_norm = target_vars['l1_norm']
    l2_norm = target_vars['l2_norm']

    label_init = np.random.uniform(0, 1, (FLAGS.batch_size, 10))
    label_init = label_init / label_init.sum(axis=1, keepdims=True)

    label_init = np.tile(np.eye(10)[None :, :], (FLAGS.batch_size, 1, 1))
    label_init = np.reshape(label_init, (-1, 10))

    for i in range(1):
        emp_accuracies = []

        for data_corrupt, data, label_gt in tqdm(test_dataloader):
            feed_dict = {X: data, Y_GT: label_gt, Y: label_init, l1_norm: l1val, l2_norm: l2val}
            emp_accuracy = sess.run([accuracy], feed_dict)
            emp_accuracies.append(emp_accuracy)
            print(np.array(emp_accuracies).mean())

        print("Received total accuracy of {} for li of {} and l2 of {}".format(np.array(emp_accuracies).mean(), l1val, l2val))

    return np.array(emp_accuracies).mean()


def labelfinetune(dataloader, test_dataloader, target_vars, sess, savedir, saver, l1val=8, l2val=40):
    X = target_vars['X']
    Y = target_vars['Y']
    Y_GT = target_vars['Y_GT']
    accuracy = target_vars['accuracy']
    train_op = target_vars['train_op']
    l1_norm = target_vars['l1_norm']
    l2_norm = target_vars['l2_norm']

    label_init = np.random.uniform(0, 1, (FLAGS.batch_size, 10))
    label_init = label_init / label_init.sum(axis=1, keepdims=True)

    label_init = np.tile(np.eye(10)[None :, :], (FLAGS.batch_size, 1, 1))
    label_init = np.reshape(label_init, (-1, 10))

    itr = 0

    if FLAGS.train:
        for i in range(1):
            for data_corrupt, data, label_gt in tqdm(dataloader):
                feed_dict = {X: data, Y_GT: label_gt, Y: label_init}
                acc, _ = sess.run([accuracy, train_op], feed_dict)

                itr += 1

                if itr % 10 == 0:
                    print(acc)

        saver.save(sess, osp.join(savedir, "model_supervised"))

    saver.restore(sess, osp.join(savedir, "model_supervised"))


    for i in range(1):
        emp_accuracies = []

        for data_corrupt, data, label_gt in tqdm(test_dataloader):
            feed_dict = {X: data, Y_GT: label_gt, Y: label_init, l1_norm: l1val, l2_norm: l2val}
            emp_accuracy = sess.run([accuracy], feed_dict)
            emp_accuracies.append(emp_accuracy)
            print(np.array(emp_accuracies).mean())


        print("Received total accuracy of {} for li of {} and l2 of {}".format(np.array(emp_accuracies).mean(), l1val, l2val))

    return np.array(emp_accuracies).mean()

def unsup_finetune(model, FLAGS_model):
    train_dataset = Cifar10(FLAGS, train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, num_workers=FLAGS.data_workers, shuffle=True, drop_last=False)

    classifier = ModelLinear(FLAGS_model).cuda()
    optimizer = optim.Adam(classifier.parameters(),
                          lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    running_accuracy = []

    count = 0
    for i in range(100):
        for data_corrupt, data, label_gt in tqdm(train_dataloader):
            data = data.permute(0, 3, 1, 2).float().cuda()
            target = label_gt.long().cuda()

            with torch.no_grad():
                model_feat = model.compute_feat(data, None)
                model_feat = F.normalize(model_feat, dim=-1, p=2)

            output = classifier(model_feat)

            loss = criterion(output, target)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            running_accuracy.append(acc1.item())
            running_accuracy = running_accuracy[-10:]
            print(count, loss, np.mean(running_accuracy))

            count += 1


def energyevalmix(model):
    # dataset = Cifar100(FLAGS, train=False)
    # dataset = Svhn(train=False)
    # dataset = Textures(train=True)
    # dataset = Cifar10(FLAGS, train=False)
    dataset = CelebaSmall()
    test_dataloader = DataLoader(dataset, batch_size=FLAGS.batch_size, num_workers=FLAGS.data_workers, shuffle=True, drop_last=False)

    train_dataset = Cifar10(FLAGS, train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, num_workers=FLAGS.data_workers, shuffle=True, drop_last=False)

    test_iter = iter(test_dataloader)

    probs = []
    labels = []
    negs = []
    pos = []
    for data_corrupt, data, label_gt in tqdm(train_dataloader):
        _, data_mix, _ = test_iter.next()
        # data_mix = data_mix[:data.shape[0]]
        # data_mix_permute = torch.cat([data_mix[1:data.shape[0]], data_mix[:1]], dim=0)
        # data_mix = (data_mix + data_mix_permute) / 2.

        data = data.permute(0, 3, 1, 2).float().cuda()
        data_mix = data_mix.permute(0, 3, 1, 2).float().cuda()

        pos_energy = model.forward(data, None).detach().cpu().numpy().mean(axis=-1)
        neg_energy = model.forward(data_mix, None).detach().cpu().numpy().mean(axis=-1)
        print("pos_energy", pos_energy.mean())
        print("neg_energy", neg_energy.mean())

        probs.extend(list(-1*pos_energy))
        probs.extend(list(-1*neg_energy))
        pos.extend(list(-1*pos_energy))
        negs.extend(list(-1*neg_energy))
        labels.extend([1]*pos_energy.shape[0])
        labels.extend([0]*neg_energy.shape[0])

    pos, negs = np.array(pos), np.array(negs)
    np.save("pos.npy", pos)
    np.save("neg.npy", negs)
    auroc = sk.roc_auc_score(labels, probs)
    print("Roc score of {}".format(auroc))


def nearest_neighbor(dataset, sess, target_vars, logdir):
    X = target_vars['X']
    Y_GT = target_vars['Y_GT']
    x_final = target_vars['X_final']

    noise = np.random.uniform(0, 1, size=[10, 32, 32, 3])
    # label = np.random.randint(0, 10, size=[10])
    label = np.eye(10)

    coarse = noise

    for i in range(10):
        x_new = sess.run([x_final], {X:coarse, Y_GT:label})[0]
        coarse = x_new

    x_new_dense = x_new.reshape(10, 1, 32*32*3)
    dataset_dense = dataset.reshape(1, 50000, 32*32*3)

    diff = np.square(x_new_dense - dataset_dense).sum(axis=2)
    diff_idx = np.argsort(diff, axis=1)

    panel = np.zeros((32*10, 32*6, 3))

    dataset_rescale = rescale_im(dataset)
    x_new_rescale = rescale_im(x_new)

    for i in range(10):
        panel[i*32:i*32+32, :32] = x_new_rescale[i]
        for j in range(5):
            panel[i*32:i*32+32, 32*j+32:32*j+64] = dataset_rescale[diff_idx[i, j]]

    imsave(osp.join(logdir, "nearest.png"), panel)


def main():

    logdir = osp.join(FLAGS.logdir, FLAGS.exp)
    sandbox_logdir = osp.join('sandbox_cachedir', FLAGS.exp)

    if not osp.exists(sandbox_logdir):
        os.makedirs(sandbox_logdir)

    model_path = osp.join(logdir, "model_{}.pth".format(FLAGS.resume_iter))
    checkpoint = torch.load(model_path)
    FLAGS_model = checkpoint['FLAGS']

    if FLAGS.dataset == "celeba":
        model = CelebAModel(FLAGS_model).eval().cuda()
    else:
        model = ResNetModel(FLAGS_model).eval().cuda()

    if not FLAGS.random_init:
        if FLAGS.ema:
            model.load_state_dict(checkpoint['ema_model_state_dict_0'])
        else:
            model.load_state_dict(checkpoint['model_state_dict_0'])

    logdir = osp.join(FLAGS.logdir, FLAGS.exp)
    model = model.eval()

    if FLAGS.task == 'mixenergy':
        energyevalmix(model)
    if FLAGS.task == 'unsup_finetune':
        unsup_finetune(model, FLAGS_model)


if __name__ == "__main__":
    main()
