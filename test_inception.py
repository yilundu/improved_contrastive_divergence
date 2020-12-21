import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.python.platform import flags
from models import ResNetModel, CelebAModel, MNISTModel
from data import Mnist
import os.path as osp
import os
from tqdm import tqdm
import torch
import random
from scipy.misc import imsave
from data import Cifar10, CelebAHQ
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2

from inception import get_inception_score
from fid import get_fid_score

flags.DEFINE_string('logdir', 'cachedir', 'location where log of experiments will be stored')
flags.DEFINE_string('exp', 'default', 'name of experiments')

# Architecture settings
flags.DEFINE_bool('multiscale', False, 'A multiscale EBM')
flags.DEFINE_float('step_lr', 10.0, 'Size of steps for gradient descent')
flags.DEFINE_integer('num_steps', 10, 'number of steps to optimize the label')
flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_integer('resume_iter', -1, 'resume iteration')
flags.DEFINE_integer('ensemble', 10, 'number of ensembles')
flags.DEFINE_integer('im_number', 1000, 'number of ensembles')
flags.DEFINE_integer('repeat_scale', 50, 'number of repeat iterations')
flags.DEFINE_float('noise_scale', 0.001, 'amount of noise to output')
flags.DEFINE_integer('idx', 0, 'save index')
flags.DEFINE_integer('nomix', 30, 'number of intervals to stop mixing')
flags.DEFINE_bool('scaled', True, 'whether to scale noise added')
flags.DEFINE_bool('ema', False, 'whether to scale noise added')
flags.DEFINE_bool('norm', True, 'whether to add normalization to model')
flags.DEFINE_string('dataset', 'cifar10', 'cifar10 or imagenet or imagenetfull')
flags.DEFINE_bool('transform', True, 'remove transform')
flags.DEFINE_bool('self_attn', True, 'Use self attention in models')
flags.DEFINE_bool('anneal', False, 'Anneal the step size when sampling with Langevin')

FLAGS = flags.FLAGS


def gen_image(label, FLAGS, model, im_neg, num_steps):
    im_noise = torch.randn_like(im_neg).detach()
    im_neg = im_neg.contiguous()

    for i in range(num_steps):
        im_noise.normal_()

        if FLAGS.anneal:
            im_neg = im_neg + FLAGS.noise_scale * (num_steps - 1 - i) / (num_steps - 1) * im_noise
        else:
            im_neg = im_neg + FLAGS.noise_scale * im_noise

        im_neg.requires_grad_(requires_grad=True)
        energy = model.forward(im_neg, None)
        im_grad = torch.autograd.grad([energy.sum()], [im_neg])[0]

        im_neg = im_neg - FLAGS.step_lr * im_grad
        im_neg = im_neg.detach()

        im_neg = torch.clamp(im_neg, 0, 1)

    return im_neg, energy


class GaussianBlur(object):

    def __init__(self, min=0.1, max=2.0, kernel_size=9):
        self.min = min
        self.max = max
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample


class InceptionReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._label_storage = []
        self._maxsize = size
        self._next_idx = 0

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

        if FLAGS.dataset == "celeba":
            self.transform = transforms.Compose([transforms.RandomResizedCrop(128, scale=(0.08, 1.0)), transforms.RandomHorizontalFlip(), color_transform, transforms.ToTensor()])
        elif FLAGS.dataset == "mnist":
            self.transform = transforms.ToTensor()
        else:
            self.transform = transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.08, 1.0)), transforms.RandomHorizontalFlip(), color_transform, GaussianBlur(kernel_size=5), transforms.ToTensor()])

    def __len__(self):
        return len(self._storage)

    def add(self, ims, labels):
        batch_size = ims.shape[0]
        if self._next_idx >= len(self._storage):
            self._storage.extend(list(ims))
            self._label_storage.extend(list(labels))
        else:
            if batch_size + self._next_idx < self._maxsize:
                self._storage[self._next_idx:self._next_idx+batch_size] = list(ims)
                self._label_storage[self._next_idx:self._next_idx+batch_size] = list(labels)
            else:
                split_idx = self._maxsize - self._next_idx
                self._storage[self._next_idx:] = list(ims)[:split_idx]
                self._storage[:batch_size-split_idx] = list(ims)[split_idx:]
                self._label_storage[self._next_idx:] = list(labels)[:split_idx]
                self._label_storage[:batch_size-split_idx] = list(labels)[split_idx:]

        self._next_idx = (self._next_idx + ims.shape[0]) % self._maxsize

    def _encode_sample(self, idxes, transform=True):
        ims = []
        labels = []
        for i in idxes:
            im = self._storage[i]

            if transform and (FLAGS.dataset != "mnist"):
                im = im.transpose((1, 2, 0))
                im = np.array(self.transform(Image.fromarray(np.array(im * 255, dtype=np.uint8))))

            ims.append(im)
            labels.append(self._label_storage[i])

        return np.array(ims), np.array(labels)

    def sample(self, batch_size, transform=True):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes, transform=transform), idxes

    def set_elms(self, idxes, data, labels):
        for i, ix in enumerate(idxes):
            self._storage[ix] = data[i]
            self._label_storage[ix] = labels[i]


def rescale_im(im):
    return np.clip(im * 256, 0, 255).astype(np.uint8)

def compute_inception(model):
    size = FLAGS.im_number
    num_steps = size // 1000

    images = []
    test_ims = []

    if FLAGS.dataset == "cifar10":
        test_dataset = Cifar10(FLAGS)
    elif FLAGS.dataset == "celeba":
        test_dataset = CelebAHQ()
    elif FLAGS.dataset == "mnist":
        test_dataset = Mnist(train=True)

    test_dataloader = DataLoader(test_dataset, batch_size=FLAGS.batch_size, num_workers=4, shuffle=True, drop_last=False)

    if FLAGS.dataset == "cifar10":
        for data_corrupt, data, label_gt in tqdm(test_dataloader):
            data = data.numpy()
            test_ims.extend(list(rescale_im(data)))

            if len(test_ims) > 10000:
                break
    elif FLAGS.dataset == "mnist":
        for data_corrupt, data, label_gt in tqdm(test_dataloader):
            data = data.numpy()
            test_ims.extend(list(np.tile(rescale_im(data), (1, 1, 3))))

            if len(test_ims) > 10000:
                break

    test_ims = test_ims[:10000]

    classes = 10

    print(FLAGS.batch_size)
    data_buffer = None

    for j in range(num_steps):
        itr = int(1000 / 500 * FLAGS.repeat_scale)

        if data_buffer is None:
            data_buffer = InceptionReplayBuffer(1000)

        curr_index = 0

        identity = np.eye(classes)

        if FLAGS.dataset == "celeba":
            n = 128
            c = 3
        elif FLAGS.dataset == "mnist":
            n = 28
            c = 1
        else:
            n = 32
            c = 3

        for i in tqdm(range(itr)):
            noise_scale = [1]
            if len(data_buffer) < 1000:
                x_init = np.random.uniform(0, 1, (FLAGS.batch_size, c, n, n))
                label = np.random.randint(0, classes, (FLAGS.batch_size))

                x_init = torch.Tensor(x_init).cuda()
                label = identity[label]
                label = torch.Tensor(label).cuda()

                x_new, _ = gen_image(label, FLAGS, model, x_init, FLAGS.num_steps)
                x_new = x_new.detach().cpu().numpy()
                label = label.detach().cpu().numpy()
                data_buffer.add(x_new, label)
            else:
                if i < itr - FLAGS.nomix:
                    (x_init, label), idx = data_buffer.sample(FLAGS.batch_size, transform=FLAGS.transform)
                else:
                    if FLAGS.dataset == "celeba":
                        n = 20
                    else:
                        n = 2

                    ix = i % n
                    # for i in range(n):
                    start_idx = (1000 // n) * ix
                    end_idx = (1000 // n) * (ix+1)
                    (x_init, label) = data_buffer._encode_sample(list(range(start_idx, end_idx)), transform=False)
                    idx = list(range(start_idx, end_idx))

                x_init = torch.Tensor(x_init).cuda()
                label = torch.Tensor(label).cuda()
                x_new, energy = gen_image(label, FLAGS, model, x_init, FLAGS.num_steps)
                energy = energy.cpu().detach().numpy()
                x_new = x_new.cpu().detach().numpy()
                label = label.cpu().detach().numpy()
                data_buffer.set_elms(idx, x_new, label)

                if FLAGS.im_number != 50000:
                    print(np.mean(energy), np.std(energy))

            curr_index += 1

        ims = np.array(data_buffer._storage[:1000])
        ims = rescale_im(ims).transpose((0, 2, 3, 1))

        if FLAGS.dataset == "mnist":
            ims = np.tile(ims, (1, 1, 1, 3))

        images.extend(list(ims))

    random.shuffle(images)
    saveim = osp.join('sandbox_cachedir', FLAGS.exp, "test{}.png".format(FLAGS.idx))


    if FLAGS.dataset == "cifar10":
        rix = np.random.permutation(1000)[:100]
        ims = ims[rix]
        im_panel = ims.reshape((10, 10, 32, 32, 3)).transpose((0, 2, 1, 3, 4)).reshape((320, 320, 3))
        imsave(saveim, im_panel)

        print("Saved image!!!!")
        splits = max(1, len(images) // 5000)
        score, std = get_inception_score(images, splits=splits)
        print("Inception score of {} with std of {}".format(score, std))

        # FID score
        n = min(len(images), len(test_ims))
        fid = get_fid_score(images, test_ims)
        print("FID of score {}".format(fid))

    elif FLAGS.dataset == "mnist":
        # ims = ims[:100]
        # im_panel = ims.reshape((10, 10, 32, 32, 3)).transpose((0, 2, 1, 3, 4)).reshape((320, 320, 3))
        # imsave(saveim, im_panel)

        ims = ims[:100]
        im_panel = ims.reshape((10, 10, 28, 28, 3)).transpose((0, 2, 1, 3, 4)).reshape((280, 280, 3))
        imsave(saveim, im_panel)

        print("Saved image!!!!")
        splits = max(1, len(images) // 5000)
        # score, std = get_inception_score(images, splits=splits)
        # print("Inception score of {} with std of {}".format(score, std))

        # FID score
        n = min(len(images), len(test_ims))
        fid = get_fid_score(images, test_ims)
        print("FID of score {}".format(fid))

    elif FLAGS.dataset == "celeba":

        ims = ims[:25]
        im_panel = ims.reshape((5, 5, 128, 128, 3)).transpose((0, 2, 1, 3, 4)).reshape((5*128, 5*128, 3))
        imsave(saveim, im_panel)




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
    elif FLAGS.dataset == "mnist":
        model = MNISTModel(FLAGS_model).eval().cuda()
    else:
        model = ResNetModel(FLAGS_model).eval().cuda()

    if FLAGS.ema:
        model.load_state_dict(checkpoint['ema_model_state_dict_0'])
    else:
        model.load_state_dict(checkpoint['model_state_dict_0'])

    logdir = osp.join(FLAGS.logdir, FLAGS.exp)
    model = model.eval()
    compute_inception(model)


if __name__ == "__main__":
    main()
