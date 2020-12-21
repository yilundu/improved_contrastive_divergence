import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2


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


class ReplayBuffer(object):
    def __init__(self, size, transform, dataset):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self.gaussian_blur = GaussianBlur()

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

        if dataset == "cifar10":
            im_size = 32
        elif dataset == "continual":
            im_size = 64
        elif dataset == "celeba":
            im_size = 128
        elif dataset == "object":
            im_size = 128
        elif dataset == "mnist":
            im_size = 28
        elif dataset == "moving_mnist":
            im_size = 28
        elif dataset == "imagenet":
            im_size = 128
        elif dataset == "lsun":
            im_size = 128
        else:
            assert False

        self.dataset = dataset
        if transform:
            if dataset == "cifar10":
                self.transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.08, 1.0)), transforms.RandomHorizontalFlip(), color_transform, transforms.ToTensor()])
            elif dataset == "continual":
                color_transform = get_color_distortion(0.1)
                self.transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.7, 1.0)), color_transform, transforms.ToTensor()])
            elif dataset == "celeba":
                self.transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.08, 1.0)), transforms.RandomHorizontalFlip(), color_transform, transforms.ToTensor()])
            elif dataset == "imagenet":
                self.transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.01, 1.0)), transforms.RandomHorizontalFlip(), color_transform, transforms.ToTensor()])
            elif dataset == "object":
                self.transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.01, 1.0)), transforms.RandomHorizontalFlip(), color_transform, transforms.ToTensor()])
            elif dataset == "lsun":
                self.transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.08, 1.0)), transforms.RandomHorizontalFlip(), color_transform, transforms.ToTensor()])
            elif dataset == "mnist":
                self.transform = None
            elif dataset == "moving_mnist":
                self.transform = None
            else:
                assert False
        else:
            self.transform = None

    def __len__(self):
        return len(self._storage)

    def add(self, ims):
        batch_size = ims.shape[0]
        if self._next_idx >= len(self._storage):
            self._storage.extend(list(ims))
        else:
            if batch_size + self._next_idx < self._maxsize:
                self._storage[self._next_idx:self._next_idx +
                              batch_size] = list(ims)
            else:
                split_idx = self._maxsize - self._next_idx
                self._storage[self._next_idx:] = list(ims)[:split_idx]
                self._storage[:batch_size - split_idx] = list(ims)[split_idx:]
        self._next_idx = (self._next_idx + ims.shape[0]) % self._maxsize

    def _encode_sample(self, idxes, no_transform=False, downsample=False):
        ims = []
        for i in idxes:
            im = self._storage[i]

            if self.dataset != "mnist":
                if (self.transform is not None) and (not no_transform):
                    im = im.transpose((1, 2, 0))
                    im = np.array(self.transform(Image.fromarray(np.array(im))))

                # if downsample and (self.dataset in ["celeba", "object", "imagenet"]):
                #     im = im[:, ::4, ::4]

            im = im * 255
            ims.append(im)
        return np.array(ims)

    def sample(self, batch_size, no_transform=False, downsample=False):
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
        idxes = [random.randint(0, len(self._storage) - 1)
                 for _ in range(batch_size)]
        return self._encode_sample(idxes, no_transform=no_transform, downsample=downsample), idxes

    def set_elms(self, data, idxes):
        if len(self._storage) < self._maxsize:
            self.add(data)
        else:
            for i, ix in enumerate(idxes):
                self._storage[ix] = data[i]


class ReservoirBuffer(object):
    def __init__(self, size, transform, dataset):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self.n = 0

        def get_color_distortion(s=1.0):
        # s is the strength of color distortion.
            color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.4*s)
            rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
            rnd_gray = transforms.RandomGrayscale(p=0.2)
            color_distort = transforms.Compose([
                rnd_color_jitter,
                rnd_gray])
            return color_distort

        if dataset == "cifar10":
            im_size = 32
        elif dataset == "continual":
            im_size = 64
        elif dataset == "celeba":
            im_size = 128
        elif dataset == "object":
            im_size = 128
        elif dataset == "mnist":
            im_size = 28
        elif dataset == "moving_mnist":
            im_size = 28
        elif dataset == "imagenet":
            im_size = 128
        elif dataset == "lsun":
            im_size = 128
        elif dataset == "stl":
            im_size = 48
        else:
            assert False

        color_transform = get_color_distortion(0.5)
        self.dataset = dataset

        if transform:
            if dataset == "cifar10":
                color_transform = get_color_distortion(1.0)
                self.transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.08, 1.0)), transforms.RandomHorizontalFlip(), color_transform, transforms.ToTensor()])
                # self.transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.03, 1.0)), transforms.RandomHorizontalFlip(), color_transform, GaussianBlur(kernel_size=5), transforms.ToTensor()])
            elif dataset == "continual":
                self.transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.08, 1.0)), transforms.RandomHorizontalFlip(), color_transform, GaussianBlur(kernel_size=5), transforms.ToTensor()])
            elif dataset == "celeba":
                self.transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.08, 1.0)), transforms.RandomHorizontalFlip(), color_transform, GaussianBlur(kernel_size=5), transforms.ToTensor()])
            elif dataset == "imagenet":
                self.transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.6, 1.0)), transforms.RandomHorizontalFlip(), color_transform, GaussianBlur(kernel_size=11), transforms.ToTensor()])
            elif dataset == "lsun":
                self.transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.08, 1.0)), transforms.RandomHorizontalFlip(), color_transform, GaussianBlur(kernel_size=5), transforms.ToTensor()])
            elif dataset == "stl":
                self.transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.04, 1.0)), transforms.RandomHorizontalFlip(), color_transform, GaussianBlur(kernel_size=11), transforms.ToTensor()])
            elif dataset == "object":
                self.transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.08, 1.0)), transforms.RandomHorizontalFlip(), color_transform, transforms.ToTensor()])
            elif dataset == "mnist":
                self.transform = None
            elif dataset == "moving_mnist":
                self.transform = None
            else:
                assert False
        else:
            self.transform = None

    def __len__(self):
        return len(self._storage)

    def add(self, ims):
        batch_size = ims.shape[0]
        if self._next_idx >= len(self._storage):
            self._storage.extend(list(ims))
            self.n = self.n + ims.shape[0]
        else:
            for im in ims:
                self.n = self.n + 1
                ix = random.randint(0, self.n - 1)

                if ix < len(self._storage):
                    self._storage[ix] = im

        self._next_idx = (self._next_idx + ims.shape[0]) % self._maxsize


    def _encode_sample(self, idxes, no_transform=False, downsample=False):
        ims = []
        for i in idxes:
            im = self._storage[i]

            if self.dataset != "mnist":
                if (self.transform is not None) and (not no_transform):
                    im = im.transpose((1, 2, 0))
                    im = np.array(self.transform(Image.fromarray(im)))

                # if downsample and (self.dataset in ["celeba", "object", "imagenet"]):
                #     im = im[:, ::4, ::4]

            im = im * 255

            ims.append(im)
        return np.array(ims)

    def sample(self, batch_size, no_transform=False, downsample=False):
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
        idxes = [random.randint(0, len(self._storage) - 1)
                 for _ in range(batch_size)]
        return self._encode_sample(idxes, no_transform=no_transform, downsample=downsample), idxes


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by 0.2 every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def compute_jacobian_generic(y, x, create_graph=False):
    # Computes the jacobian by tiling values.
    # Assumes y is of shape n x d
    # Assumes x is of shape n x d also

    latent_dim = y.size(1)
    grad_y = torch.zeros_like(y)
    jacs = []

    for i in range(latent_dim):
        grad_y[:, i] = 1
        jac = torch.autograd.grad(y, x, grad_y, create_graph=create_graph, retain_graph=True)[0]
        jacs.append(jac)
        grad_y[:, i] = 0

    jacs = torch.stack(jacs, dim=1)
    return jacs


def compute_jacobian(model, im_feat, latent, optimize_partition=False, create_graph=False):
    # Computes the jacobian by tiling values.
    # Assumes y is of shape n x d
    # Assumes x is of shape n x d also
    latent_dim = model.energy_dim

    im_shape = im_feat.size()
    latent_shape = latent.size()
    im_feat_raw = im_feat

    im_feat = im_feat[:, None, :].repeat(1, latent_dim, 1).view(-1, im_shape[1])
    latent = latent[:, None, :].repeat(1, latent_dim, 1).view(-1, latent_shape[1])
    grad_y = torch.eye(latent_dim).to(im_feat.device)[None, :, :].repeat(im_shape[0], 1, 1)
    grad_y = grad_y.view(-1, latent_dim)
    energy = model.feat_energy(im_feat, latent)

    if optimize_partition:
        im_feat_raw = im_feat_raw[torch.randperm(im_feat_raw.size(0)).to(im_feat_raw.device)][:32]
        # im_feat_raw = im_feat_raw
        im_feat_partition = im_feat_raw[:, None, :].repeat(1, latent.size(0), 1)
        latent_neg_partition = latent[None, :, :].repeat(im_feat_raw.size(0), 1, 1)
        partition_est = model.feat_energy(im_feat_partition, latent_neg_partition)
        energy = energy + torch.logsumexp(-1 * partition_est, dim=0)

    jacs = torch.autograd.grad(energy, latent, grad_y, create_graph=create_graph)[0]
    s = jacs.size()
    # jacs = jacs.view(im_shape[0], -1)
    jacs_dense = jacs.view(im_shape[0], -1)
    scale_factor = torch.abs(jacs_dense).max(dim=-1, keepdim=True)[0]

    jacs = jacs_dense.view(im_shape[0], -1) / scale_factor
    jacs = jacs.view(im_shape[0], latent_dim, s[1])

    energy = energy.view(-1, latent_dim, latent_dim)
    energy = energy[:, 0, :]

    return jacs, scale_factor, energy


class WSConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(WSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
