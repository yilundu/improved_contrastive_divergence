import torch
import math
from tensorflow.python.platform import flags
from torch.utils.data import DataLoader
from data import Cifar10, CelebAHQ
from models import ResNetModel, CelebAModel
from scipy.misc import logsumexp
from scipy.misc import imsave
import os.path as osp
import numpy as np
from tqdm import tqdm
from hmc import gen_hmc_image

flags.DEFINE_string('logdir', 'cachedir', 'location where log of experiments will be stored')
flags.DEFINE_string('exp', 'default', 'name of experiments')
flags.DEFINE_integer('data_workers', 4, 'Number of different data workers to load data in parallel')
flags.DEFINE_integer('batch_size', 16, 'Size of inputs')
flags.DEFINE_string('resume_iter', '-1', 'iteration to resume training from')

flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omniglot.')
flags.DEFINE_integer('pdist', 10, 'number of intermediate distributions for ais')
flags.DEFINE_integer('rescale', 1, 'factor to rescale input outside of normal (0, 1) box')
flags.DEFINE_float('temperature', 1, 'temperature at which to compute likelihood of model')
flags.DEFINE_bool('single', False, 'Whether to evaluate the log likelihood of conditional model or not')
flags.DEFINE_bool('large_model', False, 'Use large model to evaluate')
flags.DEFINE_bool('wider_model', False, 'Use large model to evaluate')
flags.DEFINE_float('alr', 0.0045, 'Learning rate to use for HMC steps')
flags.DEFINE_bool('ema', True, 'whether to scale noise added')

FLAGS = flags.FLAGS


def unscale_im(im):
    return (255 * np.clip(im, 0, 1)).astype(np.uint8)

def gauss_prob_log(x, prec=1.0):
    nh = np.prod(x.size()[1:])
    norm_constant_log = -0.5 * (torch.log(2 * math.pi) * nh - nh * tf.log(prec))
    prob_density_log = -tf.reduce_sum(tf.square(x - 0.5), axis=[1]) / 2. * prec

    return norm_constant_log + prob_density_log


def uniform_prob_log(x):
    return torch.zeros(1).to(x.device)


def model_prob_log(x, e_func, temp):
    e_raw = e_func(x, None)
    energy = e_raw.sum(dim=-1)
    return -temp * energy


def bridge_prob_neg_log(alpha, x, model, temp):

    norm_prob =  (1-alpha) * uniform_prob_log(x) + alpha * model_prob_log(x, model, temp)
    # Add an additional log likelihood penalty so that points outside of (0, 1) box are *highly* unlikely

    return -norm_prob


def ancestral_sample(model, x, alpha_prev, alpha_new, FLAGS, batch_size=128, prop_dist=10, temp=1, hmc_step=10, approx_lr=10.0):
    x_init = x

    chain_weights = torch.zeros(x.size(0))
    # for i in range(1, prop_dist+1):
    #     print("processing loop {}".format(i))
    #     alpha_prev = (i-1) / prop_dist
    #     alpha_new = i / prop_dist

    prob_log_old_neg = bridge_prob_neg_log(alpha_prev, x, model, temp)
    prob_log_new_neg = bridge_prob_neg_log(alpha_new, x, model, temp)

    chain_weights = -prob_log_new_neg + prob_log_old_neg
    # chain_weights = tf.Print(chain_weights, [chain_weights])

    # Sample new x using HMC
    def unorm_prob(x):
        return bridge_prob_neg_log(alpha_new, x, model, temp)

    for j in range(1):
        x = gen_hmc_image(x, approx_lr, FLAGS.temperature, unorm_prob, num_steps=hmc_step)

    return chain_weights, x


def main():

    # Initialize dataset
    dataset = Cifar10(FLAGS, train=False, rescale=FLAGS.rescale)
    data_loader = DataLoader(dataset, batch_size=FLAGS.batch_size, num_workers=4, drop_last=False, shuffle=True)

    logdir = osp.join(FLAGS.logdir, FLAGS.exp)
    model_path = osp.join(logdir, "model_best.pth")
    checkpoint = torch.load(model_path)
    FLAGS_model = checkpoint['FLAGS']
    model = ResNetModel(FLAGS_model).eval().cuda()

    if FLAGS.ema:
        model.load_state_dict(checkpoint['ema_model_state_dict_0'])
    else:
        model.load_state_dict(checkpoint['model_state_dict_0'])

    print("Finished constructing ancestral sample ...................")
    e_pos_list = []

    for data_corrupt, data, label_gt in tqdm(data_loader):
        data = data.permute(0, 3, 1, 2).contiguous().cuda()
        energy = model.forward(data, None)
        energy = -FLAGS.temperature * energy.squeeze().detach().cpu().numpy()
        e_pos_list.extend(list(energy))

    print(len(e_pos_list))
    print("Positive sample probability ", np.mean(e_pos_list), np.std(e_pos_list))

    # alr = 0.0065
    alr = 10.0
# 
    for i in range(1):
        tot_weight = 0
        for j in tqdm(range(1, FLAGS.pdist+1)):

            if j == 1:
                x_curr =  torch.rand(FLAGS.batch_size, 3, 32, 32).cuda()

            alpha_prev = (j-1) / FLAGS.pdist
            alpha_new = j / FLAGS.pdist
            cweight, x_curr = ancestral_sample(model, x_curr, alpha_prev, alpha_new, FLAGS, FLAGS.batch_size, FLAGS.pdist, temp=FLAGS.temperature, approx_lr=alr)
            tot_weight = tot_weight + cweight.detach()
            x_curr = x_curr.detach()

        tot_weight = tot_weight.detach().cpu().float().numpy()
        print("Total values of lower value based off forward sampling", np.mean(tot_weight), np.std(tot_weight))

        tot_weight = 0
        x_curr = x_curr.detach()

        for j in tqdm(range(FLAGS.pdist, 0, -1)):
            alpha_new = (j-1) / FLAGS.pdist
            alpha_prev = j / FLAGS.pdist
            cweight, x_curr = ancestral_sample(model, x_curr, alpha_prev, alpha_new, FLAGS, FLAGS.batch_size, FLAGS.pdist, temp=FLAGS.temperature, approx_lr=alr)
            tot_weight = tot_weight - cweight.detach()
            x_curr = x_curr.detach()

        tot_weight = tot_weight.detach().cpu().float().numpy()
        print("Total values of upper value based off backward sampling", np.mean(tot_weight), np.std(tot_weight))



if __name__ == "__main__":
    main()
