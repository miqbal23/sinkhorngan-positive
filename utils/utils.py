import json
import os
from collections import OrderedDict
from pathlib import Path

import imageio
import numpy as np
import scipy.misc


def try_param_set(dic, key, value=None):
    try:
        return dic[key]
    except KeyError:
        return value


def print_wline(texts="", line="=", size=100):
    texts = " " + texts + " " if len(texts) > 0 else ""
    lenLine = (size - len(texts)) // 2
    odd = (size - len(texts)) % 2
    print("")
    print((line * lenLine) + texts + (line * (lenLine + odd)))
    print("")


def delete_max_save(path, maxSave):
    list_model = os.listdir(path)
    list_model = sorted(list_model)
    while (len(list_model) > maxSave):
        model_del = os.path.join(path, list_model.pop(0))
        os.remove(model_del)


def get_latest_model(path, pre_file):
    files = os.listdir(path)
    files = [f for f in files if pre_file in f]
    if len(files) < 1:
        raise ValueError('Please rename your model to {}'.format(pre_file))
    files = sorted(files, reverse=True)
    return os.path.join(path, files[0])


def add_none_params(params, list_added):
    """
    Some time we get error when get the non exist keys, so add it using this code
    :param params: params
    :param list_added: list keys you want to add
    :return:
    """
    for p in list_added:
        try:
            params[p]
        except KeyError:
            params[p] = None
    return params


def makedir_if_notexist_w_join(*args):
    path = os.path.join(*args)
    if not os.path.exists(path):
        os.makedirs(path)


def get_latest_dir(dir, only_timestamp=True):
    dirs = os.listdir(dir)
    if len(dirs) == 0:
        return None
    dirs_ = []
    for d in dirs:
        try:
            dirs_.append(int(d))
        except ValueError:
            return None
    latest_dir = str(sorted(dirs_)[-1])
    return os.path.abspath(os.path.join(dir, latest_dir))


def makedir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def makedir_exist_ok(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)
        return False
    else:
        return True


def select_method_argument(method, in_dict):
    list_args = method.__code__.co_varnames
    new_dic = {}
    for key, value in in_dict.items():
        if key in list_args:
            new_dic[key] = value
    return new_dic


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def save_images(images, size, image_path):
    return imsave(images, size, image_path)


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3, 4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')


def generate_animation(path, num):
    images = []
    for e in range(num):
        img_name = path + '_epoch%03d' % (e + 1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(path + '_generate_animation.gif', images, fps=5)


def normalize(x, dim=1):
    return x.div(x.norm(2, dim=dim).expand_as(x))


def match(x, y, dist):
    '''
    Computes distance between corresponding points points in `x` and `y`
    using distance `dist`.
    '''
    if dist == 'L2':
        return (x - y).pow(2).mean()
    elif dist == 'L1':
        return (x - y).abs().mean()
    elif dist == 'cos':
        x_n = normalize(x)
        y_n = normalize(y)
        return 2 - (x_n).mul(y_n).mean()
    else:
        assert dist == 'none', 'wtf ?'
