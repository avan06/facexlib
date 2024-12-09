"""
From the CodeFormer project by author sczhou.
"""
import cv2
import os
import os.path as osp
import re
import numpy as np
from PIL import Image
import torch
from torch.hub import download_url_to_file, get_dir
from urllib.parse import urlparse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def download_from_url(url, file_name, save_dir=None):
    """
    Downloads a file from a URL, either using gdown for Google Drive links
    or a different method for other URLs.

    Parameters:
    url (str): The URL from which to download the file.
    file_name (str): The name of the file to be saved.
    save_dir (str, optional): The directory where the file will be saved. If None, the current directory is used.
    """
    if 'drive.google.com' in url:
        print(f"Using gdown to download {file_name} from Google Drive.")
        file_id = url.split("id=")[-1]
        download_pretrained_models({file_name: file_id}, save_dir, True)
    else:
        print(f"Using load_file_from_url to download {file_name}.")
        load_file_from_url(url, file_name=file_name, save_dir=save_dir)


def download_pretrained_models(file_ids, save_path_root, skip_existing=False):
    """
    Downloads pretrained models from Google Drive using file IDs.
    
    Parameters:
    file_ids (dict): A dictionary where keys are file names and values are the corresponding Google Drive file IDs.
    save_path_root (str): The directory where the files will be saved.
    skip_existing (bool): If True, skips downloading files that already exist in the target directory.
    """
    import os
    import os.path as osp
    import gdown

    os.makedirs(save_path_root, exist_ok=True)

    for file_name, file_id in file_ids.items():
        file_url = 'https://drive.google.com/uc?id=' + file_id
        save_path = osp.abspath(osp.join(save_path_root, file_name))
        
        if osp.exists(save_path):
            if skip_existing:
                print(f'Skipping {file_name} as it already exists.')
                continue

            user_response = input(f'{file_name} already exists. Do you want to overwrite it? Y/N')
            if user_response.lower() == 'y':
                print(f'Overwriting {file_name} to {save_path}')
                gdown.download(file_url, save_path, quiet=False)
            elif user_response.lower() == 'n':
                print(f'Skipping {file_name}')
            else:
                raise ValueError('Invalid input. Only accepts Y/N.')
        else:
            print(f'Downloading {file_name} to {save_path}')
            gdown.download(file_url, save_path, quiet=False)


def imwrite(img, file_path, params=None, auto_mkdir=True):
    """Write image to file.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    return cv2.imwrite(file_path, img, params)


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def load_file_from_url(url, model_dir=None, progress=True, file_name=None, save_dir=None):
    """Ref:https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py
    """
    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    if save_dir is None:
        save_dir = os.path.join(ROOT_DIR, model_dir)
    os.makedirs(save_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.abspath(os.path.join(save_dir, filename))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.
    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.
    Returns:
        A generator for all the interested files with relative paths.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def is_gray(img, threshold=10):
    img = Image.fromarray(img)
    if len(img.getbands()) == 1:
        return True
    img1 = np.asarray(img.getchannel(channel=0), dtype=np.int16)
    img2 = np.asarray(img.getchannel(channel=1), dtype=np.int16)
    img3 = np.asarray(img.getchannel(channel=2), dtype=np.int16)
    diff1 = (img1 - img2).var()
    diff2 = (img2 - img3).var()
    diff3 = (img3 - img1).var()
    diff_sum = (diff1 + diff2 + diff3) / 3.0
    if diff_sum <= threshold:
        return True
    else:
        return False

def rgb2gray(img, out_channel=3):
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    if out_channel == 3:
        gray = gray[:,:,np.newaxis].repeat(3, axis=2)
    return gray

def bgr2gray(img, out_channel=3):
    b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    if out_channel == 3:
        gray = gray[:,:,np.newaxis].repeat(3, axis=2)
    return gray


def calc_mean_std(feat, eps=1e-5):
    """
    Args:
        feat (numpy): 3D [w h c]s
    """
    size = feat.shape
    assert len(size) == 3, 'The input feature should be 3D tensor.'
    c = size[2]
    feat_var = feat.reshape(-1, c).var(axis=0) + eps
    feat_std = np.sqrt(feat_var).reshape(1, 1, c)
    feat_mean = feat.reshape(-1, c).mean(axis=0).reshape(1, 1, c)
    return feat_mean, feat_std


def adain_npy(content_feat, style_feat):
    """Adaptive instance normalization for numpy.

    Args:
        content_feat (numpy): The input feature.
        style_feat (numpy): The reference feature.
    """
    size = content_feat.shape
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - np.broadcast_to(content_mean, size)) / np.broadcast_to(content_std, size)
    return normalized_feat * np.broadcast_to(style_std, size) + np.broadcast_to(style_mean, size)

IS_HIGH_VERSION = [int(m) for m in list(re.findall(r"^([0-9]+)\.([0-9]+)\.([0-9]+)([^0-9][a-zA-Z0-9]*)?(\+git.*)?$",\
    torch.__version__)[0][:3])] >= [1, 12, 0]

def get_device(gpu_id=None):
    """
    From the CodeFormer project by author sczhou.
    CodeFormer/basicsr/utils/misc.py
    """
    if gpu_id is None:
        gpu_str = ''
    elif isinstance(gpu_id, int):
        gpu_str = f':{gpu_id}'
    else:
        raise TypeError('Input should be int value.')

    if IS_HIGH_VERSION:
        if torch.backends.mps.is_available():
            return torch.device('mps'+gpu_str)
    return torch.device('cuda'+gpu_str if torch.cuda.is_available() and torch.backends.cudnn.is_available() else 'cpu')