"""
# Code borrowded from:
# https://github.com/zijundeng/pytorch-semantic-segmentation/blob/master/utils/transforms.py
#
#
# MIT License
#
# Copyright (c) 2017 ZijunDeng
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""

"""
Standard Transform
"""

import random
import numpy as np
from skimage.filters import gaussian
from skimage.restoration import denoise_bilateral
import torch
from PIL import Image, ImageEnhance
import torchvision.transforms as torch_tr
# from utils.confignvidia import cfg
from utils.attr_dict import AttrDict
from scipy.ndimage.interpolation import shift

from skimage.segmentation import find_boundaries
import torch.nn as nn
import torch.nn.functional as F

__C = AttrDict()
cfg = __C
__C.EPOCH = 0
# Use Class Uniform Sampling to give each class proper sampling
__C.CLASS_UNIFORM_PCT = 0.0

# Use class weighted loss per batch to increase loss for low pixel count classes per batch
__C.BATCH_WEIGHTING = False

# Border Relaxation Count
__C.BORDER_WINDOW = 1
# Number of epoch to use before turn off border restriction
__C.REDUCE_BORDER_EPOCH = -1
# Comma Seperated List of class id to relax
__C.STRICTBORDERCLASS = None



#Attribute Dictionary for Dataset
__C.DATASET = AttrDict()
#Cityscapes Dir Location
__C.DATASET.CITYSCAPES_DIR = ''
#SDC Augmented Cityscapes Dir Location
__C.DATASET.CITYSCAPES_AUG_DIR = ''
#Mapillary Dataset Dir Location
__C.DATASET.MAPILLARY_DIR = ''
#Kitti Dataset Dir Location
__C.DATASET.KITTI_DIR = ''
#SDC Augmented Kitti Dataset Dir Location
__C.DATASET.KITTI_AUG_DIR = ''
#Camvid Dataset Dir Location
__C.DATASET.CAMVID_DIR = ''
#Number of splits to support
__C.DATASET.CV_SPLITS = 3


__C.MODEL = AttrDict()
__C.MODEL.BN = 'regularnorm'
__C.MODEL.BNFUNC = None



try:
    import accimage
except ImportError:
    accimage = None


class RandomVerticalFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class MaskToTensor(object):
    def __call__(self, img, blockout_predefined_area=False):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()

class RelaxedBoundaryLossToTensor(object):
    """
    Boundary Relaxation
    """
    def __init__(self,ignore_id, num_classes):
        self.ignore_id=ignore_id
        self.num_classes= num_classes


    def new_one_hot_converter(self,a):
        ncols = self.num_classes+1
        out = np.zeros( (a.size,ncols), dtype=np.uint8)
        out[np.arange(a.size),a.ravel()] = 1
        out.shape = a.shape + (ncols,)
        return out

    def __call__(self,img):

        img_arr = np.array(img)
        img_arr[img_arr==self.ignore_id]=self.num_classes

        if cfg.STRICTBORDERCLASS != None:
            one_hot_orig = self.new_one_hot_converter(img_arr)
            mask = np.zeros((img_arr.shape[0],img_arr.shape[1]))
            for cls in cfg.STRICTBORDERCLASS:
                mask = np.logical_or(mask,(img_arr == cls))
        one_hot = 0

        border = cfg.BORDER_WINDOW
        if (cfg.REDUCE_BORDER_EPOCH !=-1 and cfg.EPOCH > cfg.REDUCE_BORDER_EPOCH):
            border = border // 2
            border_prediction = find_boundaries(img_arr, mode='thick').astype(np.uint8)

        for i in range(-border,border+1):
            for j in range(-border, border+1):
                shifted= shift(img_arr,(i,j), cval=self.num_classes)
                one_hot += self.new_one_hot_converter(shifted)

        one_hot[one_hot>1] = 1

        if cfg.STRICTBORDERCLASS != None:
            one_hot = np.where(np.expand_dims(mask,2), one_hot_orig, one_hot)

        one_hot = np.moveaxis(one_hot,-1,0)


        if (cfg.REDUCE_BORDER_EPOCH !=-1 and cfg.EPOCH > cfg.REDUCE_BORDER_EPOCH):
                one_hot = np.where(border_prediction,2*one_hot,1*one_hot)
                # print(one_hot.shape)
#        return torch.from_numpy(one_hot).byte()
        return one_hot.astype(np.uint8)

class ResizeHeight(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.target_h = size
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        target_w = int(w / h * self.target_h)
        return img.resize((target_w, self.target_h), self.interpolation)


class FreeScale(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = tuple(reversed(size))  # size: (h, w)
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize(self.size, self.interpolation)


class FlipChannels(object):
    """
    Flip around the x-axis
    """
    def __call__(self, img):
        img = np.array(img)[:, :, ::-1]
        return Image.fromarray(img.astype(np.uint8))


class RandomGaussianBlur(object):
    """
    Apply Gaussian Blur
    """
    def __call__(self, img):
        sigma = 0.15 + random.random() * 1.15
        blurred_img = gaussian(np.array(img), sigma=sigma, multichannel=True)
        blurred_img *= 255
        return Image.fromarray(blurred_img.astype(np.uint8))


class RandomBrightness(object):
    def __call__(self, img):
        if random.random() < 0.5:
            return img
        v = random.uniform(0.1, 1.9)
        return ImageEnhance.Brightness(img).enhance(v)


class RandomBilateralBlur(object):
    """
    Apply Bilateral Filtering

    """
    def __call__(self, img):
        sigma = random.uniform(0.05, 0.75)
        blurred_img = denoise_bilateral(np.array(img), sigma_spatial=sigma, multichannel=True)
        blurred_img *= 255
        return Image.fromarray(blurred_img.astype(np.uint8))


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def adjust_brightness(img, brightness_factor):
    """Adjust brightness of an Image.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.

    Returns:
        PIL Image: Brightness adjusted image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img


def adjust_contrast(img, contrast_factor):
    """Adjust contrast of an Image.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.

    Returns:
        PIL Image: Contrast adjusted image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img


def adjust_saturation(img, saturation_factor):
    """Adjust color saturation of an image.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.

    Returns:
        PIL Image: Saturation adjusted image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img


def adjust_hue(img, hue_factor):
    """Adjust hue of an image.

    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    See https://en.wikipedia.org/wiki/Hue for more details on Hue.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.

    Returns:
        PIL Image: Hue adjusted image.
    """
    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return img

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return img


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue(float): How much to jitter hue. hue_factor is chosen uniformly from
            [-hue, hue]. Should be >=0 and <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []
        if brightness > 0:
            brightness_factor = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
            transforms.append(
                torch_tr.Lambda(lambda img: adjust_brightness(img, brightness_factor)))

        if contrast > 0:
            contrast_factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(
                torch_tr.Lambda(lambda img: adjust_contrast(img, contrast_factor)))

        if saturation > 0:
            saturation_factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
            transforms.append(
                torch_tr.Lambda(lambda img: adjust_saturation(img, saturation_factor)))

        if hue > 0:
            hue_factor = np.random.uniform(-hue, hue)
            transforms.append(
                torch_tr.Lambda(lambda img: adjust_hue(img, hue_factor)))

        np.random.shuffle(transforms)
        transform = torch_tr.Compose(transforms)

        return transform

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(img)

def customsoftmax(inp, multihotmask):
    """
    Custom Softmax
    """
    soft = F.softmax(inp,dim=1)
    # This takes the mask * softmax ( sums it up hence summing up the classes in border
    # then takes of summed up version vs no summed version
    eps=1e-20
    return torch.log(
        torch.max(soft, (multihotmask * (soft * multihotmask).sum(1, keepdim=True)) + eps)
    )

class ImgWtLossSoftNLL(nn.Module):
    """
    Relax Loss
    """

    def __init__(self, classes, ignore_index=255, weights=None, upper_bound=1.0,
                 norm=False):
        super(ImgWtLossSoftNLL, self).__init__()
        self.weights = weights
        self.num_classes = classes
        self.ignore_index = ignore_index
        self.upper_bound = upper_bound
        self.norm = norm
        self.batch_weights = cfg.BATCH_WEIGHTING
        self.fp16 = False

    def calculate_weights(self, target):
        """
        Calculate weights of the classes based on training crop
        """
        if len(target.shape) == 3:
            hist = np.sum(target, axis=(1, 2)) * 1.0 / target.sum()
        else:
            hist = np.sum(target, axis=(0, 2, 3)) * 1.0 / target.sum()
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist[:-1]

    def custom_nll(self, inputs, target, class_weights, border_weights, mask):
        """
        NLL Relaxed Loss Implementation
        """
        if (cfg.REDUCE_BORDER_EPOCH != -1 and cfg.EPOCH > cfg.REDUCE_BORDER_EPOCH):
            border_weights = 1 / border_weights
            target[target > 1] = 1
        if self.fp16:
            loss_matrix = (-1 / border_weights *
                           (target[:, :-1, :, :].half() *
                            class_weights.unsqueeze(0).unsqueeze(2).unsqueeze(3) *
                            customsoftmax(inputs, target[:, :-1, :, :].half())).sum(1)) * \
                          (1. - mask.half())
        else:
            loss_matrix = (-1 / border_weights *
                           (target[:, :-1, :, :].float() *
                            class_weights.unsqueeze(0).unsqueeze(2).unsqueeze(3) *
                            customsoftmax(inputs, target[:, :-1, :, :].float())).sum(1)) * \
                          (1. - mask.float())

            # loss_matrix[border_weights > 1] = 0
        loss = loss_matrix.sum()

        # +1 to prevent division by 0
        loss = loss / (target.shape[0] * target.shape[2] * target.shape[3] - mask.sum().item() + 1)
        return loss

    def forward(self, inputs, target, pixelweights=None):
        if self.fp16:
            weights = target[:, :-1, :, :].sum(1).half()
        else:
            weights = target[:, :-1, :, :].sum(1).float()
        ignore_mask = (weights == 0)
        weights[ignore_mask] = 1
        loss = 0
        target_cpu = target.data.cpu().numpy()

        if self.batch_weights:
            class_weights = self.calculate_weights(target_cpu)

        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
                class_weights = self.calculate_weights(target_cpu[i])
            loss = loss + self.custom_nll(inputs[i].unsqueeze(0),
                                          target[i].unsqueeze(0),
                                          class_weights=torch.Tensor(class_weights).cuda(),
                                          border_weights=weights[i].unsqueeze(0), mask=ignore_mask[i])
        loss = loss/inputs.shape[0]
        if pixelweights is not None:
            loss = torch.mean(loss.cuda() * pixelweights.cuda())
            #print("Pixelweight ImgWtLossSoftNLL",loss.shape,loss)

        return loss

class CrossEntropyLoss2d(nn.Module):
    """
    Cross Entroply NLL Loss
    """

    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)
        # self.weight = weight

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)

if __name__ == '__main__':

    print('---------------------------------Test NVIDIA Transforms---------------------------------')
    # cfg.DATASET.NUM_CLASSES = 5
    # cfg.DATASET.IGNORE_LABEL = 250
    # wt_bound = 1.0
    # rlbl_tensor = RelaxedBoundaryLossToTensor(cfg.DATASET.IGNORE_LABEL,cfg.DATASET.NUM_CLASSES)
    # img = np.random.random_integers(cfg.DATASET.NUM_CLASSES, size=(4,4))
    # print("Img",img.shape)
    # lbls_relaxed = rlbl_tensor(img)
    # print("Lbls rlx",lbls_relaxed.shape)
    # lbls_relaxed = lbls_relaxed.reshape(1,cfg.DATASET.NUM_CLASSES+1,4,4).cuda()
    # # print("Lbls relaxed shape",lbls_relaxed.shape)
    # #
    # m = nn.Softmax(dim=1)
    # pred = torch.randn(1,cfg.DATASET.NUM_CLASSES+1, 4, 4)
    # print("Random out",pred.shape)
    # soft_max = m(pred).cuda()
    # print("Softmax",soft_max.shape)
    # # criterion = ImgWtLossSoftNLL(
    # #     classes=cfg.DATASET.NUM_CLASSES,
    # #     ignore_index=cfg.DATASET.IGNORE_LABEL,
    # #     upper_bound=wt_bound)
    # #criterion = CrossEntropyLoss2d()
    # # criterion = CrossEntropyLoss2d(
    # #     ignore_index=cfg.DATASET.IGNORE_LABEL).cuda()
    # criterion = nn.NLLLoss()
    # loss = criterion(soft_max, lbls_relaxed)
    # print("Loss",loss)

    # 2D loss example (used, for example, with image inputs)
    print("cfg",cfg)
    N, C = 1, 4
    loss = nn.NLLLoss()
    # input is of size N x C x height x width
    data = torch.randn(N, 16, 8, 8)
    conv = nn.Conv2d(16, C, (3, 3))
    m = nn.LogSoftmax(dim=1)
    # each element in target has to have 0 <= value < C
    target = torch.empty(N, 6, 6, dtype=torch.long).random_(0, C)
    #print("target",target)
    output = loss(m(conv(data)), target)
    print("loss standard softmax",output)
    #output.backward()

    cfg.DATASET.NUM_CLASSES = C
    cfg.DATASET.IGNORE_LABEL = 250
    wt_bound = 1.0
    rlbl_tensor = RelaxedBoundaryLossToTensor(cfg.DATASET.IGNORE_LABEL,cfg.DATASET.NUM_CLASSES)
    target_relaxed = rlbl_tensor(target[0])
    target_relaxed = target_relaxed.reshape(1,cfg.DATASET.NUM_CLASSES+1,6,6).cuda()
    criterion = ImgWtLossSoftNLL(
        classes=cfg.DATASET.NUM_CLASSES,
        ignore_index=cfg.DATASET.IGNORE_LABEL,
        upper_bound=wt_bound)
    print("target relaxed",target_relaxed.shape)
    loss = criterion(conv(data).cuda(), target_relaxed)
    print("Loss relaxed",loss)
