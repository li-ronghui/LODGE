import glob
import os
import re
from pathlib import Path
import numpy as np

import torch
from pytorch3d.transforms import (axis_angle_to_matrix, matrix_to_axis_angle,
                                  matrix_to_quaternion, matrix_to_rotation_6d,
                                  quaternion_to_matrix, rotation_6d_to_matrix)
import pickle

def increment_path(path, exist_ok=False, sep="", mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix("")
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == "" else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path

'''
class Normalizer:
    def __init__(self, data):
        if isinstance(data, str):
            self.scaler = MinMaxScaler((-1, 1), clip=True)
            with open(data, 'rb') as f:
                normalizer_state_dict = pickle.load(f)
            # normalizer_state_dict = torch.load(data)
            self.scaler.scale_ = normalizer_state_dict["scale"]
            self.scaler.min_ = normalizer_state_dict["min"]
        else:
            flat = data.reshape(-1, data.shape[-1])     # bxt , 151
            self.scaler = MinMaxScaler((-1, 1), clip=True)
            self.scaler.fit(flat)

    def normalize(self, x):
        if len(x.shape) == 3:
            batch, seq, ch = x.shape
            x = x.reshape(-1, ch)
            return self.scaler.transform(x).reshape((batch, seq, ch))
        elif len(x.shape) == 2:
            batch, ch = x.shape
            return self.scaler.transform(x)
        else:
            raise("input error!")

    def unnormalize(self, x):
        if len(x.shape) == 3:
            batch, seq, ch = x.shape
            x = x.reshape(-1, ch)
            x = torch.clip(x, -1, 1)  # clip to force compatibility
            return self.scaler.inverse_transform(x).reshape((batch, seq, ch))
        elif len(x.shape) == 2:
             x = torch.clip(x, -1, 1)
             return self.scaler.inverse_transform(x)
        else:
            raise("input error!")
'''


def quat_to_6v(q):
    assert q.shape[-1] == 4
    mat = quaternion_to_matrix(q)
    mat = matrix_to_rotation_6d(mat)
    return mat


def quat_from_6v(q):
    assert q.shape[-1] == 6
    mat = rotation_6d_to_matrix(q)
    quat = matrix_to_quaternion(mat)
    return quat


def ax_to_6v(q):
    assert q.shape[-1] == 3
    mat = axis_angle_to_matrix(q)
    mat = matrix_to_rotation_6d(mat)
    return mat


def ax_from_6v(q):
    assert q.shape[-1] == 6
    mat = rotation_6d_to_matrix(q)
    ax = matrix_to_axis_angle(mat)
    return ax


def quat_slerp(x, y, a):
    """
    Performs spherical linear interpolation (SLERP) between x and y, with proportion a

    :param x: quaternion tensor (N, S, J, 4)
    :param y: quaternion tensor (N, S, J, 4)
    :param a: interpolation weight (S, )
    :return: tensor of interpolation results
    """
    len = torch.sum(x * y, axis=-1)

    neg = len < 0.0
    len[neg] = -len[neg]
    y[neg] = -y[neg]

    a = torch.zeros_like(x[..., 0]) + a

    amount0 = torch.zeros_like(a)
    amount1 = torch.zeros_like(a)

    linear = (1.0 - len) < 0.01
    omegas = torch.arccos(len[~linear])
    sinoms = torch.sin(omegas)

    amount0[linear] = 1.0 - a[linear]
    amount0[~linear] = torch.sin((1.0 - a[~linear]) * omegas) / sinoms

    amount1[linear] = a[linear]
    amount1[~linear] = torch.sin(a[~linear] * omegas) / sinoms

    # reshape
    amount0 = amount0[..., None]
    amount1 = amount1[..., None]

    res = amount0 * x + amount1 * y

    return res


def _handle_zeros_in_scale(scale, copy=True, constant_mask=None):
    # if we are fitting on 1D arrays, scale might be a scalar
    if constant_mask is None:
        # Detect near constant values to avoid dividing by a very small
        # value that could lead to surprising results and numerical
        # stability issues.
        constant_mask = scale < 10 * torch.finfo(scale.dtype).eps

    if copy:
        # New array to avoid side-effects
        scale = scale.clone()
    scale[constant_mask] = 1.0
    return scale


class MinMaxScaler:
    _parameter_constraints: dict = {
        "feature_range": [tuple],
        "copy": ["boolean"],
        "clip": ["boolean"],
    }

    def __init__(self, feature_range=(0, 1), *, copy=True, clip=False):
        self.feature_range = feature_range
        self.copy = copy
        self.clip = clip

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """
        # Checking one attribute is enough, because they are all set together
        # in partial_fit
        if hasattr(self, "scale_"):
            del self.scale_
            del self.min_
            del self.n_samples_seen_
            del self.data_min_
            del self.data_max_
            del self.data_range_

    def fit(self, X):
        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X)

    def partial_fit(self, X):
        feature_range = self.feature_range
        if feature_range[0] >= feature_range[1]:
            raise ValueError(
                "Minimum of desired feature range must be smaller than maximum. Got %s."
                % str(feature_range)
            )

        data_min = torch.min(X, axis=0)[0]
        data_max = torch.max(X, axis=0)[0]

        self.n_samples_seen_ = X.shape[0]

        data_range = data_max - data_min
        self.scale_ = (feature_range[1] - feature_range[0]) / _handle_zeros_in_scale(
            data_range, copy=True
        )
        self.min_ = feature_range[0] - data_min * self.scale_
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range
        return self

    def transform(self, X):
        X *= self.scale_.to(X.device)
        X += self.min_.to(X.device)
        if self.clip:
            torch.clip(X, self.feature_range[0], self.feature_range[1], out=X)
        return X

    def inverse_transform(self, X):
        X -= self.min_[-X.shape[1] :].to(X.device)
        X /= self.scale_[-X.shape[1] :].to(X.device)
        return X


class Normalizer:
    def __init__(self, data):
        flat = data.reshape(-1, data.shape[-1])
        self.scaler = MinMaxScaler((-1, 1), clip=True)
        self.scaler.fit(flat)

    def normalize(self, x):
        if len(x.shape) == 3:
            batch, seq, ch = x.shape
            x = x.reshape(-1, ch)
            return self.scaler.transform(x).reshape((batch, seq, ch))
        elif len(x.shape) == 2:
            batch, ch = x.shape
            return self.scaler.transform(x)

    def unnormalize(self, x):
        if len(x.shape) == 3:
            batch, seq, ch = x.shape
            x = x.reshape(-1, ch)
            x = torch.clip(x, -1, 1)  # clip to force compatibility
            return self.scaler.inverse_transform(x).reshape((batch, seq, ch))
        elif len(x.shape) == 2:
             x = torch.clip(x, -1, 1)
             return self.scaler.inverse_transform(x)
        else:
            raise("input error!")



class Normal:
    def __init__(self, mean, std):
        self.mean = np.load(mean)
        self.std = np.load(std)
        self.mean = torch.from_numpy(self.mean)
        self.std = torch.from_numpy(self.std)

    def normalize(self, x):
        self.mean = self.mean.to(x)
        self.std = self.std.to(x)
        return (x - self.mean) / self.std 
     
    def unnormalize(self, x):
        self.mean = self.mean.to(x)
        self.std = self.std.to(x)
        return x * self.std + self.mean