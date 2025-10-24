import gzip
import hashlib
import logging
import os.path
import struct
import sys
import urllib.request
from array import array as pyarray

import numpy as np

# BASE_URL = "https://yann.lecun.com/exdb/mnist"
BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist"


IMAGES = {
    "train": {
        "path": "train-images-idx3-ubyte.gz",
        "sha256": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    },
    "test": {
        "path": "t10k-images-idx3-ubyte.gz",
        "sha256": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    },
}


LABELS = {
    "train": {
        "path": "train-labels-idx1-ubyte.gz",
        "sha256": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
    },
    "test": {
        "path": "t10k-labels-idx1-ubyte.gz",
        "sha256": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
    },
}


def check(path, sha256sum):
    valid = False
    with open(path, "rb") as f:
        valid = hashlib.sha256(f.read()).hexdigest() == sha256sum
    return valid


def download(in_url, out_fname, chunk_size=1024 * 32):
    logging.info("Downloading '{}'".format(in_url))
    # wget.download(in_url, out_fname)
    req = urllib.request.Request(in_url, headers=dict(UserAgent="MNISTDownloader"))
    with urllib.request.urlopen(req) as res:
        with open(out_fname, "wb") as f:
            for chunk in iter(lambda: res.read(chunk_size), b""):
                if not chunk:
                    continue
                f.write(chunk)


def load(split="train", path_dir=".", digits=np.arange(10), as_is=False):
    # Download and check original data if needed
    for f in IMAGES[split], LABELS[split]:
        path = os.path.join(path_dir, f["path"])
        if not os.path.exists(path):
            url = BASE_URL + "/" + f["path"]
            logging.info("Downloading {}".format(url))
            download(url, path)
        if not check(path, f["sha256"]):
            logging.error("'{}' is corrupted; aborting".format(path))
            sys.exit(1)

    with gzip.open(os.path.join(path_dir, LABELS[split]["path"]), "rb") as flbl:
        _magic_nr, size = struct.unpack(">II", flbl.read(8))
        lbl = pyarray("b", flbl.read())

    with gzip.open(os.path.join(path_dir, IMAGES[split]["path"]), "rb") as fimg:
        _magic_nr, _size, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = pyarray("B", fimg.read())

    ind = [k for k in range(size) if lbl[k] in digits]
    N = len(ind)

    images = np.zeros((N, rows * cols), dtype=np.uint8)
    labels = np.zeros((N,), dtype=np.int8)
    for i in range(N):
        m = ind[i] * rows * cols
        n = (ind[i] + 1) * rows * cols
        images[i] = np.array(img[m:n])
        labels[i] = lbl[ind[i]]

    if as_is:
        return images, labels

    images = images / 255.0
    images = np.reshape(images, (-1, 1, 28, 28))
    return images, labels
