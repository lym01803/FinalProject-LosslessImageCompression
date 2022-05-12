import gzip
import bz2
import lzma
from pickletools import optimize
import numpy as np
import io

from PIL import Image 
from PIL.WebPImagePlugin import Image as PImg
import time

from trainer import CommonDataLoader

total = 0

def gzip_compress(images):
    original_size = np.size(images)
    images = np.packbits(images) if images.dtype is np.dtype(bool) else images
    assert images.dtype is np.dtype('uint8')
    global total
    t1 = time.time()
    res = gzip.compress(images.tobytes())
    total += time.time() - t1
    return res


def bz2_compress(images):
    original_size = np.size(images)
    images = np.packbits(images) if images.dtype is np.dtype(bool) else images
    assert images.dtype is np.dtype('uint8')
    return bz2.compress(images.tobytes())


def lzma_compress(images):
    original_size = np.size(images)
    images = np.packbits(images) if images.dtype is np.dtype(bool) else images
    assert images.dtype is np.dtype('uint8')
    return lzma.compress(images.tobytes())


def pimg_compress(images, format='PNG', **params):
    compressed_data = bytearray()
    for n, image in enumerate(images):
        image = image.transpose(1, 2, 0)
        image = Image.fromarray(image) if format != 'WebP' else PImg.fromarray(image)
        img_bytes = io.BytesIO()
        image.save(img_bytes, format=format, **params)
        compressed_data.extend(img_bytes.getvalue())
    return compressed_data


def gz_and_pimg(images, format='PNG', **params):
    pimg_compressed_data = pimg_compress(images, format, optimize=True)
    return gzip.compress(pimg_compressed_data)


if __name__ == '__main__':
    loader = CommonDataLoader(
        path='/home/yanming/files/flow_project/data/img_align_test',
        batch_size=256,
        shuffle=False,
        resize=[215, 178],
        centercrop=[215, 178],
        nbits=8,
        train=False
    )
    t1 = time.time()
    ttt = 0
    func = gzip_compress
    ori_size = 0
    comp_size = 0
    for data in loader:
        images = np.array(data * 256, dtype=np.uint8)
        # print(images)
        ori_size += np.size(images)
        print(ori_size)
        t2 = time.time()
        byts = func(images)
        ttt += time.time() - t2
        comp_size += len(byts) * 8
    print(f'ori_size: {ori_size}, comp_size: {comp_size}, bpd: {comp_size / ori_size}')
    print(time.time() - t1, ttt, total)

