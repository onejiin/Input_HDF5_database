from __future__ import print_function
import h5py
import numpy as np
from PIL import Image

__author__ = 'wjung'


def HDF5_write(data_label_list, gzip_dot_h5_file_name):
    # define width x height
    cols = 512
    rows = 424
    channel = 1
    total_image = len(data_label_list)
    total_size = total_image * rows * cols

    data = np.empty((total_image, channel, cols, rows))
    multi_label = np.empty((total_image, 2))

    # to input batch mean normalization
    data_mean = np.empty((cols, rows))

    cnt_data_number = 0

    for n in range(total_image):

        line = data_label_list[n]
        line = line.rstrip('\n')
        list_line = line.split(' ')

        # data (e.g.binary input unsigned int 16bit)
        data_bin = np.fromfile(list_line[0], dtype=np.uint16)

#        # if input line is jpg(or png)
#        jpg_data = Image.open(list_line[0])
#        #jpg_data = jpg_data.resize((res_cols, res_rows), Image.ANTIALIAS)
#        pixels = jpg_data.load()

        for i in range(rows):
            for j in range(cols):
                data[cnt_data_number, 0,  j, i] = data_bin[(i*cols) + j]

#               # if input line is jpg(or png)
#                data[cnt, 0,  i, j] = pixels[i, j] * 0.00390625   # scale 0~1

                # to normalization
                data_mean[j, i] += data[cnt_data_number, 0,  j, i]

        multi_label[cnt_data_number, 0] = str(int(float(list_line[2])))
        multi_label[cnt_data_number, 1] = str(int(float(list_line[3])))

        cnt_data_number += 1

    # mean normalization based on batch size - 100
    data_mean /= total_image

    cnt_data_number = 0
    for n in range(total_image):
        for i in range(rows):
            for j in range(cols):
                data[cnt_data_number, 0,  j, i] -= data_mean[j, i]
        cnt_data_number += 1

#   # Do not compression
#    with h5py.File(dot_h5_file_name, 'w') as hf:
#        hf['data'] = data
#        hf['label'] = multi_label

    # save HDF5 database as gzip format
    with h5py.File(gzip_dot_h5_file_name, 'w') as f:
        f.create_dataset(
            'data', data=data + total_size,
            compression='gzip', compression_opts=1
        )
        f.create_dataset(
            'label', data=multi_label,
            compression='gzip', compression_opts=1,
            dtype='uint8',
        )

    del data
    f.close()