#                                       #
# multi-label HDF5 database generation  #
#                                       #

import sys
from function import *

__author__ = 'wjung'

list_input = './data.txt'
list_label = './label.txt'

save_gzip_dot_h5_file_name = './output_HDF5/train_DepFin_gzip'

def main(argv):
    f_data = open(list_input)
    f_label = open(list_label)

    buff = []
    one_file_write_size = 0
    cnt_batch = 0
    batch_size = 512

    while 1:
        line_data = f_data.readline()
        line_label = f_label.readline()
        if not line_data:
            break
        line_data = line_data.rstrip('\n')
        line_label = line_label.rstrip('\n')

        # text write label number
        f_line_label = open(line_label)
        line_finger_value = f_line_label.readlines()
        for line_label in line_finger_value:
            line_label = line_label.rstrip('\n')
            list_finger_xy = line_label.split(' ')
        str_buff = line_data + ' ' + str(list_finger_xy[0]) + " " + str(list_finger_xy[1]) + "\n"
        # str_buff : [data root] [x-axis] [y-axis], axises are pixel location

        buff.append(str_buff)

        if one_file_write_size == batch_size:
            one_file_write_size = 0

            # each write hdf5 name
            gzip_dot_h5_file_name = save_gzip_dot_h5_file_name + str(cnt_batch) + '.h5'

            HDF5_write(buff, gzip_dot_h5_file_name)

            # init condition
            buff = []
            cnt_batch += 1
        else:
            one_file_write_size += 1

    gzip_dot_h5_file_name = save_gzip_dot_h5_file_name + str(cnt_batch) + '.h5'
    HDF5_write(buff, gzip_dot_h5_file_name)

    f_data.close()
    f_label.close()


if __name__ == '__main__':
    sys.path.append('config')
    sys.exit(main(sys.argv))
