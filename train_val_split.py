import os
from glob import glob

import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--data-dir', default='/mnt/hdd02/shibuya_scramble',
                        help='original data directory')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    ## Random Train-Val split

    im_list = sorted(glob(os.path.join(args.data_dir, '*.jpg')))
    im_list = [im_name for im_name in im_list]

    tr_im_list = list(np.random.choice(im_list, size=int(len(im_list)*0.8), replace=False))
    vl_im_list = list(set(im_list) - set(tr_im_list))

    for phase in ['train', 'val']:
        with open(os.path.join(args.data_dir, './{}.txt'.format(phase)), mode='w') as f:
            if phase == 'train':
                f.write('\n'.join(tr_im_list))
            elif phase == 'val':
                f.write('\n'.join(vl_im_list))
