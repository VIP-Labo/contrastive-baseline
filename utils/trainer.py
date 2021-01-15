import os
import json
import logging
from datetime import datetime
from utils.logger import setlogger

class Trainer(object):
    def __init__(self, args):
        #sub_dir = datetime.strftime(datetime.now(), '%m%d-%H%M%S')  # prepare saving path
        #sub_dir = sub_dir + '-{}'.format(args.arch)
        sub_dir = args.arch
        self.save_dir = os.path.join(args.save_dir, sub_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        setlogger(os.path.join(self.save_dir, 'train.log'))  # set logger

        with open(os.path.join(self.save_dir, 'args.json'), 'w') as opt_file:
            json.dump(vars(args), opt_file)

        os.makedirs(os.path.join(self.save_dir, 'images')) ### 追加

        for k, v in args.__dict__.items():  # save args
            logging.info("{}: {}".format(k, v))
        self.args = args

    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        pass

    def train(self):
        """training one epoch"""
        pass
