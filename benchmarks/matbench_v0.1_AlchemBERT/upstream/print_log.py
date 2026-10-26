import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
import numpy as np

def scan_folder(parent_folder, tag='test_mae'):
    for dir_name, subdir_list, file_list in os.walk(parent_folder):
        for fname in file_list:
            if "events.out.tfevents" in fname:
                event_path = os.path.join(dir_name, fname)
                for e in tf.compat.v1.train.summary_iterator(event_path):
                    for v in e.summary.value:
                        if v.tag == tag:
                            print(f'Value for {tag} in {event_path}: {v.simple_value}')

scan_folder("logs_nl_best_0.1vali_200patience_with_mask")

