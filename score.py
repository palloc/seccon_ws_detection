#!/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import csv

# File name and full path.
file_name = os.path.basename(__file__)
full_path = os.path.dirname(os.path.abspath(__file__))

# Label directory.
LABEL_DIR = os.path.join(full_path, 'label')
LABEL_LEVEL1 = os.path.join(LABEL_DIR, 'level1_label.csv')
LABEL_LEVEL2 = os.path.join(LABEL_DIR, 'level2_label.csv')

# Answer directory.
ANS_DIR = os.path.join(full_path, 'answer')

if len(sys.argv) != 3:
    print('invalid argument.')
    sys.exit(1)


def scoring(label_path):
    # Read label.
    label = []
    with open(label_path, mode='r') as fin:
        reader = csv.reader(fin)
        for row in reader:
            label.extend(row)

    # Read answer.
    answer = []
    with open(os.path.join(ANS_DIR, sys.argv[2]), mode='r') as fin:
        reader = csv.reader(fin)
        for row in reader:
            answer.extend(row)

    if len(label) != len(answer):
        print('invalid answer list.')
        sys.exit(1)

    hit_count = 0
    for label_val, ans_val in zip(label, answer):
        if label_val == ans_val:
            hit_count += 1
    return float(hit_count/len(label))*100


if __name__ == '__main__':
    if int(sys.argv[1]) == 1:
        print('{}\'s score: {} %'.format(sys.argv[2], scoring(LABEL_LEVEL1)))
    elif int(sys.argv[1]) == 2:
        print('{}\'s score: {} %'.format(sys.argv[2], scoring(LABEL_LEVEL2)))
    else:
        print('invalid level.')
        sys.exit(1)
