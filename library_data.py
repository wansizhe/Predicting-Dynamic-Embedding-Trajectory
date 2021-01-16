"""
This is a supporting library for the loading the data.

Paper: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. S. Kumar, X. Zhang, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2019. 
"""


import numpy as np
import random
import sys
import operator
import copy
from collections import defaultdict
import os, re
import pickle
import argparse
from sklearn.preprocessing import scale


def popular_recommend(pop_items, num_items = 40):
    rec_array = np.random.choice(pop_items, (num_items, ), replace=False)
    return rec_array.tolist()


# LOAD THE NETWORK
def load_network(arg, time_scaling=True):
    """
    This function loads the input network.

    The network should be in the following format:
    One line per interaction/edge.
    Each line should be: user, item, timestamp, state label, array of features.
    Timestamp should be in cardinal format (not in datetime).
    State label should be 1 whenever the user state changes, 0 otherwise. If there are no state labels, use 0 for all interactions.
    Feature list can be as long as desired. It should be at least 1 dimensional. If there are no features, use 0 for all interactions.
    """

    network = 'train'
    datapath = '../../ydzhang/CIKM2019/data/jodie_input_all_9.npy'

    user_sequence = []
    item_sequence = []
    label_sequence = []
    feature_sequence = []
    timestamp_sequence = []
    start_timestamp = None
    y_true_labels = []

    print("\n\n**** Loading %s network from file: %s ****" % (network, datapath))
    # f = open(datapath, "r")
    # f.readline()
    f = np.load(datapath)
    user_previous_item_dict = {}
    for cnt, ls in enumerate(f):
        # FORMAT: user, item, timestamp, state label, feature list 
        # ls = l.strip().split(",")
        user_sequence.append(ls[0])
        item_sequence.append(ls[1])
        if start_timestamp is None:
            start_timestamp = float(ls[2])
        timestamp_sequence.append(float(ls[2]) - start_timestamp) 
        # y_true_labels.append(ls[3])  # label = 1 at state change, 0 otherwise
        feature_sequence.append(list(map(float, ls[3:])))
    # f.close()

    user_sequence = np.array(user_sequence) 
    item_sequence = np.array(item_sequence)
    timestamp_sequence = np.array(timestamp_sequence)

    print("Formating item sequence")
    nodeid = 0
    item2id = {}  # dict，item到id的映射
    item_timedifference_sequence = []  # list，item本次被访问和上次被访问的时间差
    item_current_timestamp = defaultdict(float)  # dict， item本次（最近一次）被访问的时间戳
    for cnt, item in enumerate(item_sequence):
        if item not in item2id:
            item2id[item] = nodeid
            nodeid += 1
        timestamp = timestamp_sequence[cnt]
        item_timedifference_sequence.append(timestamp - item_current_timestamp[item])
        item_current_timestamp[item] = timestamp
    num_items = len(item2id)
    item_sequence_id = [item2id[item] for item in item_sequence]  # item_sequence映射到id的序列

    print("Formating user sequence")
    nodeid = 0
    user2id = {}  # dict，user到id到映射
    user_timedifference_sequence = []  # list，user本次操作和上次操作的时间差
    user_current_timestamp = defaultdict(float)  # dict，user本次（最近一次）操作的时间戳
    user_previous_itemid_sequence = []  # list，当前user上次访问的item的id
    user_latest_itemid = defaultdict(lambda: num_items)  # dict，当前user本次（最近一次）访问的item的id
    for cnt, user in enumerate(user_sequence):
        if user not in user2id:
            user2id[user] = nodeid
            nodeid += 1
        timestamp = timestamp_sequence[cnt]
        user_timedifference_sequence.append(timestamp - user_current_timestamp[user])
        user_current_timestamp[user] = timestamp
        user_previous_itemid_sequence.append(user_latest_itemid[user])
        user_latest_itemid[user] = item2id[item_sequence[cnt]]
    num_users = len(user2id)
    user_sequence_id = [user2id[user] for user in user_sequence]  # user_sequence映射到id的序列

    if time_scaling:
        print("Scaling timestamps")
        user_timedifference_sequence = scale(np.array(user_timedifference_sequence) + 1)
        item_timedifference_sequence = scale(np.array(item_timedifference_sequence) + 1)

    print("*** Network loading completed ***\n\n")
    return [user2id, user_sequence_id, user_timedifference_sequence, user_previous_itemid_sequence, \
        item2id, item_sequence_id, item_timedifference_sequence, \
        timestamp_sequence, \
        feature_sequence, \
        y_true_labels]


def load_network_test(arg, time_scaling=True):
    """
    This function loads the input network.

    The network should be in the following format:
    One line per interaction/edge.
    Each line should be: user, item, timestamp, state label, array of features.
    Timestamp should be in cardinal format (not in datetime).
    State label should be 1 whenever the user state changes, 0 otherwise. If there are no state labels, use 0 for all interactions.
    Feature list can be as long as desired. It should be at least 1 dimensional. If there are no features, use 0 for all interactions.
    """

    network = 'train'
    datapath = '../../ydzhang/CIKM2019/data/jodie_input_all_9.npy'

    user_sequence = []
    item_sequence = []
    label_sequence = []
    feature_sequence = []
    timestamp_sequence = []
    start_timestamp = None
    y_true_labels = []

    print("\n\n**** Loading %s network from file: %s ****" % (network, datapath))
    # f = open(datapath, "r")
    # f.readline()
    f = np.load(datapath)
    # user_previous_item_dict = {}
    for cnt, ls in enumerate(f):
        # FORMAT: user, item, timestamp, state label, feature list
        # ls = l.strip().split(",")
        # if ls[0] not in user_previous_item_dict:
        #     user_previous_item_dict[ls[0]] = ls[1]
        user_sequence.append(ls[0])
        item_sequence.append(ls[1])
        if start_timestamp is None:
            start_timestamp = float(ls[2])
        timestamp_sequence.append(float(ls[2]) - start_timestamp)
        # y_true_labels.append(ls[3])  # label = 1 at state change, 0 otherwise
        feature_sequence.append(list(map(float, ls[3:])))
    # f.close()

    user_sequence = np.array(user_sequence)
    item_sequence = np.array(item_sequence)
    timestamp_sequence = np.array(timestamp_sequence)

    print("Formating item sequence")
    nodeid = 0
    item2id = {}  # dict，item到id的映射
    item_last = {}
    item_timedifference_sequence = []  # list，item本次被访问和上次被访问的时间差
    item_current_timestamp = defaultdict(float)  # dict， item本次（最近一次）被访问的时间戳
    for cnt, item in enumerate(item_sequence):
        if item not in item2id:
            item2id[item] = nodeid
            nodeid += 1
        item_last[item] = cnt
        timestamp = timestamp_sequence[cnt]
        item_timedifference_sequence.append(timestamp - item_current_timestamp[item])
        item_current_timestamp[item] = timestamp
    num_items = len(item2id)
    item_sequence_id = [item2id[item] for item in item_sequence]  # item_sequence映射到id的序列

    print("Formating user sequence")
    nodeid = 0
    user2id = {}  # dict，user到id到映射
    user_last = {}
    user_timedifference_sequence = []  # list，user本次操作和上次操作的时间差
    user_current_timestamp = defaultdict(float)  # dict，user本次（最近一次）操作的时间戳
    user_previous_itemid_sequence = []  # list，当前user上次访问的item的id
    user_latest_itemid = defaultdict(lambda: num_items)  # dict，当前user本次（最近一次）访问的item的id
    for cnt, user in enumerate(user_sequence):
        if user not in user2id:
            user2id[user] = nodeid
            nodeid += 1
        user_last[user] = cnt
        timestamp = timestamp_sequence[cnt]
        user_timedifference_sequence.append(timestamp - user_current_timestamp[user])
        user_current_timestamp[user] = timestamp
        user_previous_itemid_sequence.append(user_latest_itemid[user])
        user_latest_itemid[user] = item2id[item_sequence[cnt]]
    num_users = len(user2id)
    user_sequence_id = [user2id[user] for user in user_sequence]  # user_sequence映射到id的序列

    if time_scaling:
        print("Scaling timestamps")
        user_timedifference_sequence = scale(np.array(user_timedifference_sequence) + 1)
        item_timedifference_sequence = scale(np.array(item_timedifference_sequence) + 1)

    print("*** Network loading completed ***\n\n")
    return [user2id, user_sequence_id, user_timedifference_sequence, user_previous_itemid_sequence, \
        item2id, item_sequence_id, item_timedifference_sequence, \
        timestamp_sequence, \
        feature_sequence, \
        y_true_labels, user_last, item_last]

