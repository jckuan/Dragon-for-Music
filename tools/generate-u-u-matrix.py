import numpy as np
from collections import defaultdict
from tqdm import tqdm
import torch
import pandas as pd
import os
import sys
import yaml
import argparse

# Get project root (two levels up from file1a.py)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add utils folder to sys.path if not already present
utils_path = os.path.join(project_root, 'utils')
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from gcs_utils import (
    is_gcs_path, get_gcs_file_path, read_csv_from_gcs, save_numpy_to_gcs,
    file_exists, read_file_content
)



def gen_user_matrix(all_edge, no_users):
    edge_dict = defaultdict(set)

    for edge in all_edge:
        user, item = edge
        edge_dict[user].add(item)

    min_user = 0             # 0
    num_user = no_users      # in our case, users/items ids start from 1
    user_graph_matrix = torch.zeros(num_user, num_user)
    key_list = list(edge_dict.keys())
    key_list.sort()
    bar = tqdm(total=len(key_list))
    for head in range(len(key_list)):
        bar.update(1)
        for rear in range(head+1, len(key_list)):
            head_key = key_list[head]
            rear_key = key_list[rear]
            # print(head_key, rear_key)
            item_head = edge_dict[head_key]
            item_rear = edge_dict[rear_key]
            # print(len(user_head.intersection(user_rear)))
            inter_len = len(item_head.intersection(item_rear))
            if inter_len > 0:
                user_graph_matrix[head_key-min_user][rear_key-min_user] = inter_len
                user_graph_matrix[rear_key-min_user][head_key-min_user] = inter_len
    bar.close()

    return user_graph_matrix


if __name__ == 	'__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='games', help='name of dataset')
    args = parser.parse_args()
    dataset_name = args.dataset
    print(f'Generating u-u matrix for {dataset_name} ...\n')

    config = {}
    con_dir = os.path.join(project_root, 'configs') # get config dir
    overall_config_file = os.path.join(con_dir, "overall.yaml")
    dataset_config_file = os.path.join(con_dir, "dataset", f"{dataset_name}.yaml")
    conf_files = [overall_config_file, dataset_config_file]
    # load configs
    for file in conf_files:
        if os.path.isfile(file):
            with open(file, 'r', encoding='utf-8') as f:
                tmp_d = yaml.safe_load(f)
                config.update(tmp_d)

    # Handle both local and GCS data paths
    data_path = config['data_path']
    if is_gcs_path(data_path):
        dataset_path = get_gcs_file_path(data_path, dataset_name)
        inter_file_path = get_gcs_file_path(dataset_path, config['inter_file_name'])
        
        # Read CSV from GCS
        train_df = read_csv_from_gcs(inter_file_path, sep='\t')
    else:
        dataset_path = os.path.abspath(config['data_path'] + dataset_name)
        inter_file_path = os.path.join(dataset_path, config['inter_file_name'])
        train_df = pd.read_csv(inter_file_path, sep='\t')
    
    uid_field = config['USER_ID_FIELD']
    iid_field = config['ITEM_ID_FIELD']
    num_user = len(pd.unique(train_df[uid_field]))
    train_df = train_df[train_df['x_label'] == 0].copy()
    train_data = train_df[[uid_field, iid_field]].to_numpy()
    # item_item_pairs =[]
    user_graph_matrix = gen_user_matrix(train_data, num_user)
    #####################################################################generate user-user matrix
    # pdb.set_trace()
    user_graph = user_graph_matrix
    # user_num = torch.zeros(num_user)
    user_num = torch.zeros(num_user)

    user_graph_dict = {}
    item_graph_dict = {}
    edge_list_i = []
    edge_list_j = []

    for i in range(num_user):
        user_num[i] = len(torch.nonzero(user_graph[i]))
        print("this is ", i, "num", user_num[i])

    for i in range(num_user):
        if user_num[i] <= 200:
            user_i = torch.topk(user_graph[i],int(user_num[i]))
            edge_list_i =user_i.indices.numpy().tolist()
            edge_list_j =user_i.values.numpy().tolist()
            edge_list = [edge_list_i, edge_list_j]
            user_graph_dict[i] = edge_list
        else:
            user_i = torch.topk(user_graph[i], 200)
            edge_list_i = user_i.indices.numpy().tolist()
            edge_list_j = user_i.values.numpy().tolist()
            edge_list = [edge_list_i, edge_list_j]
            user_graph_dict[i] = edge_list
    # pdb.set_trace()
    
    # Save to GCS or local based on data path
    output_file_path = get_gcs_file_path(dataset_path, config['user_graph_dict_file']) if is_gcs_path(data_path) else os.path.join(dataset_path, config['user_graph_dict_file'])
    
    if is_gcs_path(data_path):
        save_numpy_to_gcs(user_graph_dict, output_file_path, allow_pickle=True)
    else:
        np.save(output_file_path, user_graph_dict, allow_pickle=True)
