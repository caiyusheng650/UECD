import json
import os
import sys

import torch
from collections import Counter
import csv
import random
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import ast

class DataLoader:
    def __init__(self, dataframe, batch_size):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.current_index = 0

    def next_batch(self):
        start_index = self.current_index
        end_index = start_index + self.batch_size

        batch_data = self.dataframe.iloc[start_index:end_index]

        self.current_index = end_index

        # if self.current_index >= len(self.dataframe):
        #     self.current_index = 0

        # self.dataframe['non_exer_length'] = self.dataframe['non_exer_ids'].apply(lambda x: len(x.split(',')))
        # total_non_exer_length = self.dataframe['non_exer_length'].sum()



        user_ids = torch.tensor(batch_data['user_id'].tolist(), dtype=torch.long)
        item_ids = torch.tensor(batch_data['item_id'].tolist(), dtype=torch.long)
        know_emb = [torch.tensor(list(map(float, ids.split(','))), dtype=torch.long) for ids in batch_data['know_emb']]
        know_emb = pad_sequence(know_emb, batch_first=True)
        neg_ids = [list(map(int, ids.split(','))) for ids in batch_data['non_exer_ids']]
        neg_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(ids, dtype=torch.long) for ids in neg_ids],
                                                  batch_first=True)
        scores = torch.tensor(batch_data['score'].tolist(), dtype=torch.float)

        return user_ids, item_ids, know_emb, scores, neg_ids

    def reset(self):
        self.current_index = 0

    def is_end(self):
        print(self.current_index)
        return self.current_index >= len(self.dataframe)

class TrainDataLoader(object):
    '''
    data loader for training
    '''
    def __init__(self,topk,batch_size):
        self.topk = topk
        n_cluster = 50
        # save_name = 'data/result' + str(n_cluster) + '.json'
        # # save_name = 'data/junyi/result' + str(n_cluster) + '.json'
        # with open(save_name, 'r') as f:
        #     self.clustering_results = json.load(f)
        self.batch_size = batch_size
        self.ptr = 0
        self.data = []
        self.data_group = 10

        # UECD_TRAIN_FILE lets overfit_test.py point at a tiny subset;
        # normal training always uses data/neg_exer_ours.json.
        data_file = os.environ.get('UECD_TRAIN_FILE', 'data/neg_exer_ours.json')
        # data_file = 'data/train_set.json'
        config_file = 'config.txt'
        print('loading %s ...' % data_file, flush=True)
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        print('loaded %d training records' % len(self.data), flush=True)
        with open(config_file) as i_f:
            i_f.readline()
            _, _, knowledge_n = i_f.readline().split(',')
        self.knowledge_dim = int(knowledge_n)
        # self.exer_knowledge_data, self.kn_max_length = self.load_data_from_csv('data/item.csv')
        # # self.exer_knowledge_data, self.kn_max_length = self.load_data_from_csv_junyi('data/junyi/items.csv')
        # self.user_interactions = self.get_user_interactions()

        # self.exer_difficulty = self.get_diff()

    def get_data_group(self):
        return self.data_group

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        input_stu_ids, input_exer_ids, input_knowedge_embs, ys, neg_pos_exer_ids = [], [], [], [], []
        # input_exer_diff = []
        json_data = []

        # for log in self.data:
        #     matching_knowledge_code = self.exer_knowledge_data.get(log['exer_id'] - 1)
        #     result = self.get_matching_exer(self.user_interactions, log['user_id'] - 1)
        #     top_exer_ids = self.get_same_kn_exer_ids(result, matching_knowledge_code, log['exer_id'] - 1)
        #
        #     # user_inter = self.user_interactions[log['user_id'] - 1]
        #     # top_exer_ids = self.random_all(self.user_interactions, user_inter)
        #     json_data.append({
        #         'user_id': log['user_id'] - 1,
        #         'exer_id': log['exer_id'] - 1,
        #         'score': log['score'],
        #         'knowledge_code': log['knowledge_code'],
        #         'neg_exer_ids': top_exer_ids
        #
        #     })
        # with open('data/neg_exer.json','w') as json_file:
        #     json.dump(json_data, json_file, indent=5)
        # sys.exit()

        # for record in self.data:
        #     if isinstance(record['knowledge_code'], int):
        #         input_knowedge_embs.append([record['knowledge_code'] - 1])
        #     else:
        #         input_knowedge_embs.append([code - 1 for code in record['knowledge_code']])
        #
        #     y = record['score']
        #     input_stu_ids.append(record['user_id'] - 1)
        #     input_exer_ids.append(record['exer_id'] - 1)
        #     ys.append(y)
        #
        #
        #     matching_knowledge_code = self.exer_knowledge_data.get(record['exer_id'] - 1)
        #     # if y:
        #     result = self.get_matching_exer(self.user_interactions, record['user_id'] - 1)
        #     #     # random_in_match += len(result)
        #     top_exer_ids = self.get_same_kn_exer_ids(result, matching_knowledge_code, record['exer_id'] - 1)
        #     neg_pos_exer_ids.append(top_exer_ids)
        #
        # df = pd.DataFrame({
        #     'user_id': input_stu_ids,
        #     'item_id': input_exer_ids,
        #     'know_emb': [",".join(map(str, knows)) for knows in input_knowedge_embs],
        #     'non_exer_ids': [",".join(map(str, ids)) for ids in neg_pos_exer_ids],
        #     'score': ys
        # })
        #
        # df.to_csv('data/user_item_nonexer_new.csv', index=False)
        # sys.exit()

        # file_path = 'data/user_item_nonexer.csv'
        # df = pd.read_csv(file_path)
        # #
        # # def read_csv_row_by_row(file_path):
        # #     for chunk in pd.read_csv(file_path, chunksize=1):
        # #         user_id = chunk['user_id'].values[0]
        # #         item_id = chunk['item_id'].values[0]
        # #         neg_ids = list(map(int, chunk['exer_ids'].values[0].split(',')))
        # #         score = chunk['score'].values[0]
        # #         yield user_id, item_id, neg_ids, score
        # #
        # # for user_id, item_id, neg_ids, score in read_csv_row_by_row(file_path):
        # #     print("User ID:", user_id)
        # #     print("Item ID:", item_id)
        # #     print("Neg IDs:", neg_ids)
        # #     print("Score:", score)
        # #     print('-' * 30)
        # def read_csv_in_batches(file_path, batch_size, device):
        #     for chunk in pd.read_csv(file_path, chunksize=batch_size):
        #         user_ids = torch.tensor(chunk['user_id'].values, dtype=torch.long).to(device)
        #         item_ids = torch.tensor(chunk['item_id'].values, dtype=torch.long).to(device)
        #
        #         neg_ids = [list(map(int, x.split(','))) for x in chunk['non_exer_ids'].values]
        #         neg_ids_tensors = [torch.tensor(ids, dtype=torch.long) for ids in neg_ids]
        #         neg_ids_padded = pad_sequence(neg_ids_tensors, batch_first=True, padding_value=-1).to(device)
        #
        #         scores = torch.tensor(chunk['score'].values, dtype=torch.float).to(device)
        #
        #         yield user_ids, item_ids, neg_ids_padded, scores
        #
        # batch_size = 256
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #
        # for user_ids, item_ids, neg_ids_padded, scores in read_csv_in_batches(file_path, batch_size, device):
        #     print("User IDs shape:", user_ids.shape)
        #     print("Item IDs shape:", item_ids.shape)
        #     print("Padded Neg IDs shape:", neg_ids_padded.shape)
        #     print("Scores shape:", scores.shape)
        #     sys.exit()
        # sys.exit()

        # problem_ids = []
        # # random_in_match = 0
        # for key, value in self.exer_knowledge_data.items():
        #     problem_ids.append(key)
        for count in range(self.batch_size):
            log = self.data[self.ptr + count]
            knowledge_emb = [0.] * self.knowledge_dim
            if isinstance(log['knowledge_code'], float):
                log['knowledge_code'] = [int(log['knowledge_code'])]
            if isinstance(log['knowledge_code'], str):
                log['knowledge_code'] = ast.literal_eval(log['knowledge_code'])
            log['knowledge_code'] = [int(x) for x in log['knowledge_code']]
            for knowledge_code in log['knowledge_code']:
                knowledge_emb[knowledge_code - 1] = 1.0
            y = log['score']
            # input_stu_ids.append(log['user_id'] - 1)
            # input_exer_ids.append(log['exer_id'] - 1)
            input_stu_ids.append(log['user_id'])
            input_exer_ids.append(log['exer_id'])
            # input_exer_diff.append(self.exer_difficulty[log['exer_id']])

            # if len(log['neg_exer_ids']) < self.topk:
            #     neg_pos_exer_ids.append(log['neg_exer_ids'][:self.topk])
            # else:
            #     neg_pos_exer_ids.append(np.random.choice(log['neg_exer_ids'], size=self.topk, replace=True))
            neg_pos_exer_ids.append(log['neg_exer_ids'])
            input_knowedge_embs.append(knowledge_emb)
            ys.append(y)
            
        #     matching_knowledge_code = self.exer_knowledge_data.get(log['exer_id'] - 1)
        #     # if y:
        #     result = self.get_matching_exer(self.user_interactions, log['user_id'] - 1)
        # #     # random_in_match += len(result)
        #     top_exer_ids = self.get_same_kn_exer_ids(log['user_id']-1,result, matching_knowledge_code, log['exer_id'] - 1)
        #     # top_exer_ids = self.random_exer_ids(problem_ids,log['exer_id']-1, result)
        #
        #     # inter_exer = self.user_interactions[log['user_id']-1]
        #     # random_in_inter = [item in inter_exer for item in top_exer_ids]
        #     # random_in_top = [item in top_exer_ids for item in random_top_exer_ids]
        #     # count = sum(random_in_top) / self.topk
        #     # count = sum(random_in_inter)/self.topk
        #     # random_in_match +=count
        #     # random_in_match += matching_questions_num
        #     # rand_in_exer = [item in result for item in top_exer_ids]
        #     # count = sum(rand_in_exer)
        #     # random_in_match += count
        #     neg_pos_exer_ids.append(top_exer_ids)
        #     # else:
        #     #     result = self.get_matching_exer(self.user_interactions_neg, log['user_id'] - 1)
        #     #     top_exer_ids = self.get_same_kn_exer_ids(result, matching_knowledge_code, log['exer_id'] - 1)
        #     #     neg_pos_exer_ids.append(top_exer_ids)

        # # print("neg_pos_exer_ids",neg_pos_exer_ids)
        self.ptr += self.batch_size
        return torch.LongTensor(input_stu_ids), input_exer_ids, torch.Tensor(input_knowedge_embs), torch.LongTensor(ys), neg_pos_exer_ids
    def load_data_from_csv(self,file_path):
        data = {}
        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)

            for row in reader:
                item_id = int(row[0])
                item_id = item_id - 1
                knowledge_code = eval(row[1])

                if item_id not in data:
                    data[item_id] = []

                data[item_id].extend([x - 1 for x in knowledge_code])
        data = {item_id: list(set(lst)) for item_id, lst in data.items()}
        max_length = max(len(lst) for lst in data.values())

        return data,max_length

    def load_data_from_csv_junyi(self, file_path):
        data = {}
        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)

            for row in reader:
                item_id = int(row[0])
                item_id = item_id
                knowledge_code = eval(row[1])

                if item_id not in data:
                    data[item_id] = []

                data[item_id].extend([x for x in knowledge_code])
        data = {item_id: list(set(lst)) for item_id, lst in data.items()}
        max_length = max(len(lst) for lst in data.values())

        return data, max_length

    def get_key_from_value(self,dictionary, target_value):
        for key, value in dictionary.items():
            if target_value in value:
                return key
        return None

    def get_other_users(self,cluster_data, cur_cluster):
        other_users = []

        for cluster, users in cluster_data.items():
            if cluster == cur_cluster:
                other_users.extend(users)
                break
        
        return other_users

    def get_user_interactions(self):
        user_interactions = {}

        for log in self.data:
            user_id = log['user_id']-1
            exer_id = log['exer_id']-1
            if user_id in user_interactions:
                user_interactions[user_id].append(exer_id)
            else:
                user_interactions[user_id] = [exer_id]

        return user_interactions

    def get_user_interactions_17(self):
        user_interactions = {}

        for log in self.data:
            user_id = log['user_id']
            exer_id = log['exer_id']
            if user_id in user_interactions:
                user_interactions[user_id].append(exer_id)
            else:
                user_interactions[user_id] = [exer_id]

        return user_interactions

    def get_matching_exer(self, dict_B, user_id):
        matching_students = []

        # matching_knowledge_code = self.exer_knowledge_data.get(log['exer_id'] - 1)
        cluster=self.get_key_from_value(self.clustering_results,user_id)

        matching_students=self.get_other_users(self.clustering_results,cluster)

        matching_exer_ids = []

        for stu_id in matching_students:
            if stu_id in dict_B:
                matching_exer_ids.extend(dict_B[stu_id])
        return matching_exer_ids


    def get_same_kn_exer_ids(self, stu, matching_exer_ids, choose_exer_kn, cur_exer_id):
        matching_questions = []
        for exer_id in matching_exer_ids:
            if exer_id not in self.user_interactions[stu]:
                knowledge_codes = self.exer_knowledge_data.get(exer_id)
                if knowledge_codes == choose_exer_kn:
                    matching_questions.append(exer_id)

        matching_counts = Counter(matching_questions)

        matching_questions = [element for element, count in matching_counts.most_common()]

        if len(matching_questions) < self.topk:
            if len(matching_questions) == 0:
                top_questions = [cur_exer_id] * self.topk
            else:
                fill_count = self.topk - len(matching_questions)
                sampled_elements = random.choices(matching_questions, k=fill_count)
                top_questions = matching_questions + sampled_elements
        else:
            top_questions = matching_questions

        return top_questions

    def is_end(self):
        if self.ptr + self.batch_size > len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0

    def random_exer_ids(self, problem_ids, cur_exer_id, match_exercises):
        filtered_problems = [pid for pid in problem_ids if pid != cur_exer_id]
        random.shuffle(filtered_problems)
        selected_problems = filtered_problems[:self.topk]
        return selected_problems

    def random_all(self, exer_list, user_inter):
        filtered_list1 = [item for item in exer_list if item not in user_inter]

        random.shuffle(filtered_list1)

        topk_selection = filtered_list1[:self.topk]

        return topk_selection

    def get_diff(self):
        with open('data/train_set.json', 'r') as f:
            data = json.load(f)

        exer_total_count = {}
        exer_wrong_count = {}

        for item in data:
            exer_id = item['exer_id']
            score = item['score']

            if exer_id not in exer_total_count:
                exer_total_count[exer_id] = 0
                exer_wrong_count[exer_id] = 0
            exer_total_count[exer_id] += 1

            if score == 0.0:
                exer_wrong_count[exer_id] += 1

        exer_difficulty = {}
        for exer_id in exer_total_count:
            total = exer_total_count[exer_id]
            wrong = exer_wrong_count.get(exer_id, 0)
            difficulty = wrong / total
            exer_difficulty[exer_id] = difficulty

        return exer_difficulty

class ValTestDataLoader(object):
    def __init__(self, d_type='validation'):
        self.ptr = 0
        self.data = []
        self.d_type = d_type
        # self.exer_knowledge_data, self.kn_max_length = self.load_data_from_csv('item.csv')

        # UECD_VAL_FILE lets the overfit sanity check validate on a tiny
        # subset instead of the full val set; normal runs use the real file.
        if d_type == 'validation':
            data_file = os.environ.get('UECD_VAL_FILE', 'data/Eedi/val_set.json')
        else:
            data_file = 'data/Eedi/test_set.json'
        config_file = 'config_Eedi.txt'
        print('loading %s ...' % data_file, flush=True)
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        print('loaded %d %s groups' % (len(self.data), d_type), flush=True)
        with open(config_file) as i_f:
            i_f.readline()
            _, _, knowledge_n = i_f.readline().split(',')
            self.knowledge_dim = int(knowledge_n)

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        logs = self.data[self.ptr]['logs']
        user_id = self.data[self.ptr]['user_id']
        input_stu_ids, input_exer_ids, input_knowledge_embs, ys = [], [], [], []
        for log in logs:
            input_stu_ids.append(user_id - 1)
            input_exer_ids.append(log['exer_id'] - 1)
            knowledge_emb = [0.] * self.knowledge_dim
            if isinstance(log['knowledge_code'], str):
                log['knowledge_code'] = ast.literal_eval(log['knowledge_code'])
            log['knowledge_code'] = [int(x) for x in log['knowledge_code']]
            for knowledge_code in log['knowledge_code']:
                knowledge_emb[knowledge_code - 1] = 1.0
            input_knowledge_embs.append(knowledge_emb)
            y = log['score']
            ys.append(y)
        self.ptr += 1
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowledge_embs), torch.LongTensor(ys)

    def is_end(self):
        if self.ptr >= len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0

    def load_data_from_csv(self, file_path):
        data = {}

        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)

            for row in reader:
                item_id = int(row[0])
                item_id = item_id - 1
                knowledge_code = eval(row[1])

                if item_id not in data:
                    data[item_id] = []

                data[item_id].extend(
                    [x - 1 for x in knowledge_code])
        data = {item_id: list(set(lst)) for item_id, lst in data.items()}
        max_length = max(len(lst) for lst in data.values())

        return data, max_length
