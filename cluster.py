import json
import sys
import time

from sklearn.cluster import AgglomerativeClustering
import numpy as np
from tqdm import tqdm

n_clusters=50

print('loading train_set.json ...', flush=True)
with open('./data/Eedi/train_set.json', 'r') as file:
    data = json.load(file)
print(f'loaded {len(data)} records', flush=True)

user_ids = [item['user_id']-1 for item in data]
exer_ids = [item['exer_id']-1 for item in data]
scores = [item['score'] for item in data]
knowledge_codes = [item['knowledge_code'] for item in data]

all_knowledge_codes = set()
for item in tqdm(data, desc='collecting concepts'):
    knowledge = item['knowledge_code']
    all_knowledge_codes.update(knowledge)

user_exer_matrix = {}
for user, exer, score, codes in tqdm(
        zip(user_ids, exer_ids, scores, knowledge_codes),
        total=len(data), desc='building user matrices'):
    if user not in user_exer_matrix:
        user_exer_matrix[user] = {}
    # user_exer_matrix[user][exer] = {'score': score, 'codes': codes}
    for code in codes:
        user_exer_matrix[user][code] = {'score': score}

user_ids = list(set([item['user_id'] - 1 for item in data]))
user_knowledge_matrix = {}
for user in user_ids:
    if user not in user_knowledge_matrix:
        user_knowledge_matrix[user] = []

for item in tqdm(data, desc='collecting user concepts'):
    user = item['user_id']-1
    knowledge = item['knowledge_code']
    user_knowledge_matrix[user].extend(knowledge)

# user_knowledge_list = []
# for user, knowledge in user_knowledge_matrix.items():
#     feature_vector = [1 if code in knowledge else 0 for code in all_knowledge_codes]
#     user_knowledge_list.append(feature_vector)
user_knowledge_list = []
for user, knowledge in tqdm(user_knowledge_matrix.items(),
                            total=len(user_knowledge_matrix),
                            desc='building feature vectors'):
    feature_vector = []
    for code in all_knowledge_codes:
        if code in knowledge:
            if user_exer_matrix[user][code]['score'] == 1:
                feature_vector.append(1)
            else:
                feature_vector.append(0)
        else:
            feature_vector.append(-1)
    user_knowledge_list.append(feature_vector)

print(f'agglomerative clustering ({n_clusters} clusters, '
      f'{len(user_knowledge_list)} users) ...', flush=True)
cluster_start = time.perf_counter()
clustering = AgglomerativeClustering(n_clusters)
user_cluster_labels = clustering.fit_predict(user_knowledge_list)
print('clustering finished in %.1f s' % (time.perf_counter() - cluster_start),
      flush=True)

clusters = {}
for i, label in enumerate(user_cluster_labels):
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(user_ids[i])

new_clusters = {}
for label, users in clusters.items():
    new_clusters[str(label)] = users

save_name='data/Eedi/result'+str(n_clusters)+'.json'
with open(save_name, 'w') as file:
    json.dump(new_clusters, file)

print("Clustering completed and result saved to result.json.")
sys.exit()

# ###########处理17数据集，user_id,exer_id都从0开始编号的
# user_ids = [item['user_id'] for item in data]
# exer_ids = [item['exer_id'] for item in data]
# scores = [item['score'] for item in data]
# knowledge_codes = [item['knowledge_code'] for item in data]
#
# all_knowledge_codes = set()
# for item in data:
#     knowledge = item['knowledge_code']
#     all_knowledge_codes.update(knowledge)
#
# user_exer_matrix = {}
# for user, exer, score, codes in zip(user_ids, exer_ids, scores, knowledge_codes):
#     if user not in user_exer_matrix:
#         user_exer_matrix[user] = {}
#     # user_exer_matrix[user][exer] = {'score': score, 'codes': codes}
#     for code in codes:
#         user_exer_matrix[user][code] = {'score': score}
#
# user_ids = list(set([item['user_id'] for item in data]))
# user_knowledge_matrix = {}
# for user in user_ids:
#     if user not in user_knowledge_matrix:
#         user_knowledge_matrix[user] = []
#
# for item in data:
#     user = item['user_id']
#     knowledge = item['knowledge_code']
#     user_knowledge_matrix[user].extend(knowledge)
#
# # user_knowledge_list = []
# # for user, knowledge in user_knowledge_matrix.items():
# #     feature_vector = [1 if code in knowledge else 0 for code in all_knowledge_codes]
# #     user_knowledge_list.append(feature_vector)
#
# user_knowledge_list = []
# for user, knowledge in user_knowledge_matrix.items():
#     feature_vector = []
#     for code in all_knowledge_codes:
#         if code in knowledge:
#             if user_exer_matrix[user][code]['score'] == 1:
#                 feature_vector.append(1)
#             else:
#                 feature_vector.append(0)
#         else:
#             feature_vector.append(-1)
#     user_knowledge_list.append(feature_vector)
#
# clustering = AgglomerativeClustering(n_clusters)
# user_cluster_labels = clustering.fit_predict(user_knowledge_list)
#
# clusters = {}
# for i, label in enumerate(user_cluster_labels):
#     if label not in clusters:
#         clusters[label] = []
#     clusters[label].append(user_ids[i])
#
# new_clusters = {}
# for label, users in clusters.items():
#     new_clusters[str(label)] = users
#
# save_name='./data/a2017/result'+str(n_clusters)+'.json'
# with open(save_name, 'w') as file:
#     json.dump(new_clusters, file)
#
# print("Clustering completed and result saved to result.json.")
