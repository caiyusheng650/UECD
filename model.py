import sys

import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def partial_order_ranking_loss_with_eac(y_pos_obs, y_neg_unobs, y_pos_unobs, y_neg_obs, eac_weights):
    loss = 0.0
    sigma = nn.Sigmoid()

    for i in range(y_pos_obs.size(0)):
        loss += -eac_weights[i] * torch.log(sigma(y_pos_obs[i] - y_neg_unobs[i]))

    for i in range(y_neg_obs.size(0)):
        loss += -eac_weights[i] * torch.log(sigma(y_pos_unobs[i] - y_neg_obs[i]))

    return loss

class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)

class Net(nn.Module):
    '''
    NeuralCDM
    '''

    def __init__(self, student_n, exer_n, knowledge_n, topk, device):
        super(Net, self).__init__()
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable
        self.topk = topk
        # self.device = torch.device(('cuda:0') if torch.cuda.is_available() else 'cpu')
        self.device = device

        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_discrimination = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

        # 2 heads require an even knowledge dim (author Eedi setting: 286);
        # fall back to 1 head when the dim is odd (e.g. 281 from our filtering)
        attn_heads = 2 if self.knowledge_dim % 2 == 0 else 1
        self.multi_attention = MultiHeadAttention(self.knowledge_dim, attn_heads)
        self.lr = 0.001

    def sinusoidal_positional_encoding(self, inputs):
        seq_len = inputs.shape[1]
        d_model = inputs.shape[2]

        position_enc = torch.zeros((seq_len, d_model))
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        position_enc[:, 0::2] = torch.sin(position * div_term)
        position_enc[:, 1::2] = torch.cos(position * div_term)

        return inputs + position_enc.unsqueeze(0)

    def forward(self, stu_id, exer_id, kn_emb, pos_id=None, ground_truth=None):
        '''
        :param stu_id: LongTensor
        :param exer_id: LongTensor
        :param kn_emb: FloatTensor, the knowledge relevancy vectors
        :return: FloatTensor, the probabilities of answering correctly
        '''

        stu_emb = torch.sigmoid(self.student_emb(stu_id))

        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id))
        mse_loss = nn.MSELoss()
        tao = 0.8

        # if k_difficulty.size(1) != self.knowledge_dim:
        if k_difficulty.size(1)==self.topk:
            neg_list, neg_score_list = [], []
            qdcc_neg_list = []
            bpr_losses = 0
            qdcc_losses = 0

            k_pos = torch.sigmoid(self.k_difficulty(pos_id))
            e_disc = torch.sigmoid(self.e_discrimination(pos_id))

            # pos_input_x = e_disc * (stu_emb - k_pos)
            pos_input_x = e_disc * (stu_emb - k_pos) * kn_emb
            pos_input_x = self.drop_1(torch.sigmoid(self.prednet_full1(pos_input_x)))
            pos_input_x = self.drop_2(torch.sigmoid(self.prednet_full2(pos_input_x)))
            pos_score = torch.sigmoid(self.prednet_full3(pos_input_x))


            pos_neg_diff=torch.cat((k_difficulty,k_pos.unsqueeze(1)),dim=1)
            pos_neg_disc=torch.cat((e_discrimination,e_disc.unsqueeze(1)),dim=1)

            k_diff_fusion = self.multi_attention(pos_neg_diff).to(self.device)


            # item_pos = torch.cat((exer_id, pos_id.unsqueeze(-1)), dim=1)  # [256,21]
            # eac_weights = self.get_weights(stu_id, item_pos, kn_emb, self.device).detach()  # [256,21]
            eac_weights = self.calculate_eac_weights(pos_score, stu_emb, pos_neg_disc, k_diff_fusion)  # [256,21]

            ####计算习题相似性作为neg_label
            k_pos = k_pos.unsqueeze(1)
            sim_score = F.cosine_similarity(k_difficulty, k_pos, dim=-1)    #[256,20]

            for i in range(self.topk + 1):
                each_neg = k_diff_fusion[torch.arange(k_diff_fusion.size(0)), i, :]  # [256,123]
                neg_disc = pos_neg_disc[torch.arange(e_discrimination.size(0)), i]  # [256,123]

                # each_neg_kn_emb = pos_neg_kn_emb[torch.arange(pos_neg_kn_emb.size(0)), i, :]

                # neg_eac_weight = eac_weights[torch.arange(eac_weights.size(0)),i]    #[256]
                # each_eac = eac_weights[:, i].squeeze(-1)
                each_eac = eac_weights[:, i]

                # each_neg = k_difficulty[torch.arange(k_difficulty.size(0)), i]
                # neg_disc = e_discrimination[torch.arange(e_discrimination.size(0)), i]

                # neg_score,bpr_loss=self.predict(stu_id,pos_id,each_neg,neg_disc,ground_truth,eac_weights[:,i])
                # neg_score = self.predict(stu_id, pos_id, each_neg, neg_disc,
                #                                    ground_truth)
                # neg_score,bpr_loss = self.predict(stu_id, pos_id, each_neg, neg_disc,
                #                          ground_truth)
                # neg_input_x = neg_disc * (stu_emb - each_neg)
                # neg_input_x = ncd_net.drop_1(torch.sigmoid(ncd_net.prednet_full1(neg_input_x))) #[256,512]
                # neg_input_x = ncd_net.drop_2(torch.sigmoid(ncd_net.prednet_full2(neg_input_x)))
                # neg_score = torch.sigmoid(ncd_net.prednet_full3(neg_input_x))
                # neg_score = torch.round(neg_score).long()
                # neg_score = neg_score.long()
                # neg_score = torch.ones(256, device=self.device).long().unsqueeze(1)
                # neg_score, bpr_loss = self.predict(stu_id, pos_id, each_neg, neg_disc, ground_truth)

                pred_neg_score_input = neg_disc * (stu_emb - each_neg) * kn_emb
                input_x = self.drop_1(torch.sigmoid(self.prednet_full1(pred_neg_score_input)))
                input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
                output = torch.sigmoid(self.prednet_full3(input_x))


                pos_minus_neg = -torch.mean(torch.log(torch.sigmoid(pos_score - output)))
                neg_minus_pos = -torch.mean(torch.log(torch.sigmoid(output - pos_score)))

                bpr_loss = torch.where(ground_truth == 1, pos_minus_neg, neg_minus_pos)     #[256]
                bpr_loss = torch.mean(each_eac * bpr_loss)
                bpr_loss = torch.mean(bpr_loss)

                bpr_losses += bpr_loss

            for i in range(self.topk):
                qdcc_neg_diff = k_difficulty[torch.arange(k_difficulty.size(0)), i]
                qdcc_neg_disc = e_discrimination[torch.arange(e_discrimination.size(0)), i]
                qdcc_neg_x = qdcc_neg_disc * (stu_emb - qdcc_neg_diff) * kn_emb
                qdcc_neg_x = self.drop_1(torch.sigmoid(self.prednet_full1(qdcc_neg_x)))
                qdcc_neg_x = self.drop_2(torch.sigmoid(self.prednet_full2(qdcc_neg_x)))
                qdcc_neg_score = torch.sigmoid(self.prednet_full3(qdcc_neg_x))

                neg_list.append(qdcc_neg_score)

                each_smi_score = sim_score[torch.arange(sim_score.size(0)), i]
                adjusted_sim_score = torch.where(ground_truth == 1, each_smi_score, 1 - each_smi_score)

                # mask = (adjusted_sim_score > tao)
                # if mask.any():
                #     neg_list.append(qdcc_neg_score[mask])
                #     neg_score_list.append(adjusted_sim_score[mask])

                neg_score_list.append(adjusted_sim_score)

                # sim_score_expanded = ground_truth.unsqueeze(1).expand(-1, self.topk)
                # each_smi_score = sim_score_expanded[torch.arange(sim_score_expanded.size(0)),i]
                # neg_score_list.append(each_smi_score)



                qdcc_loss = self.compute_qdcc_loss(pos_score, qdcc_neg_score, k_pos, qdcc_neg_diff)

                qdcc_losses += qdcc_loss

            # qdcc_neg_net = torch.stack(qdcc_neg_list, dim=1)


            neg_net = torch.stack(neg_list, dim=1)  # 将负例练习的预测结果沿着新的维度进行对贴
            neg_score = torch.stack(neg_score_list, dim=1)

            return neg_net, neg_score, bpr_losses, qdcc_losses


        else:
            # input_x = e_discrimination * (stu_emb - k_difficulty)
            input_x = e_discrimination * (stu_emb - k_difficulty) * kn_emb
            input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
            input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
            output = torch.sigmoid(self.prednet_full3(input_x))
            return output

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

    def get_knowledge_status(self, stu_id):
        stat_emb = torch.sigmoid(self.student_emb(stu_id))
        return stat_emb.data

    def get_exer_params(self, exer_id):
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        return k_difficulty.data, e_discrimination.data

    def compute_qdcc_loss(self, original_predictions, alternative_predictions, original_difficulties, alternative_difficulties):
        # Compute QDCC loss
        # L = len(original_predictions)
        # qdcc_loss = (1 / L) * torch.sum((original_predictions - alternative_predictions) - (alternative_difficulties - original_difficulties))
        difficulty_diff = alternative_difficulties - original_difficulties

        prediction_diff = alternative_predictions - original_predictions

        qdcc_loss = torch.mean(torch.relu(difficulty_diff * prediction_diff))

        return qdcc_loss

    def calculate_eac_weights(self, pos_score, stu_emb, pos_neg_disc, k_diff_fusion):

        eac_weights = torch.zeros(k_diff_fusion.size()).to(k_diff_fusion.device)    #[256,21,123]

        for i in range(k_diff_fusion.size(1)):
            # # neg_id = neg_ids[:, i]
            neg_input_x = pos_neg_disc[:, i] * (stu_emb - k_diff_fusion[:, i])
            neg_input_x = self.drop_1(torch.sigmoid(self.prednet_full1(neg_input_x)))
            neg_input_x = self.drop_2(torch.sigmoid(self.prednet_full2(neg_input_x)))
            neg_score = torch.sigmoid(self.prednet_full3(neg_input_x))
            delta_Hs = torch.abs(pos_score - neg_score)
            eac_weights[:, i] = torch.exp(delta_Hs)
        eac_weights = eac_weights / torch.sum(eac_weights, dim=1, keepdim=True)
        eac_weights = torch.mean(eac_weights, dim=2)    #[256,21]

        return eac_weights

    def calculate_similarity(self, pos_diff, pos_disc, neg_diff, neg_disc):
        he = torch.stack([pos_diff, pos_disc], dim=-1)  # [batch_size, know_dim,2]
        heu = torch.stack([neg_diff, neg_disc], dim=-1)  # [batch_size, know_dim,2]
        return torch.sum(he * heu, dim=-1)

    def sampling_probability(self, pos_diff, pos_disc, neg_diff, neg_disc):
        similarities = [self.calculate_similarity(pos_diff, pos_disc, neg_diff[:, i, :], neg_disc[:, i, :]) for i in
                        range(neg_diff.size(1))]
        similarities = torch.stack(similarities, dim=1)
        total_similarity = similarities.sum(dim=1, keepdim=True)
        probabilities = similarities / total_similarity
        return probabilities

    def item_similarity(self,pos_id, neg_id,topk,device):
        # preparation: params and id tensor

        inter_item_tensor = torch.tensor(pos_id).unsqueeze(0)
        non_item_tensor = torch.tensor(neg_id).unsqueeze(-1)
        inter_item_tensor = inter_item_tensor.to(device)
        non_item_tensor = non_item_tensor.to(device)

        # difficulty
        inter_beta_e = self.k_difficulty(inter_item_tensor)
        non_beta_e = self.k_difficulty(non_item_tensor)
        dif_score = torch.sum(torch.sum(inter_beta_e * non_beta_e, dim=1), dim=-1)

        # discrimination
        inter_alp_e = self.e_discrimination(inter_item_tensor)
        non_alp_e = self.e_discrimination(non_item_tensor)
        dis_score = torch.sum(torch.sum(inter_alp_e * non_alp_e, dim=1), dim=-1)

        # sum of two embedding score
        sim_score = (F.softmax(dif_score, dim=-1) + F.softmax(dis_score, dim=-1)) / 2
        # sim_score = (dif_score + dis_score) / 2
        # sim_score = torch.clamp(sim_score, min=0)
        # sim_score = sim_score / sim_score.sum()
        sim_score_cpu = sim_score.cpu()
        # sim_score_cpu = (sim_score_cpu + 1) / 2
        # sim_score_cpu = sim_score_cpu.clamp(min=1e-8)
        # sample according to the probability
        sampled_index = torch.multinomial(sim_score_cpu, topk, replacement=True).to(device)
        # if label == 1:
        #     sample_label = sim_score[sampled_index]
        # else:
        #     sample_label = 1 - sim_score[sampled_index]


        # return inter_item_tensor, non_item_tensor[sampled_index], sample_label
        return inter_item_tensor, non_item_tensor[sampled_index]


    def cosine_similarity(self,a, b):
        # return F.cosine_similarity(a.unsqueeze(1), b.unsqueeze(0), dim=-1)
        a_expanded = a.expand_as(b)
        return F.cosine_similarity(a_expanded, b, dim=-1)

    def diversity_loss(self,similarity_matrix):
        # 假设相似度在[0,1]之间，多样性为(1 - 相似度)
        diversity_matrix = 1 - similarity_matrix
        # return torch.mean(diversity_matrix)
        return diversity_matrix


    def item_sim_sample(self, pos_ids, neg_ids, topk, device):
        can_samples = []
        pos_items = []
        sample_labels = []
        # diff_losses = 0
        for i in range(len(pos_ids)):
            # inter_item, can_item, sample_label = self.item_similarity(pos_ids[i],neg_ids[i],topk,device, labels[i])
            # inter_item, can_item, dif_loss = self.item_similarity(user_ids[i], pos_ids[i], neg_ids[i], topk, device)
            inter_item, can_item = self.item_similarity(pos_ids[i],neg_ids[i],topk,device)
            can_samples.append(can_item.clone().detach().squeeze())
            pos_items.append(inter_item.clone().detach().squeeze())
            # sample_labels.append(sample_label.clone().detach().squeeze())
            # diff_losses += dif_loss
        # return torch.stack(pos_items), torch.stack(can_samples), torch.stack(sample_labels)
        return torch.stack(pos_items), torch.stack(can_samples)


    def information_sample(self, score, can_score, candits, info_num, can_kcs=None):
        can_num = candits.size(1)
        bz = candits.size(0)
        score = score.unsqueeze(1).repeat(1, can_num)
        inf = abs(score - can_score)
        _, indices = torch.sort(inf, descending=False)
        nid = indices[:, :info_num]
        row_id = torch.arange(bz).unsqueeze(1)
        if can_kcs is not None:
            return candits[row_id, nid], can_kcs[row_id, nid]

        return candits[row_id, nid]

    def get_weights(self, users, infos, kcs, device):
        info_num = infos.size(-1)
        bz = infos.size(0)
        net = Net(self.emb_num, self.exer_n, self.stu_dim, self.topk, device).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)

        # Freeze parameters of the network except for student_emb
        for name, param in net.named_parameters():
            if 'student_emb' not in name:
                param.requires_grad = False

        original_weights = net.student_emb.weight.data.clone()
        users = users.to(device)
        infos = infos.to(device)
        kcs = kcs.to(device)
        updates = torch.tensor([], device=device)
        point_function = nn.NLLLoss()

        for i in range(info_num):
            info = infos[:, i]
            correct = torch.tensor([0] * bz, device=device,
                                   dtype=torch.long)  # Ensure correct is LongTensor with class index 0
            wrong = torch.tensor([1] * bz, device=device,
                                 dtype=torch.long)  # Ensure wrong is LongTensor with class index 1

            optimizer.zero_grad()
            pred = net(users, info, kcs)
            output_0 = torch.ones(pred.size()).to(device) - pred
            output = torch.cat((output_0, pred), 1)
            loss_correct = point_function(torch.log(output), correct)
            loss_correct.backward()
            optimizer.step()

            up_weights = net.student_emb.weight.data.clone()
            net.student_emb.weight.data.copy_(original_weights)

            optimizer.zero_grad()
            pred = net(users, info, kcs)
            output_0 = torch.ones(pred.size()).to(device) - pred
            output = torch.cat((output_0, pred), 1)
            loss_wrong = point_function(torch.log(output), wrong)
            loss_wrong.backward()
            optimizer.step()

            down_weights = net.student_emb.weight.data.clone()
            net.student_emb.weight.data.copy_(original_weights)

            # Ensure update computation is done outside of inplace operations
            update = pred * torch.norm(up_weights - original_weights).item() + \
                     (1 - pred) * torch.norm(down_weights - original_weights).item()
            updates = torch.cat((updates, update.unsqueeze(0)), dim=0)

        weights = F.softmax(updates, dim=0)
        for param in net.parameters():
            param.requires_grad = True
        return weights.transpose(0, 1)


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0, "输入维度无法被num_heads整除"

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        batch_size, num, dim = x.size()

        assert dim % self.num_heads == 0, "输入维度无法被num_heads整除"
        head_dim = dim // self.num_heads

        query = self.query(x).reshape(batch_size, num, self.num_heads, head_dim).transpose(1, 2)
        key = self.key(x).reshape(batch_size, num, self.num_heads, head_dim).transpose(1, 2)
        value = self.value(x).reshape(batch_size, num, self.num_heads, head_dim).transpose(1, 2)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))

        attention_weights = torch.softmax(attention_scores, dim=-1)
        attended_values = torch.matmul(attention_weights, value)

        attended_values = attended_values.transpose(1, 2).contiguous().reshape(batch_size, num, dim)
        output = self.fc(attended_values)

        return output


def load_snapshot(model, filename):
    f = open(filename, 'rb')
    model.load_state_dict(torch.load(f, map_location=lambda s, loc: s))
    f.close()