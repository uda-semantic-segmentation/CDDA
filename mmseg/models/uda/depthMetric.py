from collections import Counter
from mmseg.ops import resize
import torch


def celoss(logits, label, weight):
    summ = torch.sum(label, dim=3)
    mask1 = summ > 1
    if torch.sum(mask1) > 0:
        tmp = label[mask1]
        tmp_max, _ = torch.max(tmp, dim=1)
        mask2 = tmp_max == 1
        tmp[tmp == 1] = 2 - summ[mask1][mask2]
        label[mask1] = tmp

    label = label.permute(0, 3, 1, 2)

    logits = resize(
        input=logits,
        size=label.shape[2:],
        mode='bilinear',
        align_corners=False)

    log_softmax_result = torch.nn.functional.log_softmax(logits, dim=1)

    tmp = -log_softmax_result * label
    tmp_x = torch.sum(tmp, dim=1)
    x = tmp_x * weight
    return x.mean()


class DepthMetric():
    def __init__(self, cfg):
        # dep Parameters
        self.dataset = cfg['dataset']
        self.reweight_threshold = cfg['reweight_threshold']
        self.relabel_threshold = cfg['relabel_threshold']
        self.threshold_decay = cfg['threshold_decay']

        # load depth distributions of source domain
        src_matrix_num = torch.load(cfg['matrix_path'], map_location=torch.device('cpu'))
        src_matrix_num.requires_grad_(False)
        cls_sum1 = torch.sum(src_matrix_num, dim=2, keepdim=True)
        src_matrix = src_matrix_num / cls_sum1
        src_matrix = torch.where(torch.isnan(src_matrix), torch.full_like(src_matrix, 0), src_matrix)
        self.src_matrix = src_matrix.cpu()
        self.src_matrix_num = src_matrix_num.cpu()

    def depth_math(self, pseudo_weight, dev, batch_size, pseudo_label, s2_deps, ps_large_p, local_iter):
        # threshold_decay
        if 10000 <= local_iter < 40000 and local_iter % 1000 == 0:
            self.reweight_threshold -= self.threshold_decay
            self.relabel_threshold -= self.threshold_decay
        pseudo_weight_relabel = pseudo_weight.clone()

        new_src_img_idx = []
        relabel_batch_msg = []
        relabel_flag = False
        tmp_src_matrix = self.src_matrix.to(dev)
        tmp_src_matrix_num = self.src_matrix_num.to(dev)
        for i in range(batch_size):

            sudo_seg = pseudo_label[i].to(dev)
            tgt_depth = s2_deps[i].to(dev)

            # get depth distribution of target image
            s2_matrix_num = self.create_matrix(sudo_seg, tgt_depth, 19).to(dev)
            s2_matrix = s2_matrix_num / torch.sum(s2_matrix_num, dim=1, keepdim=True)
            s2_matrix = torch.where(torch.isnan(s2_matrix), torch.full_like(s2_matrix, 0), s2_matrix)
            s2_matrix = torch.unsqueeze(s2_matrix, dim=0)

            s2_matrix_num = torch.unsqueeze(s2_matrix_num, dim=0)
            if self.dataset == 'synthia':
                preprocessed_nums = 9400
            elif self.dataset == 'gta':
                preprocessed_nums = 24966
            else:
                print('input correct dataset name!!')
                exit(0)

            s2_matrix = s2_matrix.repeat(preprocessed_nums, 1, 1)

            s2_matrix_num = s2_matrix_num.repeat(preprocessed_nums, 1, 1)

            # similarity vec for image
            s1_cls_sum = torch.sum(tmp_src_matrix_num, dim=2)
            s2_cls_sum = torch.sum(s2_matrix_num, dim=2)
            total_cls_cos = torch.cosine_similarity(s1_cls_sum, s2_cls_sum, dim=1)

            # similarity vec for class
            total_simi_cos = torch.cosine_similarity(tmp_src_matrix, s2_matrix, dim=2)

            # similarity num for class
            total_cls_sum = torch.sum(s2_matrix_num + tmp_src_matrix_num, dim=2, keepdim=True)
            src_matrix_tmp = tmp_src_matrix_num / total_cls_sum
            tgt_matrix_tmp = s2_matrix_num / total_cls_sum
            src_matrix_tmp = torch.where(torch.isnan(src_matrix_tmp), torch.full_like(src_matrix_tmp, 0),
                                         src_matrix_tmp)
            tgt_matrix_tmp = torch.where(torch.isnan(tgt_matrix_tmp), torch.full_like(tgt_matrix_tmp, 0),
                                         tgt_matrix_tmp)
            total_simi_num = self.l1_dis(src_matrix_tmp, tgt_matrix_tmp)

            # final similarity
            total_simi = (total_simi_cos + 1 - total_simi_num) / 2  #
            mask = torch.sum(s2_matrix_num, dim=2) == 0
            total_simi[mask] = 0
            total = torch.sum(total_simi, dim=1) * total_cls_cos
            max_idx = torch.argmax(total, dim=0)
            most_simi = total_simi[max_idx]

            tgt_cls_num = 19 - torch.sum(mask[max_idx])
            total_mean = total[max_idx] / tgt_cls_num

            tmp_msg = {'flag': False}

            # reweight
            if total_mean > self.reweight_threshold:
                new_src_img_idx.append(max_idx)
                for cls in range(19):
                    if self.dataset == 'synthia':
                        if cls in [9, 14, 16]:
                            continue
                    cls_sum = torch.sum(s2_matrix_num[0, cls, :])
                    if cls_sum == 0:
                        continue

                    cls_mask = sudo_seg[0] == cls

                    tmp = most_simi[cls]
                    tmp[tmp < 0] = 0
                    if tmp > 0.75:
                        pseudo_weight[i][cls_mask & ps_large_p[i]] = 1
                    elif tmp < 0.1:
                        pseudo_weight[i][cls_mask] = 0
                    else:
                        pseudo_weight[i][cls_mask] = (tmp - 0.1) / 0.65
            else:
                new_src_img_idx.append(-1)

            # relabel
            if total_mean > self.relabel_threshold:
                relabel_flag = True
                cls_mask = (torch.sum(s2_matrix_num[0, :, :], dim=1) == 0) & (
                        torch.sum(tmp_src_matrix_num[max_idx, :, :], dim=1) != 0)
                tgt_dep_num = torch.sum(s2_matrix_num[0], dim=0)
                tmp_msg['flag'] = True
                tmp_msg['cls_mask'] = cls_mask
                tmp_msg['tgt_dep_num'] = tgt_dep_num
                tmp_msg['max_idx'] = max_idx
            relabel_batch_msg.append(tmp_msg)

        # relabel
        relabel_loss_flag = False
        if relabel_flag:
            pseudo_label = torch.unsqueeze(pseudo_label, dim=1)
            tgt_pseudo_weight = pseudo_weight_relabel * 0.1

            pseudo_label[pseudo_label == 255] = 19
            tgt_onehot_lbl = torch.nn.functional.one_hot(pseudo_label, num_classes=20)[:, :, :, :, :19].float()
            tgt_onehot_lbl = torch.squeeze(tgt_onehot_lbl, 1)

            for i in range(batch_size):
                if relabel_batch_msg[i]['flag']:
                    # ===relabel
                    cls_mask = relabel_batch_msg[i]['cls_mask']
                    max_idx = relabel_batch_msg[i]['max_idx']
                    total_dep_num = torch.sum(tmp_src_matrix_num[max_idx], dim=0)
                    for cls_idx, cls_flag in enumerate(cls_mask):
                        if not cls_flag: continue
                        src_dep_num = tmp_src_matrix_num[max_idx][cls_idx]
                        dep_weight = (src_dep_num / total_dep_num)
                        dep_weight[(dep_weight >= 1)] = 0.99999
                        dep_weight[torch.isnan(dep_weight)] = 0

                        nozero_idxs = torch.where(dep_weight!=0)[0].cpu()
                        combine = torch.cat([s2_deps[i].unique().view(-1),nozero_idxs])
                        _,counts = combine.unique(return_counts=True)
                        relabel_loss_flag = torch.any(counts>1)
                        if relabel_loss_flag:
                            s2_deps_1 = s2_deps[i].long()
                            nozero_mask = dep_weight != 0
                            tgt_onehot_lbl[i, :, :, cls_idx] = torch.where(nozero_mask[s2_deps_1], dep_weight[s2_deps_1],
                                                                           tgt_onehot_lbl[i, :, :, cls_idx])

            return pseudo_weight.to(dev), new_src_img_idx, relabel_loss_flag, tgt_onehot_lbl.to(
                dev), tgt_pseudo_weight.to(dev)

        return pseudo_weight.to(dev), new_src_img_idx, relabel_loss_flag, None, None


    def create_matrix(self, seg_img, depth_img, cls_num):
        temp_depth = torch.zeros((cls_num, 256))
        valid_indices = torch.nonzero((seg_img < cls_num) & (depth_img < 256))
        seg_vals = seg_img[valid_indices[:, 0], valid_indices[:, 1]].long()

        depth_vals = depth_img[valid_indices[:, 0], valid_indices[:, 1]].long()
        counts = Counter(zip(seg_vals.tolist(), depth_vals.tolist()))
        for (seg, depth), count in counts.items():
            temp_depth[seg, depth] += count

        return temp_depth

    def l1_dis(self, matrix1, matrix2):
        tmp = torch.abs(matrix1 - matrix2)
        dis = torch.sum(tmp, dim=2)
        return dis

