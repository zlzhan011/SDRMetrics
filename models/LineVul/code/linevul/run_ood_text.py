import os
import time
# from util.args_loader import get_args
# from util import metrics
import metrics
import torch
import faiss
import numpy as np

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# args = get_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

# cache_name = f"cache/{args.in_dataset}_train_{args.name}_in_alllayers.npy"
cache_name = f"cache/multi_train_last_second_layer.npy"
np_file = np.load(cache_name, allow_pickle=True)
feat_log, score_log, label_log = np_file['arr_0'],np_file['arr_1'],np_file['arr_2']
feat_log, score_log = feat_log.T.astype(np.float32), score_log.T.astype(np.float32)
class_num = score_log.shape[1]

# cache_name = f"cache/{args.in_dataset}_val_{args.name}_in_alllayers.npy"
cache_name = r"cache/multi_test_last_second_layer.npy"
np_file = np.load(cache_name, allow_pickle=True)
feat_log_val, score_log_val, label_log_val = np_file['arr_0'],np_file['arr_1'],np_file['arr_2']
feat_log_val, score_log_val = feat_log_val.T.astype(np.float32), score_log_val.T.astype(np.float32)
out_datasets = ['text']
ood_feat_log_all = {}
for ood_dataset in out_datasets:
    # cache_name = f"cache/{ood_dataset}vs{args.in_dataset}_{args.name}_out_alllayers.npy"
    cache_name = f"cache/multi_ood_last_second_layer.npy"
    cache_name = f"cache/multi_devign_train_last_second_layer.npy"
    cache_name = f"cache/multi_text_test_last_second_layer.npy"
    np_file = np.load(cache_name, allow_pickle=True)
    ood_feat_log, ood_score_log = np_file['arr_0'],np_file['arr_1']
    ood_feat_log, ood_score_log = ood_feat_log.T.astype(np.float32), ood_score_log.T.astype(np.float32)
    ood_feat_log_all[ood_dataset] = ood_feat_log

normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)

# prepos_feat = lambda x: np.ascontiguousarray(normalizer(x[:, range(448, 960)]))# Last Layer only
prepos_feat = lambda x: np.ascontiguousarray(normalizer(x[:, range(-774, -6)]))# Last Layer only


ftrain = prepos_feat(feat_log)
ftest = prepos_feat(feat_log_val)
food_all = {}
for ood_dataset in out_datasets:
    food_all[ood_dataset] = prepos_feat(ood_feat_log_all[ood_dataset])



#################### KNN score OOD detection #################

index = faiss.IndexFlatL2(ftrain.shape[1])
index.add(ftrain)
for K in [1,10,20,30,40,50,100,200,500]:

    D, _ = index.search(ftest, K)
    scores_in = -D[:,-1]
    all_results = []
    all_score_ood = []
    for ood_dataset, food in food_all.items():
        D, _ = index.search(food, K)
        scores_ood_test = -D[:,-1]
        all_score_ood.extend(scores_ood_test)
        results = metrics.cal_metric(scores_in, scores_ood_test)
        all_results.append(results)

    metrics.print_all_results(all_results, out_datasets, f'KNN k={K}')
    print()




# #################### SSD+ score OOD detection #################
# begin = time.time()
# mean_feat = ftrain.mean(0)
# std_feat = ftrain.std(0)
# prepos_feat_ssd = lambda x: (x - mean_feat) / (std_feat + 1e-10)
# ftrain_ssd = prepos_feat_ssd(ftrain)
# ftest_ssd = prepos_feat_ssd(ftest)
# food_ssd_all = {}
# for ood_dataset in args.out_datasets:
#     food_ssd_all[ood_dataset] = prepos_feat_ssd(food_all[ood_dataset])
#
# inv_sigma_cls = [None for _ in range(class_num)]
# covs_cls = [None for _ in range(class_num)]
# mean_cls = [None for _ in range(class_num)]
# cov = lambda x: np.cov(x.T, bias=True)
# for cls in range(class_num):
#     mean_cls[cls] = ftrain_ssd[label_log == cls].mean(0)
#     feat_cls_center = ftrain_ssd[label_log == cls] - mean_cls[cls]
#     inv_sigma_cls[cls] = np.linalg.pinv(cov(feat_cls_center))
#
# def maha_score(X):
#     score_cls = np.zeros((class_num, len(X)))
#     for cls in range(class_num):
#         inv_sigma = inv_sigma_cls[cls]
#         mean = mean_cls[cls]
#         z = X - mean
#         score_cls[cls] = -np.sum(z * (inv_sigma.dot(z.T)).T, axis=-1)
#     return score_cls.max(0)
#
# dtest = maha_score(ftest_ssd)
# all_results = []
# for name, food in food_ssd_all.items():
#     print(f"SSD+: Evaluating {name}")
#     dood = maha_score(food)
#     results = metrics.cal_metric(dtest, dood)
#     all_results.append(results)
#
# metrics.print_all_results(all_results, args.out_datasets, 'SSD+')
# print(time.time() - begin)

