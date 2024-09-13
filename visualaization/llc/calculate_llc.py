import numpy as np
from sklearn.neighbors import NearestNeighbors


def calculate_llc_core(X, y, args, k=5, llc_type="A_B_N"):
    # 定义最近邻模型
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)

    # 初始化LLC数组
    LLC = np.zeros(X.shape[0])

    for i in range(X.shape[0]):
        xi = X[i]
        yi = y[i]
        neighbors_idx = indices[i][1:]  # 跳过自己
        neighbors_dist = distances[i][1:]  # 跳过自己

        llc_sum = 0
        for j, idx in enumerate(neighbors_idx):
            xj = X[idx]
            yj = y[idx]
            distance_ij = np.linalg.norm(xj - xi)
            llc_sum += ((yj - yi) ** 2) / (2 * k) * (distance_ij ** 2)

        LLC[i] = llc_sum
    if llc_type=="A_B_N":
        b_llc_sum = sum(LLC[:args.b_cnt])/args.b_cnt
        a_llc_sum = sum(LLC[args.b_cnt: args.b_cnt + args.a_cnt])/args.a_cnt
        b_a_llc_sum = sum(LLC[: args.b_cnt + args.a_cnt])/(args.b_cnt + args.a_cnt)
        non_paired_llc_sum = sum(LLC[-1 * args.non_cnt:]) / args.non_cnt
        return b_llc_sum, a_llc_sum, non_paired_llc_sum, b_a_llc_sum
    elif llc_type == "B_N":
        b_llc_sum = sum(LLC[:args.b_cnt]) / args.b_cnt

        non_paired_llc_sum = sum(LLC[-1 * args.non_cnt:]) / args.non_cnt
        return b_llc_sum, non_paired_llc_sum
    else:
        print("please check llc_type")
        return ""


import numpy as np
from sklearn.neighbors import NearestNeighbors


def calculate_llc_v2(embeddings, labels, k=5):
    # 初始化 NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn.fit(embeddings)

    # 找到 k 个最近邻
    distances, indices = nn.kneighbors(embeddings)

    llc_values = []

    for i, xi in enumerate(embeddings):
        neighbors = indices[i]
        yi = labels[i]

        llc = 0
        for j in neighbors:
            yj = labels[j]
            emb_j = embeddings[j]

            # 计算 LLC
            llc += ((yi - yj) ** 2 / (2 * k)) * np.linalg.norm(emb_j - xi) ** 2

        llc_values.append(llc)

    return np.array(llc_values)


# # 使用示例
# # embeddings = np.array(...) # 您的嵌入向量
# # labels = np.array(...) # 对应的标签，+1 或 -1
#
# # llc_results = calculate_llc(embeddings, labels)
#
# X = np.random.rand(100, 10)  # 100个样本，每个样本有10个特征
# y = np.random.choice([1, -1], size=100)  # 标签为1或-1
#
#
# llc_values = calculate_llc_core(X, y, k=5)
# print(llc_values)
