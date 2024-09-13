
import numpy as np
def count_inversions(arr):
    n = len(arr)
    inv_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] > arr[j]:
                inv_count += 1
    return inv_count

def normalize_inversions(inv_count, n):
    return 2 * inv_count / (n * (n - 1))




def get_normalized_inversions(arr):
    # 计算逆序数
    inversions1 = count_inversions(arr)

    # 归一化
    norm_inv1 = normalize_inversions(inversions1, len(arr))
    return norm_inv1


def inversions(array):
    # 计算每个1*768元素到第一个1*768元素的距离
    first_element  = array[0, :]
    last_element = array[-1, :]
    distances_first = np.linalg.norm(array - first_element, axis=1)
    distances_last = np.linalg.norm(array - last_element, axis=1)

    any_element = array

    # 计算向量AB
    vector_AB = last_element - first_element

    # 计算向量CA和CB (对于数组中的每个any_element)
    vector_CA = first_element - any_element
    vector_CB = last_element - any_element

    # 计算CA与AB形成的角度的余弦值
    cosine_CA_AB = np.dot(vector_CA, vector_AB) / (np.linalg.norm(vector_CA, axis=1) * np.linalg.norm(vector_AB))

    # 计算CB与BA（即AB的反方向，因此只需改变AB的符号）形成的角度的余弦值
    # 注意：BA向量即为-AB，但在计算余弦值时，向量方向的变化不影响结果，因此CB与AB的余弦值计算相同
    cosine_CB_BA = np.dot(vector_CB, -vector_AB) / (np.linalg.norm(vector_CB, axis=1) * np.linalg.norm(vector_AB))

    cosine_CA_AB = np.where(cosine_CA_AB > 0, 1, -1)
    cosine_CB_BA = np.where(cosine_CB_BA > 0, 1, -1)
    # print(distances_first)
    # print(cosine_CA_AB)
    distances_first = np.array(cosine_CA_AB) * np.array(distances_first)
    distances_last = np.array(cosine_CB_BA) * np.array(distances_last)

    normalized_inversions_first = get_normalized_inversions(distances_first)
    normalized_inversions_last = get_normalized_inversions(distances_last)
    return normalized_inversions_first, normalized_inversions_last


# 测试列表
arr1 = [3, 1, 2, 5, 4]
arr2 = [1, 2, 3, 5, 4]

# 计算逆序数
inversions1 = count_inversions(arr1)
inversions2 = count_inversions(arr2)

# 归一化
norm_inv1 = normalize_inversions(inversions1, len(arr1))
norm_inv2 = normalize_inversions(inversions2, len(arr2))



print(f"列表1的归一化逆序数: {norm_inv1}")
print(f"列表2的归一化逆序数: {norm_inv2}")

# 比较两个列表
if norm_inv1 < norm_inv2:
    print("列表1排序程度更接近完全有序。")
elif norm_inv1 > norm_inv2:
    print("列表2排序程度更接近完全有序。")
else:
    print("两个列表的排序程度相同。")
