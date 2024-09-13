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
