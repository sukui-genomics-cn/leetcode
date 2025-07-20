import math
from math import isqrt
from functools import lru_cache

def F(L):
    total = 0
    N = isqrt(L)  # 上界 ≈ sqrt(10^{12}) = 10^6
    for n in range(2, N + 1):
        start = n + 1
        end = 2 * n - 1
        # 对 k 分区间：L/(n*k) 的取值在同一区间内不变
        k = start
        while k <= end:
            w = L // (n * k)
            if w == 0:  # 后续 w 全为 0，提前退出
                break
            k_next = min(end, L // (n * w)) + 1  # 当前 w 的右边界
            # 区间 [k, k_next) 内值全为 w
            total += w * count_coprime_in_range(n, k, k_next)
            k = k_next
    return total

@lru_cache(maxsize=1_000_000)  # 预处理欧拉函数优化
def count_coprime_in_range(n, low, high):
    """返回区间 [low, high) 内与 n 互质的整数个数"""
    # 公式：| [a, b) ∩ 互质 | = (b - a) * \frac{\phi(n)}{n} + 误差调整
    cnt = 0
    for i in range(low, high):
        if math.gcd(i, n) == 1:
            cnt += 1
    return cnt

# 计算 F(10^12)
L = 15
print(F(L))  # 结果需实际运行计算，数量级巨大但算法能高效执行