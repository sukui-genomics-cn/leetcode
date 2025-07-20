import math

def get_factors(n):
    if n <= 0:
        raise ValueError("输入必须是正整数")
    factors = set()
    # 遍历从1到sqrt(n)，找到所有因数对
    for i in range(1, math.isqrt(n) + 1):
        if n % i == 0:
            factors.add(i)
            factors.add(n // i)
    return sorted(factors)  # 返回排序后的因数列表

# 示例使用
if __name__ == "__main__":
    num = int(input("请输入一个正整数: "))
    factors = get_factors(num)
    print(f"nums: {len(factors)}, 因数列表: {factors}")