import math
import sys
from decimal import Decimal
from functools import reduce
from collections import Counter
from itertools import product

class PrimeFactorizer:
    """A class for finding the smallest integer with exactly k factors."""
    
    def __init__(self, max_primes=100):
        """Initialize with a list of primes."""
        self.primes = self._generate_primes(max_primes)
    
    @staticmethod
    def _generate_primes(n):
        """Generate first n primes using Sieve of Eratosthenes."""
        primes = []
        num = 2
        while len(primes) < n:
            is_prime = True
            for p in primes:
                if p * p > num:
                    break
                if num % p == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(num)
            num += 1
        return primes
    
    def _factor_decompose(self, k):
        """Find all non-increasing factor sequences of k (factors >= 2)."""
        results = []
        
        def _decompose(n, last, path):
            if n == 1:
                results.append(path.copy())
                return
            for factor in range(min(n, last), 1, -1):
                if n % factor == 0:
                    path.append(factor)
                    _decompose(n // factor, factor, path)
                    path.pop()
        
        _decompose(k, k, [])
        return results
    
    def get_min_num(self, k):
        """Return the smallest integer with exactly k factors."""
        if k == 1:
            return 1
        decomp_list = self._factor_decompose(k)
        min_n = float('inf')
        
        for dec in decomp_list:
            exponents = [d - 1 for d in dec]
            n = 1
            for i in range(len(dec)):
                n *= self.primes[i] ** exponents[i]
                if n > min_n:
                    break
            if n < min_n:
                min_n = n
        return min_n

    def get_prime_factors(self, n):
        """Return the prime factorization of n as a dictionary."""
        if n == 1:
            return {}
        
        factors = {}
        remaining = n
        
        for p in self.primes:
            if p * p > remaining:
                break
            while remaining % p == 0:
                factors[p] = factors.get(p, 0) + 1
                remaining = remaining // p
        
        if remaining > 1:
            factors[remaining] = factors.get(remaining, 0) + 1
        
        return factors

    def get_factors(self, n):
        """Return all factors of n in sorted order."""
        if n == 1:
            return [1]
        
        prime_factors = self.get_prime_factors(n)
        if not prime_factors:
            return [1]
        
        # Generate all possible exponent combinations
        factor_combinations = []
        for prime, exp in prime_factors.items():
            factor_combinations.append([prime**e for e in range(exp + 1)])
        
        # Multiply all combinations to get factors
        factors = set()
        for combo in product(*factor_combinations):
            factors.add(reduce(lambda x, y: x * y, combo))
        
        return sorted(factors)
    
    def get_sn_sets(self, l=36):
        # sqrt_l = int(math.sqrt(l))+1
        ava_sets = 0
        for n in range(2, l+1):
            ava_n = n + n
            if n < int(math.sqrt(l)):
                prime_sets = self.get_factors(n**2)
                mid_idx = len(prime_sets)//2
                ava_sets += mid_idx
                print(f"n: {n}, {prime_sets}; mid_idx: {mid_idx}")
            elif ava_n <= l:
                prime_sets = self.get_factors(n**2)
                mid_idx = len(prime_sets)//2
                min_prime_set_value = n + prime_sets[-mid_idx]
                if min_prime_set_value <= l:
                    sn_sets = [(prime_sets[i]+n,prime_sets[-i-1]+n) for i in range(mid_idx) if prime_sets[-i-1]+n <= l]
                    print(f"n: {n}, {prime_sets}; sn: {sn_sets}")
                    ava_sets += len(sn_sets)
        print(f"ava sets: {ava_sets}")


if __name__ == "__main__":
    import numpy as np
    sys.set_int_max_str_digits(1_000_000)  # Allow very large integers to be printed

    factorizer = PrimeFactorizer(max_primes=1000)
    # test_cases = [5, 225, 1_999, 7_999_999]
    # for tau in test_cases:
    #     min_num = factorizer.get_min_num(tau)
    #     print(f"=====Number of solutions(s(n)): {(tau+1)/2:.0f}=====")
    #     print(f"Number of factors: tau = {tau}")
    #     print(f"Minimal n² ≈ {Decimal(min_num):.3e}")
    #     print(f"Minimal n ≈ {Decimal(math.isqrt(min_num)):.3e}")

    # large_num = 16
    # print(f"\nPrime factors of {large_num}: {factorizer.get_prime_factors(large_num)}")
    # sn_sets = np.array(factorizer.get_factors(large_num)) # + large_num
    # for i in range(len(sn_sets)//2):
    #     print(f"prime_sets[{i}]: {sn_sets[i]} {sn_sets[-i-1]}, sn_set: {sn_sets[i]+math.sqrt(large_num)} {sn_sets[-i-1]+math.sqrt(large_num)} add {large_num}")
    # print(f"All factors of {large_num}: sn_sets: {sn_sets}")

    # sn sets
    factorizer.get_sn_sets(16)