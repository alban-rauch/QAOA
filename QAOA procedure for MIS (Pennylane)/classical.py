"""
classical.py
============
Classical resolution.
"""

import graphs as gph


def generate_bitstrings(n):
    if n == 1:
        return ['0', '1']
    small = generate_bitstrings(n-1)
    one = ['0' + b for b in small]
    two = ['1' + b for b in small]
    return one + two

def best_config(graph):
    n = len(graph)
    pool = generate_bitstrings(n)
    best_score = None
    best_bit = []
    for b in pool:
        if gph.is_legal(graph, b):
            score = sum(int(b[i]) for i in range(n))
            if best_score is None or best_score < score:
                best_score = score
                best_bit = [b]
            elif best_score == score:
                best_bit.append(b)
    return (best_score, best_bit[0])


def best_cost(graph):
    _, bitstring = best_config(graph)
    return sum(int(bitstring[i]) for i in range(len(bitstring)))
