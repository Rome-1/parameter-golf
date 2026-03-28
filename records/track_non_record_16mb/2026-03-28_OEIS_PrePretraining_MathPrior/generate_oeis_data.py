"""
Generate synthetic OEIS-style integer sequence data as .bin shards
compatible with the parameter-golf training pipeline.

Sequences are formatted as text (e.g., "1, 1, 2, 3, 5, 8, 13, 21\n")
then tokenized with the 1024-BPE SentencePiece model and packed into
the standard shard binary format (256-int32 header + uint16 tokens).

Supports mixing operations (element-wise addition, scaling, interleaving)
to generate arbitrarily large datasets from ~50 base sequence families.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import random
import struct
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import sentencepiece as spm

# ─── Sequence Generators ───────────────────────────────────────────────────

def fibonacci(n: int) -> list[int]:
    a, b = 0, 1
    out = []
    for _ in range(n):
        out.append(a)
        a, b = b, a + b
    return out

def lucas(n: int) -> list[int]:
    a, b = 2, 1
    out = []
    for _ in range(n):
        out.append(a)
        a, b = b, a + b
    return out

def tribonacci(n: int) -> list[int]:
    a, b, c = 0, 0, 1
    out = []
    for _ in range(n):
        out.append(a)
        a, b, c = b, c, a + b + c
    return out

def primes(n: int) -> list[int]:
    out = []
    candidate = 2
    while len(out) < n:
        if all(candidate % p != 0 for p in out if p * p <= candidate):
            out.append(candidate)
        candidate += 1
    return out

def triangular(n: int) -> list[int]:
    return [k * (k + 1) // 2 for k in range(n)]

def square_numbers(n: int) -> list[int]:
    return [k * k for k in range(n)]

def pentagonal(n: int) -> list[int]:
    return [k * (3 * k - 1) // 2 for k in range(n)]

def catalan(n: int) -> list[int]:
    out = [1]
    for k in range(1, n):
        out.append(out[-1] * 2 * (2 * k - 1) // (k + 1))
    return out

def factorial(n: int) -> list[int]:
    out = [1]
    for k in range(1, n):
        out.append(out[-1] * k)
    return out

def powers_of_2(n: int) -> list[int]:
    return [2**k for k in range(n)]

def powers_of_3(n: int) -> list[int]:
    return [3**k for k in range(n)]

def bell_numbers(n: int) -> list[int]:
    """Bell numbers via triangle."""
    b = [[0] * (n + 1) for _ in range(n + 1)]
    b[0][0] = 1
    for i in range(1, n):
        b[i][0] = b[i - 1][i - 1]
        for j in range(1, i + 1):
            b[i][j] = b[i][j - 1] + b[i - 1][j - 1]
    return [b[i][0] for i in range(n)]

def motzkin(n: int) -> list[int]:
    """Motzkin numbers."""
    m = [1, 1]
    for k in range(2, n):
        m.append(((2 * k + 1) * m[-1] + 3 * (k - 1) * m[-2]) // (k + 2))
    return m[:n]

def collatz_length(start: int) -> int:
    n, count = start, 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        count += 1
    return count

def collatz_lengths(n: int) -> list[int]:
    return [collatz_length(k) if k > 0 else 0 for k in range(n)]

def partition_numbers(n: int) -> list[int]:
    """Number of partitions of k."""
    p = [0] * (n + 1)
    p[0] = 1
    for k in range(1, n + 1):
        for j in range(k, n + 1):
            p[j] += p[j - k]
    return p[:n]

def look_and_say(n: int) -> list[int]:
    """Look-and-say sequence (as integers)."""
    out = [1]
    s = "1"
    for _ in range(n - 1):
        new_s = ""
        i = 0
        while i < len(s):
            ch = s[i]
            count = 1
            while i + count < len(s) and s[i + count] == ch:
                count += 1
            new_s += str(count) + ch
            i += count
        s = new_s
        val = int(s) if len(s) < 15 else int(s[:15])  # cap to avoid huge ints
        out.append(val)
    return out

def arithmetic_progression(n: int, start: int = 0, step: int = 1) -> list[int]:
    return [start + k * step for k in range(n)]

def geometric_like(n: int, base: int = 2, offset: int = 0) -> list[int]:
    return [base**k + offset for k in range(n)]

def recurrence_2(n: int, a0: int, a1: int, c0: int, c1: int) -> list[int]:
    """a(n) = c0*a(n-1) + c1*a(n-2)"""
    out = [a0, a1]
    for _ in range(n - 2):
        out.append(c0 * out[-1] + c1 * out[-2])
    return out[:n]

def recurrence_3(n: int, a0: int, a1: int, a2: int, c0: int, c1: int, c2: int) -> list[int]:
    """a(n) = c0*a(n-1) + c1*a(n-2) + c2*a(n-3)"""
    out = [a0, a1, a2]
    for _ in range(n - 3):
        out.append(c0 * out[-1] + c1 * out[-2] + c2 * out[-3])
    return out[:n]

def euler_totient(n: int) -> list[int]:
    """Euler's totient function phi(k) for k=0..n-1."""
    out = [0]
    for k in range(1, n):
        result = k
        p = 2
        temp = k
        while p * p <= temp:
            if temp % p == 0:
                while temp % p == 0:
                    temp //= p
                result -= result // p
            p += 1
        if temp > 1:
            result -= result // temp
        out.append(result)
    return out

def divisor_count(n: int) -> list[int]:
    """Number of divisors of k for k=0..n-1."""
    out = [0]
    for k in range(1, n):
        count = 0
        for d in range(1, int(k**0.5) + 1):
            if k % d == 0:
                count += 2 if d * d != k else 1
        out.append(count)
    return out

def digit_sum(n: int) -> list[int]:
    return [sum(int(d) for d in str(k)) for k in range(n)]

def alternating_factorial(n: int) -> list[int]:
    """a(n) = sum_{k=1}^{n} (-1)^{n-k} k!"""
    out = []
    for m in range(n):
        s = 0
        f = 1
        for k in range(1, m + 1):
            f *= k
            s += ((-1) ** (m - k)) * f
        out.append(s)
    return out

# ─── Base sequence catalog ─────────────────────────────────────────────────

BASE_GENERATORS: list[tuple[str, Callable[[int], list[int]]]] = [
    ("fibonacci", fibonacci),
    ("lucas", lucas),
    ("tribonacci", tribonacci),
    ("primes", primes),
    ("triangular", triangular),
    ("squares", square_numbers),
    ("pentagonal", pentagonal),
    ("catalan", catalan),
    ("factorial", factorial),
    ("powers_of_2", powers_of_2),
    ("powers_of_3", powers_of_3),
    ("bell", bell_numbers),
    ("motzkin", motzkin),
    ("collatz_lengths", collatz_lengths),
    ("partitions", partition_numbers),
    ("look_and_say", look_and_say),
    ("euler_totient", euler_totient),
    ("divisor_count", divisor_count),
    ("digit_sum", digit_sum),
    ("alternating_factorial", alternating_factorial),
]


def make_mixed_generators(rng: random.Random) -> list[tuple[str, Callable[[int], list[int]]]]:
    """Generate additional sequences via mixing operations."""
    mixed = []

    # Arithmetic progressions with varied parameters
    for start in range(0, 10, 3):
        for step in [1, 2, 3, 5, 7, 11]:
            name = f"arith_s{start}_d{step}"
            s, d = start, step
            mixed.append((name, lambda n, s=s, d=d: arithmetic_progression(n, s, d)))

    # Geometric-like with offsets
    for base in [2, 3, 5]:
        for offset in [-1, 0, 1, 2]:
            name = f"geo_b{base}_o{offset}"
            b, o = base, offset
            mixed.append((name, lambda n, b=b, o=o: geometric_like(n, b, o)))

    # 2-term recurrences with varied coefficients
    for c0 in [1, 2, 3]:
        for c1 in [1, -1, 2]:
            for a0, a1 in [(0, 1), (1, 1), (1, 2), (2, 1)]:
                name = f"rec2_{a0}_{a1}_{c0}_{c1}"
                _a0, _a1, _c0, _c1 = a0, a1, c0, c1
                mixed.append((name, lambda n, a0=_a0, a1=_a1, c0=_c0, c1=_c1: recurrence_2(n, a0, a1, c0, c1)))

    # 3-term recurrences
    for c0, c1, c2 in [(1, 1, 1), (1, 0, 1), (2, 1, 0), (1, 1, -1)]:
        name = f"rec3_{c0}_{c1}_{c2}"
        _c0, _c1, _c2 = c0, c1, c2
        mixed.append((name, lambda n, c0=_c0, c1=_c1, c2=_c2: recurrence_3(n, 0, 0, 1, c0, c1, c2)))

    return mixed


# ─── Mixing operations ─────────────────────────────────────────────────────

def mix_add(a: list[int], b: list[int]) -> list[int]:
    n = min(len(a), len(b))
    return [a[i] + b[i] for i in range(n)]

def mix_multiply(a: list[int], b: list[int]) -> list[int]:
    n = min(len(a), len(b))
    return [a[i] * b[i] for i in range(n)]

def mix_interleave(a: list[int], b: list[int]) -> list[int]:
    out = []
    for x, y in zip(a, b):
        out.extend([x, y])
    return out

def mix_diff(seq: list[int]) -> list[int]:
    return [seq[i + 1] - seq[i] for i in range(len(seq) - 1)]

def mix_partial_sums(seq: list[int]) -> list[int]:
    out = []
    s = 0
    for x in seq:
        s += x
        out.append(s)
    return out

def mix_scale(seq: list[int], factor: int) -> list[int]:
    return [x * factor for x in seq]

def mix_offset(seq: list[int], offset: int) -> list[int]:
    return [x + offset for x in seq]

def mix_modular(seq: list[int], mod: int) -> list[int]:
    return [x % mod for x in seq]


# ─── Sequence formatting and tokenization ──────────────────────────────────

def format_sequence(seq: list[int], max_terms: int = 50) -> str:
    """Format a sequence as comma-separated integers, capping term magnitude."""
    # Cap individual terms to avoid extremely long token sequences for huge numbers
    capped = []
    for x in seq[:max_terms]:
        if abs(x) > 10**12:
            break
        capped.append(x)
    if not capped:
        return ""
    return ", ".join(str(x) for x in capped) + "\n"


def generate_all_sequences(
    rng: random.Random,
    num_sequences: int,
    seq_length: int = 40,
    mix_probability: float = 0.5,
) -> list[str]:
    """Generate a batch of formatted sequences, optionally mixed."""
    base_gens = BASE_GENERATORS[:]
    mixed_gens = make_mixed_generators(rng)
    all_gens = base_gens + mixed_gens

    sequences: list[str] = []

    # Phase 1: Raw base sequences with varied lengths
    for name, gen in all_gens:
        for length in range(10, seq_length + 1, 5):
            try:
                seq = gen(length)
                text = format_sequence(seq)
                if text:
                    sequences.append(text)
            except (OverflowError, ZeroDivisionError, ValueError):
                continue

    # Phase 2: Mixing until we reach target count
    mix_ops = [
        ("add", mix_add),
        ("mul", mix_multiply),
        ("interleave", mix_interleave),
    ]
    unary_ops = [
        ("diff", mix_diff),
        ("partial_sums", mix_partial_sums),
    ]

    attempts = 0
    while len(sequences) < num_sequences and attempts < num_sequences * 3:
        attempts += 1
        try:
            if rng.random() < mix_probability and len(all_gens) >= 2:
                # Binary mix
                _, gen_a = rng.choice(all_gens)
                _, gen_b = rng.choice(all_gens)
                length = rng.randint(10, seq_length)
                a = gen_a(length)
                b = gen_b(length)

                op_name, op = rng.choice(mix_ops)
                seq = op(a, b)

                # Optionally chain a unary op
                if rng.random() < 0.3 and seq:
                    uname, uop = rng.choice(unary_ops)
                    seq = uop(seq)

                # Optionally apply scale/offset/mod
                if rng.random() < 0.3 and seq:
                    factor = rng.choice([2, 3, 5, -1, 10])
                    seq = mix_scale(seq, factor)
                if rng.random() < 0.2 and seq:
                    offset = rng.choice([-1, 0, 1, 2, -2, 10])
                    seq = mix_offset(seq, offset)
                if rng.random() < 0.15 and seq:
                    mod = rng.choice([7, 10, 13, 100, 1000])
                    seq = mix_modular(seq, mod)
            else:
                # Unary transform on a base sequence
                _, gen = rng.choice(all_gens)
                length = rng.randint(15, seq_length)
                seq = gen(length)
                uname, uop = rng.choice(unary_ops)
                seq = uop(seq)

                if rng.random() < 0.3 and seq:
                    factor = rng.choice([2, 3, -1])
                    seq = mix_scale(seq, factor)

            text = format_sequence(seq)
            if text and len(text) > 5:  # skip trivially short
                sequences.append(text)
        except (OverflowError, ZeroDivisionError, ValueError, IndexError):
            continue

    rng.shuffle(sequences)
    return sequences[:num_sequences]


# ─── Shard writing ─────────────────────────────────────────────────────────

SHARD_MAGIC = 20240520
SHARD_VERSION = 1

def write_shard(path: Path, token_ids: list[int]) -> int:
    """Write a .bin shard in the parameter-golf format. Returns token count."""
    n = len(token_ids)
    header = np.zeros(256, dtype="<i4")
    header[0] = SHARD_MAGIC
    header[1] = SHARD_VERSION
    header[2] = n
    tokens = np.array(token_ids, dtype="<u2")
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.tobytes())
    return n


def tokenize_sequences(sp: spm.SentencePieceProcessor, texts: list[str]) -> list[int]:
    """Tokenize a list of text sequences into a flat token ID list."""
    all_ids: list[int] = []
    for text in texts:
        ids = sp.encode(text, out_type=int)
        all_ids.extend(ids)
    return all_ids


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate OEIS-style sequence data shards")
    parser.add_argument("--tokenizer", default="./data/tokenizers/fineweb_1024_bpe.model",
                        help="Path to SentencePiece .model file")
    parser.add_argument("--output-dir", default="./data/datasets/oeis_synthetic_sp1024",
                        help="Output directory for .bin shards")
    parser.add_argument("--num-sequences", type=int, default=500_000,
                        help="Total number of sequences to generate")
    parser.add_argument("--tokens-per-shard", type=int, default=100_000_000,
                        help="Target tokens per shard (matches FineWeb shard size)")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Number of shards to produce")
    parser.add_argument("--seq-length", type=int, default=40,
                        help="Max terms per sequence")
    parser.add_argument("--mix-probability", type=float, default=0.5,
                        help="Probability of generating mixed (vs base) sequences")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    print(f"Tokenizer vocab size: {sp.vocab_size()}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Show tokenization examples
    examples = [
        "1, 1, 2, 3, 5, 8, 13, 21, 34, 55",
        "0, 1, 4, 9, 16, 25, 36, 49, 64, 81",
        "2, 3, 5, 7, 11, 13, 17, 19, 23, 29",
    ]
    print("\n--- Tokenization examples ---")
    for ex in examples:
        ids = sp.encode(ex, out_type=int)
        pieces = sp.encode(ex, out_type=str)
        print(f"  Text: {ex}")
        print(f"  IDs ({len(ids)} tokens): {ids[:20]}...")
        print(f"  Pieces: {pieces[:20]}...")
        print()

    total_tokens = 0
    for shard_idx in range(args.num_shards):
        seqs_per_shard = args.num_sequences // args.num_shards
        print(f"Generating shard {shard_idx}: {seqs_per_shard} sequences...")

        texts = generate_all_sequences(
            rng,
            num_sequences=seqs_per_shard,
            seq_length=args.seq_length,
            mix_probability=args.mix_probability,
        )
        print(f"  Generated {len(texts)} text sequences")

        token_ids = tokenize_sequences(sp, texts)
        print(f"  Tokenized: {len(token_ids)} tokens")

        # If we need to pad to reach target shard size, repeat
        if args.tokens_per_shard and len(token_ids) < args.tokens_per_shard:
            repeats = (args.tokens_per_shard // len(token_ids)) + 1
            token_ids = (token_ids * repeats)[:args.tokens_per_shard]
            print(f"  Padded to: {len(token_ids)} tokens (repeated {repeats}x)")

        shard_path = output_dir / f"oeis_train_{shard_idx:06d}.bin"
        n = write_shard(shard_path, token_ids)
        total_tokens += n
        print(f"  Wrote: {shard_path} ({n} tokens, {shard_path.stat().st_size} bytes)")

    print(f"\nTotal: {total_tokens} tokens across {args.num_shards} shards")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
