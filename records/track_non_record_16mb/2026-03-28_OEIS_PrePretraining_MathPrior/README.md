# OEIS Pre-Pretraining: Mathematical Prior via Integer Sequence Data

**Non-record submission** — Experiment testing whether pre-pretraining on synthetic
integer sequence data (OEIS-style) can build useful attention/aggregation structure
that transfers to FineWeb language modeling.

## Hypothesis

Even though the 1024-BPE tokenizer shreds numbers into character bigrams, the
sequential *structure* of integer sequences (look-back patterns, recurrences,
partial sums) might prime attention heads and MLP aggregation in ways that
accelerate early FineWeb convergence — enough to offset the time cost.

## Key Question

Does Muon's Newton-Schulz orthogonalization immediately destroy OEIS-learned structure?
If so, does Adam preserve it better during the transition?

## Experiment Matrix

| ID | Phase 1 | Phase 2 | Purpose |
|----|---------|---------|---------|
| 0 | — | Muon | Control (baseline) |
| 1 | OEIS(Adam,30s) | Muon | Core hypothesis |
| 2 | OEIS(Muon,30s) | Muon | Does Muon kill phase-1 structure? |
| 3 | OEIS(Adam,30s) | Adam(60s)→Muon | Adam warmstart preserves structure longer |
| 4 | OEIS(Adam,60s) | Muon | More phase-1 time |
| 5 | OEIS(Adam,15s) | Muon | Less phase-1 time |
| 6 | — | Adam | Adam-only control |

## Data Generation

The OEIS data generator produces ~50 base sequence families (Fibonacci, primes,
Catalan, recurrences, etc.) plus unlimited mixed sequences via:
- Element-wise addition/multiplication
- Interleaving
- Finite differences and partial sums
- Scaling, offset, modular reduction

## Running

```bash
# 1. Generate OEIS data
python records/track_non_record_16mb/2026-03-28_OEIS_PrePretraining_MathPrior/generate_oeis_data.py \
  --num-sequences 500000 --num-shards 1

# 2. Run all experiments
bash records/track_non_record_16mb/2026-03-28_OEIS_PrePretraining_MathPrior/run_experiments.sh
```

## Files

- `generate_oeis_data.py` — Synthetic OEIS data generator (outputs .bin shards)
- `train_gpt_oeis.py` — Two-phase training script with Adam/Muon switching
- `run_experiments.sh` — Full experiment suite (7 runs)

## Expected Outcome

Likely negative result. But if Exp 1 or 3 show val_bpb improvement >0.001 over
Exp 0, the approach warrants further investigation with optimized time splits
and potentially a custom number-aware tokenizer.
