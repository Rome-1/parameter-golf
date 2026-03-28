#!/bin/bash
# OEIS Pre-Pretraining Experiment Suite
#
# Run from the pgolf root directory:
#   bash records/track_non_record_16mb/2026-03-28_OEIS_PrePretraining_MathPrior/run_experiments.sh
#
# Prerequisites:
#   1. Download FineWeb data: python data/cached_challenge_fineweb.py --variant sp1024
#   2. Generate OEIS data:
#      python records/track_non_record_16mb/2026-03-28_OEIS_PrePretraining_MathPrior/generate_oeis_data.py \
#        --num-sequences 500000 --num-shards 1 --tokens-per-shard 50000000
#
# Each experiment runs for 10 minutes on 8xH100.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train_gpt_oeis.py"
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

LAUNCH="torchrun --nproc_per_node=8"

echo "============================================================"
echo "Experiment 0: CONTROL (no OEIS, Muon only — matches baseline)"
echo "============================================================"
OEIS_PHASE1_SECONDS=0 \
PHASE2_OPTIMIZER=muon \
RUN_ID=oeis_exp0_control \
  $LAUNCH "$TRAIN_SCRIPT" 2>&1 | tee "$RESULTS_DIR/exp0_control.log"

echo ""
echo "============================================================"
echo "Experiment 1: OEIS(Adam,30s) → FineWeb(Muon)"
echo "============================================================"
OEIS_PHASE1_SECONDS=30 \
OEIS_OPTIMIZER=adam \
PHASE2_OPTIMIZER=muon \
RUN_ID=oeis_exp1_adam30_muon \
  $LAUNCH "$TRAIN_SCRIPT" 2>&1 | tee "$RESULTS_DIR/exp1_adam30_muon.log"

echo ""
echo "============================================================"
echo "Experiment 2: OEIS(Muon,30s) → FineWeb(Muon)"
echo "============================================================"
OEIS_PHASE1_SECONDS=30 \
OEIS_OPTIMIZER=muon \
PHASE2_OPTIMIZER=muon \
RUN_ID=oeis_exp2_muon30_muon \
  $LAUNCH "$TRAIN_SCRIPT" 2>&1 | tee "$RESULTS_DIR/exp2_muon30_muon.log"

echo ""
echo "============================================================"
echo "Experiment 3: OEIS(Adam,30s) → FineWeb(Adam,60s) → FineWeb(Muon)"
echo "============================================================"
OEIS_PHASE1_SECONDS=30 \
OEIS_OPTIMIZER=adam \
PHASE2_OPTIMIZER=adam_to_muon \
ADAM_TO_MUON_SWITCH_SECONDS=60 \
RUN_ID=oeis_exp3_adam30_adam60_muon \
  $LAUNCH "$TRAIN_SCRIPT" 2>&1 | tee "$RESULTS_DIR/exp3_adam30_adam60_muon.log"

echo ""
echo "============================================================"
echo "Experiment 4: OEIS(Adam,60s) → FineWeb(Muon) [longer phase 1]"
echo "============================================================"
OEIS_PHASE1_SECONDS=60 \
OEIS_OPTIMIZER=adam \
PHASE2_OPTIMIZER=muon \
RUN_ID=oeis_exp4_adam60_muon \
  $LAUNCH "$TRAIN_SCRIPT" 2>&1 | tee "$RESULTS_DIR/exp4_adam60_muon.log"

echo ""
echo "============================================================"
echo "Experiment 5: OEIS(Adam,15s) → FineWeb(Muon) [shorter phase 1]"
echo "============================================================"
OEIS_PHASE1_SECONDS=15 \
OEIS_OPTIMIZER=adam \
PHASE2_OPTIMIZER=muon \
RUN_ID=oeis_exp5_adam15_muon \
  $LAUNCH "$TRAIN_SCRIPT" 2>&1 | tee "$RESULTS_DIR/exp5_adam15_muon.log"

echo ""
echo "============================================================"
echo "Experiment 6: No OEIS, Adam-only baseline (control for Adam)"
echo "============================================================"
OEIS_PHASE1_SECONDS=0 \
PHASE2_OPTIMIZER=adam \
RUN_ID=oeis_exp6_adam_only \
  $LAUNCH "$TRAIN_SCRIPT" 2>&1 | tee "$RESULTS_DIR/exp6_adam_only.log"

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "============================================================"
echo "Results in: $RESULTS_DIR/"
echo ""
echo "Compare val_bpb across experiments:"
grep -h "final_int8_roundtrip_exact" "$RESULTS_DIR"/*.log | sort
