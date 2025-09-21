#!/bin/bash

# CSDG vs KgCoOp Experiment Runner for Colab
# Usage: ./run_csdg_experiments.sh [--dry-run|--run] [--bonus]

set -e
RESULTS_DIR="/content/drive/MyDrive/CSDG"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Parse args
DRY_RUN=false
RUN_EXPERIMENTS=false
INCLUDE_BONUS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --run) RUN_EXPERIMENTS=true; shift ;;
        --bonus) INCLUDE_BONUS=true; shift ;;
        *) echo "Usage: $0 [--dry-run|--run] [--bonus]"; exit 1 ;;
    esac
done

if [[ "$DRY_RUN" == false && "$RUN_EXPERIMENTS" == false ]]; then
    echo "Error: Use --dry-run to preview or --run to execute"
    exit 1
fi

# Setup
mkdir -p "$RESULTS_DIR"
cd KgCoOp

log() { echo "[$(date '+%H:%M:%S')] $1"; }

run_exp() {
    local name="$1"
    local cmd="$2"
    echo -e "\n=== $name ==="
    echo "Command: $cmd"

    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY RUN] Would execute: $cmd"
        return
    fi

    log "Starting: $name"
    if eval "$cmd"; then
        log "✅ Completed: $name"
        # Backup results
        local output_dir=$(echo "$cmd" | grep -o 'OUTPUT_DIR [^ ]*' | cut -d' ' -f2)
        if [[ -n "$output_dir" && -d "$output_dir" ]]; then
            cp -r "$output_dir" "$RESULTS_DIR/"
            log "📁 Backed up to: $RESULTS_DIR/$(basename "$output_dir")"
        fi
    else
        log "❌ Failed: $name"
    fi
}

# Core Experiments (1 week total)
echo "🚀 Core Experiments"

# 1. CSDG baseline (use existing config)
run_exp "CSDG Clipart Validation" \
    "python train.py --config-file configs/trainers/CSDG/office_home_clipart_standard.yaml OPTIM.MAX_EPOCH 10 OUTPUT_DIR outputs/csdg_clipart_10ep"

# 2. KgCoOp baseline (modify existing config)
run_exp "KgCoOp Direct Comparison" \
    "python train.py --config-file configs/trainers/KgCoOp/vit_b16_ep100_ctxv1.yaml TRAINER.NAME KgCoOp OPTIM.MAX_EPOCH 10 DATASET.NAME OfficeHomeDG DATASET.ROOT data DATASET.SOURCE_DOMAINS '[\"art\",\"product\",\"real_world\"]' DATASET.TARGET_DOMAINS '[\"clipart\"]' OUTPUT_DIR outputs/kgcoop_clipart_10ep"

# 3. Ablations (modify CSDG config)
run_exp "CSDG No Gating (Content Only)" \
    "python train.py --config-file configs/trainers/CSDG/office_home_clipart_standard.yaml OPTIM.MAX_EPOCH 10 TRAINER.CSDG.GATE.INIT_BIAS 10.0 OUTPUT_DIR outputs/csdg_no_gate_10ep"

run_exp "CSDG No Decorrelation" \
    "python train.py --config-file configs/trainers/CSDG/office_home_clipart_standard.yaml OPTIM.MAX_EPOCH 10 TRAINER.CSDG.LOSS.STYLE_DECORR_WEIGHT 0.0 OUTPUT_DIR outputs/csdg_no_decorr_10ep"

run_exp "CSDG Style Only" \
    "python train.py --config-file configs/trainers/CSDG/office_home_clipart_standard.yaml OPTIM.MAX_EPOCH 10 TRAINER.CSDG.GATE.INIT_BIAS -10.0 OUTPUT_DIR outputs/csdg_style_only_10ep"

# Bonus experiments
if [[ "$INCLUDE_BONUS" == true ]]; then
    echo -e "\n🎯 Bonus Experiments"

    run_exp "CSDG Full Art Domain" \
        "python train.py --config-file configs/trainers/CSDG/office_home_clipart_standard.yaml DATASET.TARGET_DOMAINS '[\"art\"]' OUTPUT_DIR outputs/csdg_art_50ep"

    run_exp "KgCoOp Full Art Domain" \
        "python train.py --config-file configs/trainers/KgCoOp/vit_b16_ep100_ctxv1.yaml TRAINER.NAME KgCoOp DATASET.NAME OfficeHomeDG DATASET.ROOT data DATASET.SOURCE_DOMAINS '[\"art\",\"product\",\"real_world\"]' DATASET.TARGET_DOMAINS '[\"art\"]' OUTPUT_DIR outputs/kgcoop_art_50ep"
fi

if [[ "$DRY_RUN" == true ]]; then
    echo -e "\n✅ Dry run complete! Use --run to execute"
else
    echo -e "\n🎉 All experiments complete! Results in $RESULTS_DIR"
fi