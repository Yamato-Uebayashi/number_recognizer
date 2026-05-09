#!/usr/bin/env bash
set -euo pipefail

# 全自動ハイパーパラメータ探索
# 例:
# BATCH_SIZES="32 64" LEARNING_RATES="0.03 0.01" LAYERS_LIST="64,32;128,64,32" EPOCHS=5 ./scripts_run_experiment.sh

BATCH_SIZES=${BATCH_SIZES:-"32 64"}
LEARNING_RATES=${LEARNING_RATES:-"0.03 0.01"}
LAYERS_LIST=${LAYERS_LIST:-"64,32;128,64"}
EPOCHS=${EPOCHS:-5}
SUMMARY_CSV=${SUMMARY_CSV:-results_summary.csv}
EPOCH_CSV=${EPOCH_CSV:-results_epoch.csv}
RUN_PREFIX=${RUN_PREFIX:-exp}

if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "timestamp,model_name,layers,batch_size,epochs,learning_rate,cost,accuracy" > "$SUMMARY_CSV"
fi
if [[ ! -f "$EPOCH_CSV" ]]; then
  echo "model_name,epoch,cost,accuracy" > "$EPOCH_CSV"
fi

IFS=';' read -r -a layers_array <<< "$LAYERS_LIST"

for layers in "${layers_array[@]}"; do
  for bs in $BATCH_SIZES; do
    for lr in $LEARNING_RATES; do
      model_name="${RUN_PREFIX}_L${layers//,/x}_B${bs}_LR${lr}_$(date +%Y%m%d_%H%M%S)"
      tmp_epoch_csv=$(mktemp)
      output=$(cargo run --release -- train-test \
        --layers "$layers" \
        --batch-size "$bs" \
        --epochs "$EPOCHS" \
        --learning-rate "$lr" \
        --model-name "$model_name" \
        --epoch-log "$tmp_epoch_csv")

      echo "$output"
      line=$(echo "$output" | tail -n 1)
      accuracy=$(echo "$line" | sed -n 's/.*accuracy=\([0-9.]*\)%.*/\1/p')
      cost=$(echo "$line" | sed -n 's/.*cost=\([0-9.]*\).*/\1/p')
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),$model_name,$layers,$bs,$EPOCHS,$lr,$cost,$accuracy" >> "$SUMMARY_CSV"

      tail -n +2 "$tmp_epoch_csv" >> "$EPOCH_CSV"
      rm -f "$tmp_epoch_csv"
    done
  done
done

echo "saved: $SUMMARY_CSV"
echo "saved: $EPOCH_CSV"
