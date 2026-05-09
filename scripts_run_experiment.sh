#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME=${1:-exp_$(date +%Y%m%d_%H%M%S)}
LAYERS=${LAYERS:-"64,32"}
BATCH_SIZE=${BATCH_SIZE:-32}
EPOCHS=${EPOCHS:-3}
LEARNING_RATE=${LEARNING_RATE:-0.03}
LOG_FILE=${LOG_FILE:-results.csv}

output=$(cargo run --release -- train-test \
  --layers "$LAYERS" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --learning-rate "$LEARNING_RATE" \
  --model-name "$MODEL_NAME")

echo "$output"
line=$(echo "$output" | tail -n 1)
accuracy=$(echo "$line" | sed -n 's/.*accuracy=\([0-9.]*%\).*/\1/p')
cost=$(echo "$line" | sed -n 's/.*cost=\([0-9.]*\).*/\1/p')

if [[ ! -f "$LOG_FILE" ]]; then
  echo "timestamp,model_name,layers,batch_size,epochs,learning_rate,cost,accuracy" > "$LOG_FILE"
fi

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),$MODEL_NAME,$LAYERS,$BATCH_SIZE,$EPOCHS,$LEARNING_RATE,$cost,$accuracy" >> "$LOG_FILE"
echo "saved: $LOG_FILE"
