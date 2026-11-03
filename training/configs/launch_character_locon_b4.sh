#!/usr/bin/env bash
set -euo pipefail

RUN="$HOME/webtoon_lora/char_run2_clean_b4"
OUT="$RUN/output"
LOGDIR="$RUN/logs"
mkdir -p "$OUT" "$LOGDIR"

cd "$HOME/kohya_ss"
PY="$HOME/kohya_ss/.venv/bin/python"

nohup "$PY" sd-scripts/sdxl_train_network.py \
  --pretrained_model_name_or_path "$HOME/webtoon_lora/models/waiIllustriousSDXL_v160.safetensors" \
  --dataset_config "$RUN/character_locon_1024_b4.toml" \
  --output_dir "$OUT" \
  --output_name "char_locon_run2_b4" \
  --save_model_as safetensors \
  --save_precision fp16 \
  --network_module lycoris.kohya \
  --network_dim 32 --network_alpha 16 \
  --network_args "algo=locon" "conv_dim=16" "conv_alpha=8" "use_tucker=True" "train_norm=True" \
  --learning_rate 1e-4 --unet_lr 1e-4 --text_encoder_lr 5e-5 \
  --optimizer_type AdamW8bit \
  --lr_scheduler cosine --lr_warmup_steps 184 \
  --max_train_epochs 3 \
  --train_batch_size 4 --gradient_accumulation_steps 1 \
  --max_grad_norm 1 \
  --min_snr_gamma 5 --scale_weight_norms 1 \
  --mixed_precision bf16 --gradient_checkpointing \
  --cache_latents --cache_latents_to_disk \
  --vae_batch_size 1 \
  --no_half_vae \
  --sdpa \
  --max_data_loader_n_workers 0 \
  --save_every_n_epochs 1 \
  --seed 42 \
  --logging_dir "$LOGDIR" --log_prefix char_locon_run2_b4 \
  > "$RUN/train.log" 2>&1 &

PID=$!
echo "$PID" > "$RUN/train.pid"
echo "LAUNCHED_PID=$PID"
sleep 5
if kill -0 "$PID" 2>/dev/null; then
  echo "ALIVE"
else
  echo "DEAD"
  tail -80 "$RUN/train.log" || true
  exit 1
fi
