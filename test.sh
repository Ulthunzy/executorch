python examples/qualcomm/oss_scripts/llama/llama.py \
  -b build-android \
  -m SM8750 \
  -s 431e3c37 \
  --artifact ./llama-output/qwen3_5-0_8b-m1024-ar16-hybrid-SM8750 \
  --decoder_model qwen3_5-0_8b \
  --model_mode hybrid \
  --prefill_ar_len 16 \
  --max_seq_len 512 \
  --prompt "描述一下这张图片" \
  --image_path "/home/hunter/下载/test.jpg" \
  --compile_only \
  > run.log-0527-qwen3_5-0_8b-m1024-ar16-hybrid-SM8750.txt 2>&1


python examples/qualcomm/oss_scripts/llama/llama.py \
  -b build-android \
  -m SM8750 \
  -s 431e3c37 \
  --artifact ./llama-output/qwen3_5-0_8b-m1024-ar16-hybrid-SM8750 \
  --decoder_model qwen3_5-0_8b \
  --model_mode hybrid \
  --prefill_ar_len 16 \
  --max_seq_len 1024 \
  --prompt "描述一下这张图片" \
  --image_path "/home/hunter/下载/test.jpg" \
  --compile_only \
  > run.log-0527-qwen3_5-0_8b-m1024-ar16-hybrid-SM8750.txt 2>&1

python examples/qualcomm/oss_scripts/llama/llama.py \
  -b build-android \
  -m SM8750 \
  -s 431e3c37 \
  --artifact ./llama-output/qwen3_5-0_8b-m1024-ar32-hybrid-SM8750 \
  --decoder_model qwen3_5-0_8b \
  --model_mode hybrid \
  --prefill_ar_len 32 \
  --max_seq_len 1024 \
  --prompt "描述一下这张图片" \
  --image_path "/home/hunter/下载/test.jpg" \
  --compile_only \
  > run.log-0527-qwen3_5-0_8b-m1024-ar32-hybrid-SM8750.txt 2>&1


python examples/qualcomm/oss_scripts/llama/llama.py \
  -b build-android \
  -m SM8750 \
  -s 431e3c37 \
  --artifact ./llama-output/qwen3_5-0_8b-m1024-ar64-hybrid-SM8750 \
  --decoder_model qwen3_5-0_8b \
  --model_mode hybrid \
  --prefill_ar_len 64 \
  --max_seq_len 1024 \
  --prompt "描述一下这张图片" \
  --image_path "/home/hunter/下载/test.jpg" \
  --compile_only \
  > run.log-0527-qwen3_5-0_8b-m1024-ar64-hybrid-SM8750.txt 2>&1


python examples/qualcomm/oss_scripts/llama/llama.py \
  -b build-android \
  -m SM8750 \
  -s 431e3c37 \
  --artifact ./llama-output/qwen3_5-0_8b-m1024-ar128-hybrid-SM8750 \
  --decoder_model qwen3_5-0_8b \
  --model_mode hybrid \
  --prefill_ar_len 128 \
  --max_seq_len 1024 \
  --prompt "描述一下这张图片" \
  --image_path "/home/hunter/下载/test.jpg" \
  --compile_only \
  > run.log-0527-qwen3_5-0_8b-m1024-ar128-hybrid-SM8750.txt 2>&1


python examples/qualcomm/oss_scripts/llama/llama.py \
  -b build-android \
  -m SM8750 \
  -s 431e3c37 \
  --artifact ./llama-output/qwen3_5-0_8b-m1024-kv-SM8750 \
  --decoder_model qwen3_5-0_8b \
  --model_mode kv \
  --prefill_ar_len 1 \
  --max_seq_len 1024 \
  --prompt "描述一下这张图片" \
  --image_path "/home/hunter/下载/test.jpg" \
  --compile_only \
  > run.log-0527-qwen3_5-0_8b-m1024-kv-SM8750.txt 2>&1


python examples/qualcomm/oss_scripts/llama/llama.py \
  -b build-android \
  -m SM8750 \
  -s 431e3c37 \
  --artifact ./llama-output/qwen3_5-0_8b-m2048-kv-SM8750 \
  --decoder_model qwen3_5-0_8b \
  --model_mode kv \
  --prefill_ar_len 1 \
  --max_seq_len 2048 \
  --prompt "描述一下这张图片" \
  --image_path "/home/hunter/下载/test.jpg" \
  --compile_only \
  > run.log-0527-qwen3_5-0_8b-m2048-kv-SM8750.txt 2>&1