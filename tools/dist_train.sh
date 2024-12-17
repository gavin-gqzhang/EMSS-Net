#!/usr/bin/env bash

target_free_memory=10000
cuda_device=1,2,3
first_cuda=$(echo "$cuda_device" | cut -d ',' -f 1)
IFS=',' read -r -a array <<< "$cuda_device"
NUM_GUP=${#array[@]}

while true; do
    # 仅获取第一个GPU的显存总量和已使用量
    memory_info=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i "$first_cuda")
    
    # 计算空余显存
    total_memory=$(echo $memory_info | cut -d ',' -f 1 | tr -d '[:space:]')
    used_memory=$(echo $memory_info | cut -d ',' -f 2 | tr -d '[:space:]')
    free_memory=$((total_memory - used_memory))

    # 检查空余显存是否达到目标
    if [ "$free_memory" -gt "$target_free_memory" ]; then
        break
    else
        sleep 120
    fi
done

GPUS=3
PORT=1234
# CONFIG="../local_configs/pathformer/B5/pathformer.b5.1024x1024.cancerseg.160k.py"
CONFIG="../local_configs/pathformer/B5/pathformer.b5.1024x1024.cancerseg.160k.py"
WORK_DIR="/data/sdc/checkpoints/medicine_res/mix_scale_segformer_multi_cls_with_attn"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# CUDA_LAUNCH_BLOCKING=1 

CUDA_VISIBLE_DEVICES=$cuda_device python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_addr="127.0.0.1" --master_port=$PORT \
    train.py --config $CONFIG --launcher pytorch --work-dir $WORK_DIR  ${@:1}


#--load-from  'pretrained/pre_iter_58800.pth'
