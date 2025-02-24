# distributed_ppdocbee_infer_2B.sh

# 使用 Paddle 分布式启动推理
python -m paddle.distributed.launch --gpus="0,1,2,3" \
    paddlemix/examples/ppdocbee/distributed_ppdocbee_infer.py \
    --model_path PaddleMIX/PPDocBee-7B-1210 \
    --mp_degree 4
