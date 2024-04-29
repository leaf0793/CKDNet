#!/bin/bash
python train_kd.py --slim_pecentage 0.5 > zoo_zoo/kd_nogt_kl/slim_0.5/train.txt
python train_kd.py --slim_pecentage 0.8 > zoo_zoo/kd_nogt_kl/slim_0.8/train.txt



# python main.py --maxdisp 192 \
#                --model stackhourglass \
#                --datapath dataset/ \
#                --epochs 0 \
#                --loadmodel ./trained/checkpoint_10.tar \
#                --savemodel ./trained/



# python finetune.py --maxdisp 192 \
#                    --model stackhourglass \
#                    --datatype 2015 \
#                    --datapath dataset/data_scene_flow_2015/training/ \
#                    --epochs 300 \
#                    --loadmodel ./trained/checkpoint_10.tar \
#                    --savemodel ./trained/

