DATA_ROOT=../datasets
train_alg=dagger
features=vitbase
ft_dim=768
obj_features=vitbase
obj_ft_dim=768
ngpus=1
seed=0
mode=train # train or test or try

# setting 2
bev_range=5.0

seed=0
bev_grid=11
name=r2r_bev_${bev_range}_${bev_grid}_${mode}
candi2bevdir=path_of_candi2bev.json
outdir=${DATA_ROOT}/R2R/exprs_map/finetune/${name}

flag="--root_dir ${DATA_ROOT}
      --dataset r2r
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer bert

      --bev_weight 0.50
      --bev_range ${bev_range}
      --bev_height 3.0


      --bevfeaturepath path_to_bev_feature
      --bev_grid ${bev_grid}

      --bevglobal
      --candi2bev_dir ${candi2bevdir}

      --enc_full_graph
      --graph_sprels
      --fusion dynamic

      --expert_policy spl
      --train_alg ${train_alg}


      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2
      
      --max_action_len 15
      --max_instr_len 200

      --batch_size 8
      --lr 1e-5
      --iters 200000
      --log_every 1000
      --optim adamW

      --features ${features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --ml_weight 0.2   

      --feat_dropout 0.4
      --dropout 0.5
      
      --gamma 0."

# train
CUDA_VISIBLE_DEVICES='1' python r2r/main_bevnew.py $flag  \
      --tokenizer bert \
      --bert_ckpt_file ../ckpts/model_step_97500.pt \
      --eval_first

# test
# CUDA_VISIBLE_DEVICES='0' python r2r/main_bevnew.py $flag  \
#       --tokenizer bert \
#       --resume_file ../datasets/R2R/trained_models/ \
#       --test --submit