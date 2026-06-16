echo "Detecting classes ..."
# Run the knowledge distillation script
source activate ~/conda_envs/cdfsod
python distill_knowledge.py \
  --num-gpus 1 \
  --image-path inference_images/trucks_large100_cropped \
  --output-path inference_results \
  --threshold 0.3 \
  --nms-threshold 0.1 \
  --config-file configs/trucks_full_dataset/vitb_shot1_trucks_full_dataset_finetune.yaml \
  MODEL.WEIGHTS output/vitb/trucks_full_dataset_1shot/model_final.pth \
  DE.OFFLINE_RPN_CONFIG configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml

