export CUDA_VISIBLE_DEVICES=0
foundation_models=('resnet_resnet18' 'clip_RN50x64' 'clip_ViT-L_14' 'clip_ViT-L_14@336px' 'medclip_ViT' 'medclip_ResNet' 'sam_vit_b' 'sam_vit_h' 'medsam_medsam_vit_b' 'sam2_2b+' 'sam2_2.1b+' 'dino_ViT-S_16' 'dino_ViT-B_16' 'dino2_ViT-B_14' 'dino2_ViT-G_14')
al_models=('FPS' 'Typiclust' 'Probcover' 'Coreset' 'ALPS' 'RepDiv' 'BAL')
budgets=(90 135 180)

for fm in "${foundation_models[@]}"; do
  for al in "${al_models[@]}"; do
    for budget in "${budgets[@]}"; do
      plan_path="../data/AL_Plan/Kvasir_preprocessed/${fm}/${al}_${budget}.csv"
      echo "Running with Foundation Model: ${fm}, AL Model: ${al}, Budget: ${budget}"
      python train.py \
        --plan_path "${plan_path}" \
        -c 2 \
        -bs "8" \
        --dim "2" \
        --dataset "Kvasir_preprocessed" \
        --img_size "256" \
        --input_nc "3"
    done
  done
done
