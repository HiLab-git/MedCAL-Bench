foundation_models=('clip_RN50x64' 'clip_ViT-L_14' 'clip_ViT-L_14@336px' 'medsam_medsam_vit_b' 'sam_vit_b' 'sam_vit_h' 'sam2_2.1b+' 'sam2_2b+' 'medclip_ViT' 'dino_ViT-B_16' 'dino_ViT-S_16' 'dino2_ViT-B_14' 'dino2_ViT-G_14' 'medclip_ResNet')
al_models=('FPS' 'Typiclust' 'Probcover' 'Coreset' 'ALPS' 'RepDiv' 'BAL')
budgets=(129 194 258)

for fm in "${foundation_models[@]}"; do
  for al in "${al_models[@]}"; do
    for budget in "${budgets[@]}"; do
      inference_path="../outputs/TN3K_preprocessed/${fm}/${al}_${budget}"
      echo "Running with Foundation Model: ${fm}, AL Model: ${al}, Budget: ${budget}"
      CUDA_VISIBLE_DEVICES=0 python ../inference.py \
        -a "${inference_path}" \
        -s "test" \
        -d "TN3K_preprocessed" \
        --img_size 256 \
        --save_nii
    done
  done
done