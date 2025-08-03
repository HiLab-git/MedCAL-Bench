foundation_models=('sam_vit_h' 'medclip_ViT' 'dino_ViT-B_16' 'dino_ViT-S_16' 'dino2_ViT-B_14' 'dino2_ViT-G_14' 'medclip_ResNet')
al_models=('FPS' 'Typiclust' 'Probcover' 'Coreset' 'ALPS' 'RepDiv' 'BAL')
budgets=(51 101 254)

for fm in "${foundation_models[@]}"; do
  for al in "${al_models[@]}"; do
    for budget in "${budgets[@]}"; do
      inference_path="../outputs/Spleen_Date512new/${fm}/${al}_${budget}"
      echo "Running with Foundation Model: ${fm}, AL Model: ${al}, Budget: ${budget}"
      CUDA_VISIBLE_DEVICES=1 python ../inference.py \
        -a "${inference_path}" \
        -s "test" \
        -d "Spleen_Date512new" \
        --img_size 512 \
        --save_pred
    done
  done
done
