foundation_models=('medclip_ResNet')
al_models=('FPS' 'Typiclust' 'Probcover' 'Coreset' 'ALPS' 'RepDiv' 'BAL')
budgets=(90 135 180)

for fm in "${foundation_models[@]}"; do
  for al in "${al_models[@]}"; do
    for budget in "${budgets[@]}"; do
      inference_path="../outputs/Kvasir_preprocessed/${fm}/${al}_${budget}"
      echo "Running with Foundation Model: ${fm}, AL Model: ${al}, Budget: ${budget}"
      CUDA_VISIBLE_DEVICES=2 python ../inference \
        -a "${inference_path}" \
        -s "test" \
        -d "Kvasir_preprocessed" \
        --img_size 256 \
        --input_nc 3 \
        --save_pred
    done
  done
done