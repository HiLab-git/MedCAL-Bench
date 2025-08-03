export CUDA_VISIBLE_DEVICES=0
BASE_PLAN_PATH="../data/AL_Plan/Derma"
foundation_models=('resnet_resnet18' 'clip_RN50x64' 'clip_ViT-L_14' 'clip_ViT-L_14@336px' 'medclip_ViT' 'medclip_ResNet' 'sam_vit_b' 'sam_vit_h' 'medsam_medsam_vit_b' 'sam2_2b+' 'sam2_2.1b+' 'dino_ViT-S_16' 'dino_ViT-B_16' 'dino2_ViT-B_14' 'dino2_ViT-G_14')

for fm in "${foundation_models[@]}"; do
    for file in "$BASE_PLAN_PATH"/${fm}/*.csv; do
        echo "Running for $file"
        python trainer_cls.py \
            --plan_path "$file" \
            --batch_size 128 \
            --num_epochs 100 \
            --dataset "Derma" \
            --img_size 224 \
            --num_classes 7
    done
done
