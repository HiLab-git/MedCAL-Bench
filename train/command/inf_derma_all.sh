BASE_PATH="/media/ubuntu/FM-AL/train/outputs/Derma"
RANDOM_PATH="/media/ubuntu/FM-AL/train/outputs/Derma/random"

foundation_models=(clip_RN50x64 clip_ViT-L_14 clip_ViT-L_14@336px medclip_ResNet medclip_ViT medsam_medsam_vit_b sam_vit_b sam_vit_h sam2_2.1b+ sam2_2b+ dino_ViT-B_16 dino_ViT-S_16 dino2_ViT-B_14 dino2_ViT-G_14)
al_plan=(ALPS_70 ALPS_140 ALPS_210 BAL_70 BAL_140 BAL_210 Coreset_70 Coreset_140 Coreset_210 FPS_70 FPS_140 FPS_210 Probcover_70 Probcover_140 Probcover_210 RepDiv_70 RepDiv_140 RepDiv_210 Typiclust_70 Typiclust_140 Typiclust_210)
random=(Random_seed_1_70 Random_seed_1_140 Random_seed_1_210 Random_seed_2_70 Random_seed_2_140 Random_seed_2_210 Random_seed_3_70 Random_seed_3_140 Random_seed_3_210 Random_seed_4_70 Random_seed_4_140 Random_seed_4_210 Random_seed_5_70 Random_seed_5_140 Random_seed_5_210) 

for fm in "${foundation_models[@]}"; do
    for al in "${al_plan[@]}"; do
        CUDA_VISIBLE_DEVICES=0 python ../inference.py \
            -a "$BASE_PATH/${fm}/${al}" \
            -s "test" \
            -d "Derma" \
            --img_size 224 \
            --num_classes 7 \
            --save_pred
    done
done
