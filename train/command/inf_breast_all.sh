BASE_PATH="/media/ubuntu/FM-AL/train/outputs/Breast"
RANDOM_PATH="/media/ubuntu/FM-AL/train/outputs/Breast/random"

foundation_models=(clip_RN50x64 clip_ViT-L_14 clip_ViT-L_14@336px medclip_ResNet medclip_ViT medsam_medsam_vit_b sam_vit_b sam_vit_h sam2_2.1b+ sam2_2b+ dino_ViT-B_16 dino_ViT-S_16 dino2_ViT-B_14 dino2_ViT-G_14)
al_plan=(ALPS_27 ALPS_54 ALPS_81 BAL_27 BAL_54 BAL_81 Coreset_27 Coreset_54 Coreset_81 FPS_27 FPS_54 FPS_81 Probcover_27 Probcover_54 Probcover_81 RepDiv_27 RepDiv_54 RepDiv_81 Typiclust_27 Typiclust_54 Typiclust_81)
random=(Random_seed_1_27 Random_seed_1_54 Random_seed_1_81 Random_seed_2_27 Random_seed_2_54 Random_seed_2_81 Random_seed_3_27 Random_seed_3_54 Random_seed_3_81 Random_seed_4_27 Random_seed_4_54 Random_seed_4_81 Random_seed_5_27 Random_seed_5_54 Random_seed_5_81) 

for fm in "${foundation_models[@]}"; do
    for al in "${al_plan[@]}"; do
        CUDA_VISIBLE_DEVICES=0 python ../inference.py \
            -a "$BASE_PATH/${fm}/${al}" \
            -s "test" \
            -d "Breast" \
            --img_size 224 \
            --num_classes 2 \
            --save_pred
    done
done
