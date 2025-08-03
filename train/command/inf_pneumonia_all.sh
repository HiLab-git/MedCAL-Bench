BASE_PATH="/media/ubuntu/FM-AL/train/outputs/Pneumonia"
RANDOM_PATH="/media/ubuntu/FM-AL/train/outputs/Pneumonia/random"

foundation_models=(clip_RN50x64 clip_ViT-L_14 clip_ViT-L_14@336px medclip_ResNet medclip_ViT medsam_medsam_vit_b sam_vit_b sam_vit_h sam2_2.1b+ sam2_2b+ dino_ViT-B_16 dino_ViT-S_16 dino2_ViT-B_14 dino2_ViT-G_14)
al_plan=(ALPS_47 ALPS_94 ALPS_141 BAL_47 BAL_94 BAL_141 Coreset_47 Coreset_94 Coreset_141 FPS_47 FPS_94 FPS_141 Probcover_47 Probcover_94 Probcover_141 RepDiv_47 RepDiv_94 RepDiv_141 Typiclust_47 Typiclust_94 Typiclust_141)
random=(Random_seed_1_47 Random_seed_1_94 Random_seed_1_141 Random_seed_2_47 Random_seed_2_94 Random_seed_2_141 Random_seed_3_47 Random_seed_3_94 Random_seed_3_141 Random_seed_4_47 Random_seed_4_94 Random_seed_4_141 Random_seed_5_47 Random_seed_5_94 Random_seed_5_141) 

for fm in "${foundation_models[@]}"; do
    for al in "${al_plan[@]}"; do
        CUDA_VISIBLE_DEVICES=0 python ../inference.py \
            -a "$BASE_PATH/${fm}/${al}" \
            -s "test" \
            -d "Pneumonia" \
            --img_size 224 \
            --num_classes 2 \
            --save_pred
    done
done
