FOUNDATION_MODELS=(
    "resnet_resnet18"
    "clip_RN50x64"
    "medclip_ViT"
    "sam_vit_b"
    "medsam_medsam_vit_b"
    "sam2_2b+"
    "sam2_2.1b+"
    "dino_ViT-B_16"
    "dino2_ViT-B_14"
)
ORGANS=(
    "Heart_preprocessed"
    "Spleen_preprocessed"
    "Kvasir_preprocessed"
    "TN3K_preprocessed"
    "Derma"
    "Breast"
    "Pneumonia"
)
ANNOTATION_BUDGETS=(
    "38 76 114"
    "51 101 254"
    "90 135 180"
    "70 140 210"
    "129 194 258"
    "27 54 81"
    "47 94 141"
)
AL_METHODS=(
    "FPS"
    "Typiclust"
    "Probcover"
    "Coreset"
    "ALPS"
    "RepDiv"
    "BAL"
)

FOUNDATION_MODELS_STR=$(echo "${FOUNDATION_MODELS[*]}")
ORGANS_STR=$(echo "${ORGANS[*]}")
AL_METHODS_STR=$(echo "${AL_METHODS[*]}")

python select_samples.py \
    --foundation_models ${FOUNDATION_MODELS_STR} \
    --organs ${ORGANS_STR} \
    --annotation_budgets ${ANNOTATION_BUDGETS[*]} \
    --AL_methods ${AL_METHODS_STR}
