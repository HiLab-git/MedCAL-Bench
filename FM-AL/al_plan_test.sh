FOUNDATION_MODELS=(
    "sam_vit_b"
)
ORGANS=(
    "Heart_preprocessed"
)
ANNOTATION_BUDGETS=(
    "38 76 114"
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
