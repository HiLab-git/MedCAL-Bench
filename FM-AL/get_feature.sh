export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com

dataset="Heart_preprocessed"
model="dino"
version="ViT-B/16"

echo "extracting features for dataset: ${dataset}, model: ${model}, version: ${version}"
python foundation_feature.py --data_dir "../data/dataset/${dataset}" --model_version_type "${model}_version_${version}"
