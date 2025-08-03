export CUDA_VISIBLE_DEVICES=0

plan_path="/media/ubuntu/FM-AL/data/dataset/Heart_preprocessed/splits/train.csv"
echo "Running with Foundation Model: ${fm}, AL Model: ${al}, Budget: ${budget}"
  python train.py \
  --plan_path "${plan_path}" \
  -c 2 \
  -bs "8" \
  --dim "2" \
  --dataset "Heart_preprocessed" \
  --img_size "224"