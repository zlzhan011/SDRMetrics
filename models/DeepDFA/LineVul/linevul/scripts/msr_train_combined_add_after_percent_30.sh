if [ $# -lt 2 ]
then
echo fail
exit 1
fi

seed=$1
dataset=$2
shift
shift

python linevul_main.py \
  --model_name=${seed}_linevul.bin \
  --output_dir=./saved_models_combined_add_after_percent_30 \
  --model_type=roberta \
  --tokenizer_name=microsoft/codebert-base \
  --model_name_or_path=microsoft/codebert-base \
  --do_train \
  --do_test \
  --train_data_file=../data/$dataset/train_add_after_percent/train_add_after_percent_30.csv  \
  --eval_data_file=../data/$dataset/train_add_after_percent/val_add_after_percent_30.csv  \
  --test_data_file=../data/$dataset/train_add_after_percent/test_add_after_percent_30.csv  \
  --epochs 10 \
  --MSR_processing_flag=after_percent_30  \
  --block_size 512 \
  --train_batch_size 16 \
  --eval_batch_size 16 \
  --learning_rate 2e-5 \
  --max_grad_norm 1.0 \
  --evaluate_during_training \
  --seed $seed $@ 2>&1 | tee "train_${dataset}_${seed}.log"
