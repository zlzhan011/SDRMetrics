#if [ $# -lt 2 ]
#then
#echo fail
#exit 1
#fi

seed=1
dataset=MSR
#shift
#shift
input_dir=/data/cs_lzhan011/data/cs_lzhan011/vulnerability/data-package/cross_sections/code/data/subsets/multi_category_cross_project

kfold_section=fold_0_dataset
python linevul_main_multi_category_train.py \
  --model_name=${seed}_linevul.bin \
  --output_dir=./saved_models_multi_category/${kfold_section} \
  --model_type=roberta \
  --tokenizer_name=microsoft/codebert-base \
  --model_name_or_path=microsoft/codebert-base \
  --do_test \
  --train_data_file=${input_dir}/${kfold_section}/train_full_columns.csv \
  --eval_data_file=${input_dir}/${kfold_section}/valid_full_columns.csv \
  --test_data_file=${input_dir}/${kfold_section}/test_full_columns.csv \
  --epochs 10 \
  --block_size 512 \
  --train_batch_size 16 \
  --eval_batch_size 16 \
  --learning_rate 2e-5 \
  --max_grad_norm 1.0 \
  --evaluate_during_training \
  --seed $seed $@ 2>&1 | tee "train_multi_category_${dataset}_${kfold_section}.log"




#kfold_section=fold_0_dataset
#kfold_holdout=fold_0_holdout
#python linevul_main_multi_category_train.py \
#  --model_name=${seed}_linevul.bin \
#  --output_dir=./saved_models_multi_category/${kfold_section} \
#  --model_type=roberta \
#  --tokenizer_name=microsoft/codebert-base \
#  --model_name_or_path=microsoft/codebert-base \
#  --do_test \
#  --train_data_file=${input_dir}/${kfold_section}/train.csv \
#  --eval_data_file=${input_dir}/${kfold_section}/valid.csv \
#  --test_data_file=${input_dir}/${kfold_holdout}/holdout.csv \
#  --epochs 10 \
#  --block_size 512 \
#  --train_batch_size 16 \
#  --eval_batch_size 16 \
#  --learning_rate 2e-5 \
#  --max_grad_norm 1.0 \
#  --evaluate_during_training \
#  --seed $seed $@ 2>&1 | tee "train_multi_category_${dataset}_${kfold_section}.log"



