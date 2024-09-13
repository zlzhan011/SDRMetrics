
input_dir=/data/cs_lzhan011/data/cs_lzhan011/vulnerability/data-package/cross_sections/code/data/subsets/multi_category_cross_project
holdout_set=fold_0_dataset
python linevul_main_without_LineVul_train.py --model_name=1_linevul.bin \
--output_dir=./saved_models_multi_category_without_LineVul_with_downsample/${holdout_set} \
--model_type=roberta \
--tokenizer_name=microsoft/codebert-base \
--model_name_or_path=microsoft/codebert-base \
--do_train \
--do_test \
--train_data_file=${input_dir}/${holdout_set}/train.csv \
--eval_data_file=${input_dir}/${holdout_set}/test.csv \
--test_data_file=${input_dir}/${holdout_set}/val.csv \
--epochs 30 \
--block_size 512 \
--train_batch_size 16 \
--eval_batch_size 16 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 1 2>&1 | tee "logs/log_saved_models_multi_category_without_LineVul_with_downsample_$(echo $subset | sed s@/@-@g)_V2_P50.log"



#holdout_set=fold_4_holdout
#python  linevul_main_without_LineVul_train.py \
#  --input_dir=/data/cs_lzhan011/data/cs_lzhan011/vulnerability/data-package/cross_sections/code/data/subsets/multi_category_cross_project/$holdout_set  \
#  --output_dir=./saved_models/multi_category_cross_project_resample_V2_P50/$subset  \
#  --model_type=roberta \
#  --tokenizer_name=microsoft/codebert-base \
#  --model_name_or_path=microsoft/codebert-base \
#  --do_test \
#  --epochs 10 \
#  --block_size 512 \
#  --train_batch_size 8 \
#  --eval_batch_size 8 \
#  --learning_rate 2e-5 \
#  --max_grad_norm 1.0 \
#  --evaluate_during_training \
#  --seed 123456 2>&1 | tee "test_cross_project_resample_$(echo $subset | sed s@/@-@g)_V2_P50.log"