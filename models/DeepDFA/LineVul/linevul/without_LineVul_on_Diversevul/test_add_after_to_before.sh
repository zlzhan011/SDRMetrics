

python linevul_main_without_LineVul_train.py --model_name=linevul_add_after_to_before.bin \
--output_dir=./saved_models_binary_category_without_LineVul_add_after_to_before \
--model_type=roberta \
--tokenizer_name=microsoft/codebert-base \
--model_name_or_path=microsoft/codebert-base \
--do_test \
--train_data_file=../../data/MSR/train_add_after.csv \
--eval_data_file=../../data/MSR/test_add_after.csv \
--test_data_file=../../data/MSR/val_add_after.csv \
--epochs 100 \
--block_size 512 \
--train_batch_size 16 \
--eval_batch_size 16 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 1


#python linevul_main_without_LineVul_train.py --model_name=1_linevul.bin \
#--output_dir=./saved_models_multi_category_without_LineVul_with_downsample \
#--model_type=roberta \
#--tokenizer_name=microsoft/codebert-base \
#--model_name_or_path=microsoft/codebert-base \
#--do_train \
#--do_test \
#--train_data_file=../../data/MSR/multi_category/train_cme_category_to_target.csv \
#--eval_data_file=../../data/MSR/multi_category/test_cme_category_to_target.csv \
#--test_data_file=../../data/MSR/multi_category/val_cme_category_to_target.csv \
#--epochs 100 \
#--block_size 512 \
#--train_batch_size 16 \
#--eval_batch_size 16 \
#--learning_rate 2e-5 \
#--max_grad_norm 1.0 \
#--evaluate_during_training \
#--seed 1



#python linevul_main_without_LineVul_train.py --model_name=1_linevul.bin \
#--output_dir=./saved_models_multi_category_without_LineVul_No_downsample \
#--model_type=roberta \
#--tokenizer_name=microsoft/codebert-base \
#--model_name_or_path=microsoft/codebert-base \
#--do_train \
#--do_test \
#--train_data_file=../../data/MSR/multi_category/train_cme_category_to_target.csv \
#--eval_data_file=../../data/MSR/multi_category/test_cme_category_to_target.csv \
#--test_data_file=../../data/MSR/multi_category/val_cme_category_to_target.csv \
#--epochs 10 \
#--block_size 512 \
#--train_batch_size 16 \
#--eval_batch_size 16 \
#--learning_rate 2e-5 \
#--max_grad_norm 1.0 \
#--evaluate_during_training \
#--seed 1







#python linevul_main_without_LineVul_train.py --model_name=1_linevul.bin \
#--output_dir=./saved_models_binary_category_without_LineVul_No_downsample \
#--model_type=roberta \
#--tokenizer_name=microsoft/codebert-base \
#--model_name_or_path=microsoft/codebert-base \
#--do_test  \
#--train_data_file=../../data/MSR/train.csv \
#--eval_data_file=../../data/MSR/val.csv  \
#--test_data_file=../../data/MSR/test.csv  \
#--epochs 25  \
#--block_size 512  \
#--train_batch_size 16 \
#--eval_batch_size 16  \
#--learning_rate 2e-5  \
#--max_grad_norm 1.0  \
#--evaluate_during_training \
#--seed 1