
#data_path=/data/cs_lzhan011/vulnerability/DeepDFA_V3/DeepDFA/LineVul/data/MSR

python linevul_main_without_LineVul_train.py --model_name=linevul_add_after_to_before.bin \
--output_dir=./saved_models_binary_category_without_LineVul_not_use_joern_error_embedding_32 \  #saved_models_binary_category_without_LineVul_add_after_to_before_embedding_256
--model_type=roberta \
--tokenizer_name=microsoft/codebert-base \
--model_name_or_path=microsoft/codebert-base \
--do_test \
--train_data_file=../../data/MSR/train_add_after.csv \
--eval_data_file=../../data/MSR/val_add_after.csv \
--test_data_file=../../data/MSR/test.csv \
--test_add_after_file=../../data/MSR/test_add_after.csv \
--epochs 100 \
--block_size 512 \
--train_batch_size 8 \
--eval_batch_size 8 \
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
#--do_train \
#--do_test  \
#--train_data_file=../../data/MSR/train.csv \
#--eval_data_file=../../data/MSR/test.csv  \
#--test_data_file=../../data/MSR/val.csv  \
#--epochs 25  \
#--block_size 512  \
#--train_batch_size 16 \
#--eval_batch_size 16  \
#--learning_rate 2e-5  \
#--max_grad_norm 1.0  \
#--evaluate_during_training \
#--seed 1