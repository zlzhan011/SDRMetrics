
data_path=/data/cs_lzhan011/vulnerability/DeepDFA_V3/DeepDFA/LineVul/data/MSR


# Setting 1
# storage_path = Path('/data2/cs_lzhan011/vulnerability/DeepDFA_V2/DDFA/storage_MSR')
# Split_file = the original split file
# output_dir = 'saved_models_binary_category_without_LineVul_embedding_32_original'
# is_use_joern_error_instance_V2    Yes
# Embedding size: 32
# F1 = 0.6567




#python linevul_main_without_LineVul_train.py --model_name=1_linevul.bin \
#--output_dir=./saved_models_binary_category_without_LineVul_embedding_32_original_write_loss \
#--model_type=roberta \
#--tokenizer_name=microsoft/codebert-base \
#--model_name_or_path=microsoft/codebert-base \
#--do_train \
#--do_test \
#--train_data_file=../../data/MSR/train.csv \
#--eval_data_file=../../data/MSR/train.csv \
#--test_data_file=../../data/MSR/test.csv \
#--epochs 100 \
#--block_size 512 \
#--train_batch_size 16 \
#--eval_batch_size 16 \
#--learning_rate 2e-5 \
#--max_grad_norm 1.0 \
#--seed 1




# Setting 2:
# storage_path = Path('/data2/cs_lzhan011/vulnerability/DeepDFA_V2/DDFA/storage_MSR')
# Split_file = the original split file
# output_dir = 'saved_models_binary_category_without_LineVul_not_use_joern_error_v2_embedding_32'
# is_use_joern_error_instance_V2    No
# Embedding size: 32
# F1 = 0.0



#python linevul_main_without_LineVul_train.py --model_name=1_linevul.bin \
#--output_dir=./saved_models_binary_category_without_LineVul_not_use_joern_error_v2_embedding_32 \
#--model_type=roberta \
#--tokenizer_name=microsoft/codebert-base \
#--model_name_or_path=microsoft/codebert-base \
#--do_train \
#--do_test \
#--train_data_file=../../data/MSR/train.csv \
#--eval_data_file=../../data/MSR/val.csv \
#--test_data_file=../../data/MSR/test.csv \
#--epochs 100 \
#--block_size 512 \
#--train_batch_size 16 \
#--eval_batch_size 16 \
#--learning_rate 2e-5 \
#--max_grad_norm 1.0 \
#--evaluate_during_training \
#--seed 1



# setting 3:
# data path: return Path('/data2/cs_lzhan011/vulnerability/DeepDFA_V2/DDFA/storage_add_after_to_before_work_well_20240528_v2')  3GPU
# Split_file = the original split file
# output_dir = 'saved_models_binary_category_without_LineVul_embedding_32_original'
# is_use_joern_error_instance_V2    No
# Embedding size: 32
# data file:
#--train_data_file=../../data/MSR/train.csv \
#--eval_data_file=../../data/MSR/val.csv \
#--test_data_file=../../data/MSR/test.csv \


#python linevul_main_without_LineVul_train.py --model_name=1_linevul.bin \
#--output_dir=./storage_add_after_to_before_work_well_20240528_embedding_32_original_write_loss \
#--model_type=roberta \
#--tokenizer_name=microsoft/codebert-base \
#--model_name_or_path=microsoft/codebert-base \
#--do_test \
#--train_data_file=../../data/MSR/train.csv \
#--eval_data_file=../../data/MSR/val.csv \
#--test_data_file=../../data/MSR/test.csv \
#--epochs 50 \
#--block_size 512 \
#--train_batch_size 16 \
#--eval_batch_size 16 \
#--learning_rate 2e-5 \
#--max_grad_norm 1.0 \
#--seed 1



# setting 4:
# data path: return Path('/data2/cs_lzhan011/vulnerability/DeepDFA_V2/DDFA/storage_add_after_to_before_work_well_20240528')  3GPU
# Split_file = the add after split file
# output_dir = 'storage_add_after_to_before_work_well_20240528_embedding_32_add_after_write_loss'
# is_use_joern_error_instance_V2    No
# Embedding size: 32
# data file:
#--train_data_file=../../data/MSR/train_add_after.csv \
#--eval_data_file=../../data/MSR/val_add_after.csv \
#--test_data_file=../../data/MSR/test_add_after.csv \
# is_shuffle dataset:  Yes

#python linevul_main_without_LineVul_train.py --model_name=1_linevul.bin \
#--output_dir=./storage_add_after_to_before_work_well_20240528_embedding_32_add_after_write_loss \
#--model_type=roberta \
#--tokenizer_name=microsoft/codebert-base \
#--model_name_or_path=microsoft/codebert-base \
#--do_train \
#--do_test \
#--train_data_file=../../data/MSR/train_add_after.csv \
#--eval_data_file=../../data/MSR/val_add_after.csv \
#--test_data_file=../../data/MSR/test_add_after.csv \
#--epochs 100 \
#--block_size 512 \
#--train_batch_size 16 \
#--eval_batch_size 16 \
#--learning_rate 2e-5 \
#--max_grad_norm 1.0 \
#--seed 1



# setting 5:
# data path: return Path('/data2/cs_lzhan011/vulnerability/DeepDFA_V2/DDFA/storage_add_after_to_before_work_well_20240528')  3GPU
# Split_file = the add after split file
# output_dir = 'storage_add_after_to_before_work_well_20240528_embedding_32_add_after_no_shuflle_dataset_write_loss'
# is_use_joern_error_instance_V2    No
# Embedding size: 32
# data file:
#--train_data_file=../../data/MSR/train_add_after.csv \
#--eval_data_file=../../data/MSR/val_add_after.csv \
#--test_data_file=../../data/MSR/test_add_after.csv \
# is_shuffle dataset:  No

python linevul_main_without_LineVul_train.py --model_name=1_linevul.bin \
--output_dir=./storage_add_after_to_before_work_well_20240528_embedding_32_add_after_no_shuflle_dataset_write_loss \
--model_type=roberta \
--tokenizer_name=microsoft/codebert-base \
--model_name_or_path=microsoft/codebert-base \
--do_train \
--do_test \
--train_data_file=../../data/MSR/train_add_after.csv \
--eval_data_file=../../data/MSR/val_add_after.csv \
--test_data_file=../../data/MSR/test_add_after.csv \
--epochs 100 \
--block_size 512 \
--train_batch_size 16 \
--eval_batch_size 16 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--seed 1

#python linevul_main_without_LineVul_train.py --model_name=1_linevul.bin \
#--output_dir=./saved_models_binary_category_without_LineVul_diversevul_embedding_256_storage_Diversevul_add_after_to_before \
#--model_type=roberta \
#--tokenizer_name=microsoft/codebert-base \
#--model_name_or_path=microsoft/codebert-base \
#--do_train \
#--train_data_file=../../data/Diversevul/train.csv \
#--eval_data_file=../../data/Diversevul/train.csv \
#--test_data_file=../../data/Diversevul/train.csv \
#--epochs 100 \
#--block_size 512 \
#--train_batch_size 16 \
#--eval_batch_size 16 \
#--learning_rate 2e-5 \
#--max_grad_norm 1.0 \
#--seed 1





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