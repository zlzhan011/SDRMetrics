



# setting 3_3:   linevul_main_binary_category_produce_similar_func_embedding
# data path: return Path('/data2/cs_lzhan011/vulnerability/DeepDFA_V2/DDFA/storage_add_after_to_before_work_well_20240528_v2')  3GPU
# Split_file = the original split file
# output_dir = 'storage_add_after_to_before_work_well_20240528_embedding_32_original_write_loss'
# is_use_joern_error_instance_V2    No
# Embedding size: 32
# data file:
#--train_data_file=../../data/MSR/train.csv \
#--eval_data_file=../../data/MSR/val.csv \
#--test_data_file=../../data/MSR/test.csv \


python linevul_main_binary_category_produce_similar_func_embedding.py --model_name=1_linevul.bin \
--output_dir=./storage_add_after_to_before_work_well_20240528_embedding_32_original_write_loss \
--model_type=roberta \
--tokenizer_name=microsoft/codebert-base \
--model_name_or_path=microsoft/codebert-base \
--do_test \
--train_data_file=../../data/MSR/train.csv \
--eval_data_file=../../data/MSR/val.csv \
--test_data_file=../../data/MSR/test.csv \
--epochs 50 \
--block_size 512 \
--train_batch_size 16 \
--eval_batch_size 16 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--seed 1

