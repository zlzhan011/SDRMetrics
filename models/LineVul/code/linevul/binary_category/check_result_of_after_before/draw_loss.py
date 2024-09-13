import pandas as pd
import wandb
import numpy as np
import os
import argparse









if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## parameters
    parser.add_argument("--input_dir", type=str, default='/data/cs_lzhan011/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/linevul/binary_category/check_result_of_after_before/MSR_add_after_to_before/visualize_loss_v2')
    parser.add_argument("--train_data_file", default="train.csv", type=str, required=False,
                        help="The input training data file (a csv file).")
    parser.add_argument("--output_dir", default="/data/cs_lzhan011/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/linevul/binary_category/check_result_of_after_before", type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_type", default="roberta", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--block_size", default=512, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--eval_data_file", default="valid.csv", type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default="test.csv", type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--model_name", default="model.bin", type=str,
                        help="Saved model name.")
    parser.add_argument("--model_name_or_path", default='microsoft/codebert-base', type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--use_non_pretrained_model", action='store_true', default=False,
                        help="Whether to use non-pretrained model.")
    parser.add_argument("--tokenizer_name", default="microsoft/codebert-base", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.")

    parser.add_argument("--do_train", default=True, action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", default=False, action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", default=True,action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--eval_export", action='store_true',
                        help="Whether to save prediction output.")
    parser.add_argument("--sample", action='store_true')
    parser.add_argument("--no_cuda", action='store_true')

    parser.add_argument("--evaluate_during_training", default=True, action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_local_explanation", default=False, action='store_true',
                        help="Whether to do local explanation. ")
    parser.add_argument("--reasoning_method", default=None, type=str,
                        help="Should be one of 'attention', 'shap', 'lime', 'lig'")

    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    args = parser.parse_args()
    args.down_resample = False
    args.visualize_loss = True
    if args.visualize_loss:
        args.output_dir = os.path.join(args.output_dir, 'visualize_loss_v2')
        wandb.login(key="47f89ceb3eeec066f111150006e02c5ebcfb52f3")
        wandb.init(project='Vulnerable', name='add_after_to_before_batch_9th_100000000_smooth_epoch_testing')
        wandb.config = {
            "learning_rate": args.learning_rate,
            "epochs": 10,
            "batch_size": args.train_batch_size
        }

    args.input_dir = os.path.join(args.input_dir, 'batch_loss')
    # args.input_file = os.path.join(args.input_dir, 'train_func_before_loss.csv')
    args.input_file = os.path.join(args.input_dir, 'batch_loss_10.csv')
    batch_loss_9_df = pd.read_csv(args.input_file)
    print("len(batch_loss_9_df): ", len(batch_loss_9_df))

    all_batch_loss = []
    i = 0
    for index, row in batch_loss_9_df.iterrows():
        i = i + 1
        # epoch = row['epoch']
        epoch = 9
        if 'batch_loss_9.csv' in args.input_file:
            epoch_list = [9, '9']
        elif 'batch_loss_10.csv' in args.input_file:
            epoch_list = [0, '0']
        else:
            epoch_list = [9, '9']

        if epoch in [9, '9']:
            batch_loss = row['batch_loss']
            all_batch_loss.append(batch_loss)
            batch_smooth_number = 1000
            if len(all_batch_loss) <= batch_smooth_number:
                avg_loss_100_batch = sum(all_batch_loss)/len(all_batch_loss)
            else:
                avg_loss_100_batch = sum(all_batch_loss[-batch_smooth_number:])/ batch_smooth_number

            # if i >=9000:
            #     avg_loss_100_batch = batch_loss

            wandb.log({"epoch": epoch, "loss training": avg_loss_100_batch})
