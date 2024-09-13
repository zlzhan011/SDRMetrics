# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import os, sys

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
parent_parent_dir = os.path.dirname(parent_dir)
parent_parent_parent_dir = os.path.dirname(parent_parent_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)
sys.path.append(parent_parent_dir)
sys.path.append(parent_parent_parent_dir)
import random
import re
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from tqdm import tqdm
from linevul_model_without_LineVul import Model
import pandas as pd
# metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, \
    confusion_matrix
# word-level tokenizer
from tokenizers import Tokenizer
from DDFA.code_gnn.models.flow_gnn.ggnn import FlowGNNGGNNModule
from DDFA.sastvd.linevd.datamodule import BigVulDatasetLineVDDataModule
import dgl
from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens,
                 input_ids,
                 label,
                 i):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label = label
        self.index = i


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_type="train", return_index=False):
        if file_type == "train":
            file_path = args.train_data_file
        elif file_type == "eval":
            file_path = args.eval_data_file
        elif file_type == "test":
            file_path = args.test_data_file
        elif file_type == "test_add_after":
            file_path = args.test_add_after_file
        self.examples = []
        args.file_type = file_type
        # if file_path.endswith(".jsonl"):
        #     df = pd.read_json(file_path, orient="records", lines=True)
        # elif file_path.endswith(".csv"):
        #
        #     df = pd.read_csv(file_path, index_col=0)
        # else:
        #     raise NotImplementedError(file_path)

        if file_path.endswith(".jsonl"):
            df = pd.read_json(file_path, orient="records", lines=True)
        elif file_path.endswith(".xlsx"):
            c_root = args.input_file_dir
            file_path = os.path.join(c_root, file_path)
            print("file_path:", file_path)
            df = pd.read_excel(file_path)
            # df = df[df['func_before'].notna()]
        elif file_path.endswith(".csv"):

            df = pd.read_csv(file_path, index_col=0)
        else:
            raise NotImplementedError(file_path)

        if "holdout" in os.path.basename(file_path):
            df["split"].replace("holdout", "test")

        if args.sample:
            df = df.sample(100)

        if "processed_func" in df.columns:
            func_key = "processed_func"
        elif "func" in df.columns:
            func_key = "func"

        func_key = 'func_before'  # processed_func   func_after  func_before
        args.input_column = func_key
        if func_key not in df.columns:
            func_key = "processed_func"
        if args.multi_category_flag:
            target_label = 'cme_category_to_target'
        else:
            target_label = 'target'  # target  cme_category_to_target

        if args.train_data_file is not None and "devign" in args.train_data_file:
            def zonk(s):
                lines = s.splitlines()
                lines = [re.sub(r"[\t ]+", " ", l.strip()) for l in lines if len(l.strip()) > 0]
                return "\n".join(lines)

            df[func_key] = df[func_key].apply(zonk)

        for one_column in df.columns.tolist():
            df = df.dropna(subset=[one_column])
        funcs = df[func_key].tolist()
        labels = df[target_label].astype(int).tolist()
        indices = df.index.astype(int).tolist()

        if args.paired_flag:
            from LineVul.data.paired.read_paired_file_id import read_paired_file_id
            file_id_dict = read_paired_file_id()
        cnt = 0
        for i in tqdm(range(len(funcs)), desc="load dataset"):
            if args.paired_flag:
                if indices[i] not in file_id_dict:
                    continue
            cnt += 1
            self.examples.append(convert_examples_to_features(funcs[i], labels[i], tokenizer, args, indices[i]))
        print("cnt:", cnt)

        # if i > 100:
        #     print("Find paired dataset")
        # else:
        #     exit("Please check the paired dataset")

        def category_distribution(y, desc=""):
            from collections import Counter
            # 假设 y_resampled 是一个列表
            value_counts = Counter(y)
            print(desc)
            # 打印每个值及其出现的次数
            label_0_cnt = 0
            for value, count in value_counts.items():
                print(f"Value: {value}, Count: {count}")
                if value != 0:
                    label_0_cnt = label_0_cnt + count

            value_counts[0] = label_0_cnt
            return value_counts

        if args.downsample_flag:

            X = [example.input_ids for example in self.examples]
            y = [example.label for example in self.examples]
            sampling_strategy = category_distribution(y, desc="original distribution")

            # sampling_strategy = {0: 8891, 1: 2306, 2: 1113, 3: 1366, 4: 1291, 5:661, 6:2154}

            from imblearn.under_sampling import RandomUnderSampler
            rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
            X_resampled, y_resampled = rus.fit_resample(X, y)
            category_distribution(y_resampled, desc="resample distribution")
            self.examples_resampled = self.examples[:len(y_resampled)]

            for i in range(len(self.examples_resampled)):
                self.examples_resampled[i].input_ids = X_resampled[i]
                self.examples_resampled[i].label = y_resampled[i]

            self.examples = self.examples_resampled

        if file_type in ("train", "test"):
            for example in self.examples[:3]:
                logger.info("*** Example ***")
                logger.info("label: {}".format(example.label))
                logger.info("input_tokens: {}".format([x.replace('\u0120', '_') for x in example.input_tokens]))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
        self.return_index = args.eval_export or return_index

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        if self.return_index:
            return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label), torch.tensor(
                self.examples[i].index)
        else:
            return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)


def convert_examples_to_features(func, label, tokenizer, args, i):
    if args.use_word_level_tokenizer:
        encoded = tokenizer.encode(func)
        encoded = encoded.ids
        if len(encoded) > 510:
            encoded = encoded[:510]
        encoded.insert(0, 0)
        encoded.append(2)
        if len(encoded) < 512:
            padding = 512 - len(encoded)
            for _ in range(padding):
                encoded.append(1)
        source_ids = encoded
        source_tokens = []
        return InputFeatures(source_tokens, source_ids, label, i)
    # source
    code_tokens = tokenizer.tokenize(str(func))[:args.block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, label, i)


def visual(args, model, tokenizer, test_dataset, flowgnn_dataset, output_dir, best_threshold=0.5):
    # build dataloader
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=0)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    num_missing = 0
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]
    y_trues=[]

    num_classes = 2
    featdims_sum = 772
    feat_log = np.zeros((len(test_dataloader.dataset), featdims_sum))
    score_log = np.zeros((len(test_dataloader.dataset), num_classes))
    label_log = np.zeros(len(test_dataloader.dataset))
    predict_log = np.zeros(len(test_dataloader.dataset))


    all_inputs_ids = []
    if args.profile:
        prof = FlopsProfiler(model)
    if args.time:
        pass
    profs = []
    softmax_entry = []

    cache_name = os.path.join(os.path.dirname(output_dir), "DeepDFA_on_MSR" + "_last_second_layer_target_0_and_1_func_before_test_add_after.npy")

    batch_size = test_dataloader.batch_size
    batch_idx = 0

    start_ind_reset = 0
    end_ind_rest = 0
    for i, batch in enumerate(tqdm(test_dataloader, desc="evaluate test")):
        start_ind = batch_idx * batch_size
        end_ind = min((batch_idx + 1) * batch_size, len(test_dataloader.dataset))
        do_profile = args.profile and i > 2
        do_time = args.time and i > 2
        if do_profile:
            prof.start_profile()
        if do_time:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
        (inputs_ids, labels, index) = [x.to(args.device) for x in batch]
        if flowgnn_dataset is None:
            graphs=None
        else:
            graphs, keep_idx = flowgnn_dataset.get_indices(index)
            num_missing += len(labels) - len(keep_idx)
            inputs_ids = inputs_ids[keep_idx]
            print("keep_idx:", keep_idx)
            labels = labels[keep_idx]
            index_remain = index[keep_idx]
        with torch.no_grad():
            if do_time:
                start.record()
            lm_loss, logit, features_list = model(input_ids=inputs_ids, labels=labels, graphs=graphs, features_list_flag=True)
            if do_time:
                end.record()
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
            all_inputs_ids.append(index_remain)

        softmax_entry.append({"targets": labels.tolist(),
                              "feature_list_0": [item.tolist() for item in features_list[0]],
                              "feature_list_1": [item.tolist() for item in features_list[1]],
                              # "feature_list_2": [item.tolist() for item in features_list[2]],
                              "predict": np.argmax(logit.cpu().numpy(), axis=1).tolist()})
        print("features_list 0 shape: ", features_list[0].shape)
        print("features_list 1 shape: ", features_list[1].shape)
        out = torch.cat([layer_feat for layer_feat in features_list], dim=1)
        print("out shape ----:", out.shape)
        best_threshold = 0.5
        logits_np = np.concatenate(logits, 0)


        score = lm_loss

        if end_ind_rest == 0:
            start_ind_reset = 0
            end_ind_rest = start_ind_reset + out.shape[0]
        else:
            start_ind_reset = end_ind_rest
            end_ind_rest = start_ind_reset + out.shape[0]
        y_preds = logits_np[start_ind_reset:end_ind_rest, 1] > best_threshold
        print("y_preds:", y_preds)
        print("score shape ----:", score.shape)
        print("start_ind_reset:", start_ind_reset)
        print("end_ind_rest:", end_ind_rest)
        print("labels shape ----:", labels.shape)
        print("y_preds shape:", y_preds.shape)
        feat_log[start_ind_reset:end_ind_rest, :] = out.data.cpu().numpy()
        label_log[start_ind_reset:end_ind_rest] = labels.data.cpu().numpy()
        # score_log[start_ind:end_ind] = score.data.cpu().numpy()
        predict_log[start_ind_reset:end_ind_rest] = y_preds



        nb_eval_steps += 1
        if do_profile:
            flops = prof.get_total_flops(as_string=True)
            params = prof.get_total_params(as_string=True)
            macs = prof.get_total_macs(as_string=True)
            prof.print_model_profile(profile_step=i, output_file=f"{output_dir}.profile.txt")
            prof.end_profile()
            logger.info("step %d: %s flops %s params %s macs", i, flops, params, macs)
            profs.append({
                "step": i,
                "flops": flops,
                "params": params,
                "macs": macs,
                "batch_size": len(labels),
            })
        if do_time:
            torch.cuda.synchronize()
            tim = start.elapsed_time(end)
            logger.info("step %d: time %f", i, tim)
            profs.append({
                "step": i,
                "batch_size": len(labels),
                "runtime": tim,
            })
    if args.profile:
        filename = f"{output_dir}.profiledata.txt"
    elif args.time:
        filename = f"{output_dir}.timedata.txt"
    else:
        filename = None
    if filename is not None:
        with open(filename, "w") as f:
            json.dump(profs, f)
    logger.info("%d items missing", num_missing)




    np.savez(cache_name, feat_log.T, label_log, predict_log)


    # calculate scores
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    print("logits:", logits.shape)
    print("y_trues:", y_trues.shape)
    if len(list(set(y_trues.tolist()))) == 2 or len(list(set(y_trues.tolist()))) == 1 :
        best_threshold = 0.5
        best_f1 = 0
        y_preds = logits[:, 1] > best_threshold
        recall = recall_score(y_trues, y_preds)
        precision = precision_score(y_trues, y_preds)
        f1 = f1_score(y_trues, y_preds)
        result = {
            "eval_recall": float(recall),
            "eval_precision": float(precision),
            "eval_f1": float(f1),
            "eval_threshold": best_threshold,
        }


        print("all_inputs_ids:", all_inputs_ids[:10])
        all_inputs_ids = [item.cpu().numpy() for tensor in all_inputs_ids for item in tensor]
        print("all_inputs_ids:", all_inputs_ids[:10])
        y_preds = logits[:, 1] > best_threshold

        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))

        print("args.input_file_dir:", args.input_file_dir)
        # all_inputs_ids = [list(row) for row in all_inputs_ids]
        y_preds_trues = pd.DataFrame(
            {"all_inputs_ids": all_inputs_ids, "y_trues": y_trues, "DeepDFA_Without_LineVul_Predictions": y_preds, "logits":logits[:, 1]})
        if args.add_after_to_before:
            y_preds_trues.to_excel(
                os.path.join(args.input_file_dir, 'DeepDFA/without_LineVul_add_after_to_before', "val_" + args.input_column+"_"+str(args.file_type)+"_prediction_joern_error_add_after_into_before.xlsx"))
        else:
            y_preds_trues.to_excel(
                os.path.join(args.input_file_dir, 'DeepDFA/without_LineVul',
                             "val_" + args.input_column + "_prediction.xlsx"))

    else:


        target_names = [str(i) for i in sorted(list(set(y_trues.tolist())))]
        y_preds = np.argmax(logits, axis=1)
        result = classification_report(y_trues, y_preds, target_names=target_names)
        print("result:")
        print(result)
        eval_f1 = float(result.split("      ")[-2])
        result = {"result": result,
                  "eval_f1": eval_f1}

        best_threshold = 0.5
        y_trues_binary = [1 if item >= 1 else 0 for item in y_trues]
        y_preds_binary = [1 if item >= 1 else 0 for item in y_preds]
        recall = recall_score(y_trues_binary, y_preds_binary)
        precision = precision_score(y_trues_binary, y_preds_binary)
        f1 = f1_score(y_trues_binary, y_preds_binary)
        binary_result = {
            "eval_recall": float(recall),
            "eval_precision": float(precision),
            "eval_f1": float(f1),
            "eval_threshold": best_threshold,
        }
        print("binary_result:", binary_result)


        from LineVul.linevul.multi_category.overall_metrics import overall_metrics
        overall_metrics(y_trues, y_preds)




def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    dgl.seed(args.seed)


def train(args, train_dataset, model, tokenizer, eval_dataset, flowgnn_dataset):
    """ Train the model """
    # build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0)

    args.max_steps = args.epochs * len(train_dataloader)
    # evaluate the model per epoch
    args.save_steps = len(train_dataloader)
    args.warmup_steps = args.max_steps // 5
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size // max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_f1 = 0

    model.zero_grad()

    num_missing = 0

    for idx in range(args.epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            (inputs_ids, labels, index) = [x.to(args.device) for x in batch]
            if flowgnn_dataset is None:
                graphs = None
            else:
                graphs, keep_idx = flowgnn_dataset.get_indices(index)
                num_missing += len(labels) - len(keep_idx)
                inputs_ids = inputs_ids[keep_idx]
                labels = labels[keep_idx]
                # print("labels:", labels)
            model.train()
            loss, logits = model(input_ids=inputs_ids, labels=labels, graphs=graphs)
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss

            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag = True
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)

                if global_step % args.save_steps == 0:
                    results = evaluate(args, model, tokenizer, eval_dataset, flowgnn_dataset, eval_when_training=True)

                    # Save model checkpoint
                    if results['eval_f1'] > best_f1:
                        best_f1 = results['eval_f1']
                        logger.info("  " + "*" * 20)
                        logger.info("  Best f1:%s", round(best_f1, 4))
                        logger.info("  " + "*" * 20)

                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format(args.model_name))
                        torch.save(model_to_save.state_dict(), output_dir)
                        output_dir_f1 = output_dir + str(round(best_f1, 4))
                        torch.save(model_to_save.state_dict(), output_dir_f1)
                        logger.info("\n\n\nSaving model checkpoint to %s", output_dir)
        logger.info("%d items missing", num_missing)
        checkpoint_prefix = 'checkpoint-last'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        output_dir = os.path.join(output_dir, '{}'.format(args.model_name))
        torch.save(model_to_save.state_dict(), output_dir)
        logger.info("Saving model checkpoint to %s", output_dir)

    return model


def evaluate_bak(args, model, tokenizer, eval_dataset, eval_when_training=False):
    # build dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    for batch in tqdm(eval_dataloader, desc="evaluate eval"):
        (inputs_ids, labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            lm_loss, logit = model(input_ids=inputs_ids, labels=labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1

    # calculate scores
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    if len(list(set(y_trues.tolist()))) == 2:
        best_threshold = 0.5
        best_f1 = 0
        y_preds = logits[:, 1] > best_threshold
        recall = recall_score(y_trues, y_preds)
        precision = precision_score(y_trues, y_preds)
        f1 = f1_score(y_trues, y_preds)
        result = {
            "eval_recall": float(recall),
            "eval_precision": float(precision),
            "eval_f1": float(f1),
            "eval_threshold": best_threshold,
        }

        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))
    else:

        from sklearn.metrics import classification_report
        target_names = [str(i) for i in sorted(list(set(y_trues.tolist())))]
        y_preds = np.argmax(logits, axis=1)
        result = classification_report(y_trues, y_preds, target_names=target_names)
        print("result:")
        print(result)
        eval_f1 = float(result.split("      ")[-2])
        result = {"result": result,
                  "eval_f1": eval_f1}
    return result


def evaluate(args, model, tokenizer, eval_dataset, flowgnn_dataset, eval_when_training=False):
    # build dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    num_missing = 0
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    for batch in tqdm(eval_dataloader, desc="evaluate eval"):
        (inputs_ids, labels, index) = [x.to(args.device) for x in batch]
        if flowgnn_dataset is None:
            graphs = None
        else:
            graphs, keep_idx = flowgnn_dataset.get_indices(index)
            num_missing += len(labels) - len(keep_idx)
            inputs_ids = inputs_ids[keep_idx]
            labels = labels[keep_idx]
        with torch.no_grad():
            lm_loss, logit = model(input_ids=inputs_ids, labels=labels, graphs=graphs)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
    logger.info("%d items missing", num_missing)

    # calculate scores
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    if len(list(set(y_trues.tolist()))) == 2:
        best_threshold = 0.5
        best_f1 = 0
        y_preds = logits[:, 1] > best_threshold
        recall = recall_score(y_trues, y_preds)
        precision = precision_score(y_trues, y_preds)
        f1 = f1_score(y_trues, y_preds)
        result = {
            "eval_recall": float(recall),
            "eval_precision": float(precision),
            "eval_f1": float(f1),
            "eval_threshold": best_threshold,
        }

        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))
    else:

        from sklearn.metrics import classification_report
        target_names = [str(i) for i in sorted(list(set(y_trues.tolist())))]
        y_preds = np.argmax(logits, axis=1)
        result = classification_report(y_trues, y_preds, target_names=target_names)
        print("result:")
        print(result)
        eval_f1 = float(result.split("      ")[-2])
        result = {"result": result,
                  "eval_f1": eval_f1}

    return result


def t_test(args, model, tokenizer, test_dataset, flowgnn_dataset, output_dir, best_threshold=0.5):
    # build dataloader
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=0)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    num_missing = 0
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    all_inputs_ids = []
    if args.profile:
        prof = FlopsProfiler(model)
    if args.time:
        pass
    profs = []
    softmax_entry = []
    for i, batch in enumerate(tqdm(test_dataloader, desc="evaluate test")):
        do_profile = args.profile and i > 2
        do_time = args.time and i > 2
        if do_profile:
            prof.start_profile()
        if do_time:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
        (inputs_ids, labels, index) = [x.to(args.device) for x in batch]
        if flowgnn_dataset is None:
            graphs = None
        else:
            graphs, keep_idx = flowgnn_dataset.get_indices(index)
            num_missing += len(labels) - len(keep_idx)
            inputs_ids = inputs_ids[keep_idx]
            labels = labels[keep_idx]
            index_remain = index[keep_idx]
        with torch.no_grad():
            if do_time:
                start.record()
            lm_loss, logit, features_list = model(input_ids=inputs_ids, labels=labels, graphs=graphs,
                                                  features_list_flag=True)
            if do_time:
                end.record()
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
            all_inputs_ids.append(index_remain)

        softmax_entry.append({"targets": labels.tolist(),
                              "feature_list_0": [item.tolist() for item in features_list[0]],
                              "feature_list_1": [item.tolist() for item in features_list[1]],
                              # "feature_list_2": [item.tolist() for item in features_list[2]],
                              "predict": np.argmax(logit.cpu().numpy(), axis=1).tolist()})

        nb_eval_steps += 1
        if do_profile:
            flops = prof.get_total_flops(as_string=True)
            params = prof.get_total_params(as_string=True)
            macs = prof.get_total_macs(as_string=True)
            prof.print_model_profile(profile_step=i, output_file=f"{output_dir}.profile.txt")
            prof.end_profile()
            logger.info("step %d: %s flops %s params %s macs", i, flops, params, macs)
            profs.append({
                "step": i,
                "flops": flops,
                "params": params,
                "macs": macs,
                "batch_size": len(labels),
            })
        if do_time:
            torch.cuda.synchronize()
            tim = start.elapsed_time(end)
            logger.info("step %d: time %f", i, tim)
            profs.append({
                "step": i,
                "batch_size": len(labels),
                "runtime": tim,
            })
    if args.profile:
        filename = f"{output_dir}.profiledata.txt"
    elif args.time:
        filename = f"{output_dir}.timedata.txt"
    else:
        filename = None
    if filename is not None:
        with open(filename, "w") as f:
            json.dump(profs, f)
    logger.info("%d items missing", num_missing)

    # calculate scores
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)

    if len(list(set(y_trues.tolist()))) == 2:
        best_threshold = 0.5
        best_f1 = 0
        y_preds = logits[:, 1] > best_threshold
        recall = recall_score(y_trues, y_preds)
        precision = precision_score(y_trues, y_preds)
        f1 = f1_score(y_trues, y_preds)
        result = {
            "eval_recall": float(recall),
            "eval_precision": float(precision),
            "eval_f1": float(f1),
            "eval_threshold": best_threshold,
        }

        print("all_inputs_ids:", all_inputs_ids[:10])
        all_inputs_ids = [item.cpu().numpy() for tensor in all_inputs_ids for item in tensor]
        print("all_inputs_ids:", all_inputs_ids[:10])
        y_preds = logits[:, 1] > best_threshold

        print("args.input_file_dir:", args.input_file_dir)
        logger.info("***** Eval results *****")

        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))


        y_preds_trues = pd.DataFrame(
            {"all_inputs_ids": all_inputs_ids, "y_trues": y_trues, "DeepDFA_Without_LineVul_Predictions": y_preds,
             "logits": logits[:, 1]})
        if args.add_after_to_before:
            args.evaluate_output = os.path.join(args.input_file_dir, 'DeepDFA/without_LineVul_add_after_to_before')
            if not os.path.exists(args.evaluate_output):
                os.makedirs(args.evaluate_output)
            y_preds_trues.to_excel(
                os.path.join(args.evaluate_output,
                             "val_" + args.input_column + "_" + str(args.file_type) + "_prediction.xlsx"))
        else:
            args.evaluate_output = os.path.join(args.input_file_dir, 'DeepDFA/without_LineVul')
            if not os.path.exists(args.evaluate_output):
                os.makedirs(args.evaluate_output)
            y_preds_trues.to_excel(
                os.path.join(args.input_file_dir, 'DeepDFA/without_LineVul',
                             "val_" + args.input_column + "_" + str(args.file_type) + "_prediction.xlsx"))


    else:

        target_names = [str(i) for i in sorted(list(set(y_trues.tolist())))]
        y_preds = np.argmax(logits, axis=1)
        result = classification_report(y_trues, y_preds, target_names=target_names)
        print("result:")
        print(result)
        eval_f1 = float(result.split("      ")[-2])
        result = {"result": result,
                  "eval_f1": eval_f1}

        best_threshold = 0.5
        y_trues_binary = [1 if item >= 1 else 0 for item in y_trues]
        y_preds_binary = [1 if item >= 1 else 0 for item in y_preds]
        recall = recall_score(y_trues_binary, y_preds_binary)
        precision = precision_score(y_trues_binary, y_preds_binary)
        f1 = f1_score(y_trues_binary, y_preds_binary)
        binary_result = {
            "eval_recall": float(recall),
            "eval_precision": float(precision),
            "eval_f1": float(f1),
            "eval_threshold": best_threshold,
        }
        print("binary_result:", binary_result)

        from LineVul.linevul.multi_category.overall_metrics import overall_metrics
        overall_metrics(y_trues, y_preds)




def main():
    parser = argparse.ArgumentParser()
    ## parameters
    parser.add_argument("--input_file_dir", type=str,
                        default='/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/paired')
    parser.add_argument("--train_data_file", default=None, type=str, required=False,
                        help="The input training data file (a csv file).")
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")  #
    parser.add_argument("--test_add_after_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--model_name", default="model.bin", type=str,
                        help="Saved model name.")
    parser.add_argument("--model_name_or_path", default='microsoft/codebert-base', type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--use_non_pretrained_model", action='store_true', default=False,
                        help="Whether to use non-pretrained model.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--eval_export", action='store_true',
                        help="Whether to save prediction output.")
    parser.add_argument("--sample", action='store_true')
    parser.add_argument("--no_cuda", action='store_true')

    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_local_explanation", default=False, action='store_true',
                        help="Whether to do local explanation. ")
    parser.add_argument("--reasoning_method", default=None, type=str,
                        help="Should be one of 'attention', 'shap', 'lime', 'lig'")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=1,
                        help="training epochs")
    # RQ2
    parser.add_argument("--effort_at_top_k", default=0.2, type=float,
                        help="Effort@TopK%Recall: effort at catching top k percent of vulnerable lines")
    parser.add_argument("--top_k_recall_by_lines", default=0.01, type=float,
                        help="Recall@TopK percent, sorted by line scores")
    parser.add_argument("--top_k_recall_by_pred_prob", default=0.2, type=float,
                        help="Recall@TopK percent, sorted by prediction probabilities")

    parser.add_argument("--do_sorting_by_line_scores", default=False, action='store_true',
                        help="Whether to do sorting by line scores.")
    parser.add_argument("--do_sorting_by_pred_prob", default=False, action='store_true',
                        help="Whether to do sorting by prediction probabilities.")
    # RQ3 - line-level evaluation
    parser.add_argument('--top_k_constant', type=int, default=10,
                        help="Top-K Accuracy constant")
    # num of attention heads
    parser.add_argument('--num_attention_heads', type=int, default=12,
                        help="number of attention heads used in CodeBERT")
    # raw predictions
    parser.add_argument("--write_raw_preds", default=False, action='store_true',
                        help="Whether to write raw predictions on test data.")
    # word-level tokenizer
    parser.add_argument("--use_word_level_tokenizer", default=False, action='store_true',
                        help="Whether to use word-level tokenizer.")
    # bpe non-pretrained tokenizer
    parser.add_argument("--use_non_pretrained_tokenizer", default=False, action='store_true',
                        help="Whether to use non-pretrained bpe tokenizer.")
    parser.add_argument("--no_flowgnn", action="store_true", help="do not train/evaluate DDFA as part of the model")
    parser.add_argument("--really_no_flowgnn", action="store_true", help="do not load any DDFA stuff")
    parser.add_argument("--no_concat", action="store_true", help="do not concatenate DDFA abstract dataflow embedding")
    parser.add_argument("--dsname", type=str, default="bigvul", help="dataset name to load for DDFA")
    parser.add_argument("--profile", action="store_true", help="profile MACs")
    parser.add_argument("--time", action="store_true", help="measure inference time")
    args = parser.parse_args()
    args.paired_flag = False
    args.downsample_flag = False
    args.multi_category_flag = False
    args.add_after_to_before = True
    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu, )

    # Set seed
    set_seed(args)

    # load all graphs
    if args.really_no_flowgnn:
        flowgnn_datamodule = None
        flowgnn_dataset = None
    else:
        feat = "_ABS_DATAFLOW_datatype_all_limitall_10000_limitsubkeys_10000"
        gtype = "cfg"
        label_style = "graph"
        dsname = args.dsname
        concat_all_absdf = not args.no_concat
        flowgnn_datamodule = BigVulDatasetLineVDDataModule(
            feat,
            gtype,
            label_style,
            dsname,
            undersample=None,
            oversample=None,
            sample=-1,
            sample_mode=args.sample,
            train_workers=1,
            val_workers=0,
            test_workers=0,
            split="fixed",
            batch_size=256,
            # nsampling=False,
            # nsampling_hops=1,
            seed=args.seed,
            # test_every=False,
            # dataflow_defined_only=False,
            # codebert_feat=None,
            # doc2vec_feat=None,
            # glove_feat=None,
            concat_all_absdf=concat_all_absdf,
            # use_weighted_loss=False,
            # use_random_weighted_sampler=False,
            train_includes_all=True,
            load_features=not args.no_flowgnn,
        )
        flowgnn_dataset = flowgnn_datamodule.train
        logger.info("FlowGNN dataset:\n%s", flowgnn_datamodule.train.df)

    # breakpoint()

    # load model
    if args.really_no_flowgnn:
        flowgnn_model = None
    else:
        input_dim = flowgnn_datamodule.input_dim
        hidden_dim = 128
        n_steps = 5
        num_output_layers = 3
        flowgnn_model = FlowGNNGGNNModule(
            feat,
            input_dim,
            hidden_dim,
            n_steps,
            num_output_layers,
            label_style=label_style,
            concat_all_absdf=concat_all_absdf,
            # undersample_node_on_loss_factor=None,
            # test_every=False,
            # tune_nni=False,
            # positive_weight=None,
            encoder_mode=True,
        )
        logger.info("FlowGNN output dim: %d", flowgnn_model.out_dim)

    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    config.num_labels = 1
    config.num_attention_heads = args.num_attention_heads
    if args.use_word_level_tokenizer:
        print('using wordlevel tokenizer!')
        tokenizer = Tokenizer.from_file('../word_level_tokenizer/wordlevel.json')
    elif args.use_non_pretrained_tokenizer:
        tokenizer = RobertaTokenizer(vocab_file="../bpe_tokenizer/bpe_tokenizer-vocab.json",
                                     merges_file="../bpe_tokenizer/bpe_tokenizer-merges.txt")
    else:
        tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    if args.use_non_pretrained_model:
        model = RobertaForSequenceClassification(config=config)
    else:
        model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, config=config,
                                                                 ignore_mismatched_sizes=True)
        # TODO: add flowgnn to model
    model = Model(model, flowgnn_model, config, tokenizer, args)

    # print number of params
    def count_params(model):
        if model is None:
            return 0
        return sum(p.numel() for p in model.parameters())

    params = count_params(model.encoder) + count_params(model.classifier)
    if not args.no_flowgnn:
        params += count_params(model.flowgnn_encoder)
    print("parameters:", params)
    print("encoder:", model.encoder)
    print("classifier:", model.classifier)
    if not args.no_flowgnn:
        print("flowgnn_encoder:", model.flowgnn_encoder)

    # Training
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, file_type='train', return_index=True)
        eval_dataset = TextDataset(tokenizer, args, file_type='eval', return_index=True)
        model = train(args, train_dataset, model, tokenizer, eval_dataset, flowgnn_dataset)

        output_dir = os.path.join(args.output_dir, 'checkpoint-best-f1', args.model_name)
        test_dataset = TextDataset(tokenizer, args, file_type='test', return_index=True)
        args.input_file_dir =  os.path.dirname(output_dir)
        t_test(args, model, tokenizer, test_dataset, flowgnn_dataset, output_dir, best_threshold=0.5)
        test_dataset = TextDataset(tokenizer, args, file_type='test_add_after', return_index=True)
        t_test(args, model, tokenizer, test_dataset, flowgnn_dataset, output_dir, best_threshold=0.5)
    # Evaluation
    if args.do_eval:
        if args.add_after_to_before:
            output_dir = os.path.join(args.output_dir, 'checkpoint-best-f1', args.model_name)
        else:
            output_dir = os.path.join(args.output_dir, args.model_name, 'checkpoint-best-f1', 'model.bin')
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        evaluate(args, model, tokenizer)
    # Test
    if args.do_test:
        if args.add_after_to_before:
            output_dir = os.path.join(args.output_dir, 'checkpoint-best-f1',
                                      args.model_name)
        else:
            output_dir = os.path.join(args.output_dir, args.model_name, 'checkpoint-best-f1', 'model.bin')
        print("loading checkpoint '{}'".format(output_dir))
        model.load_state_dict(torch.load(output_dir, map_location=args.device))
        model.to(args.device)
        args.input_file_dir = os.path.dirname(output_dir)

        test_dataset = TextDataset(tokenizer, args, file_type='test', return_index=True)
        t_test(args, model, tokenizer, test_dataset, flowgnn_dataset, output_dir, best_threshold=0.5)
        visual(args, model, tokenizer, test_dataset, flowgnn_dataset, output_dir, best_threshold=0.5)
        test_dataset = TextDataset(tokenizer, args, file_type='test_add_after', return_index=True)
        t_test(args, model, tokenizer, test_dataset, flowgnn_dataset, output_dir, best_threshold=0.5)


if __name__ == "__main__":
    main()
