import pandas as pd
import os
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt


def get_mertics(y_true, y_pred ):
    # calculate confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    #
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    metrics = {"tn": tn,
               "fp": fp,
               "fn": fn,
               "tp": tp,
               "recall": recall,
               "precision": precision,
               "f1": f1}
    return metrics


def draw_roc_auc(y_true, y_pred_proba, args):

  fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
  roc_auc = auc(fpr, tpr)

  # plt.figure()
  # plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
  # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  # plt.xlim([0.0, 1.0])
  # plt.ylim([0.0, 1.05])
  # plt.xlabel('False Positive Rate')
  # plt.ylabel('True Positive Rate')
  # plt.title('Receiver Operating Characteristic')
  # plt.legend(loc="lower right")
  # plt.savefig(os.path.join(args['file_dir'], args['model_name'] + "_" + args['virson_name'] +'_auroc.jpg'))
  # plt.show()
  return fpr, tpr, roc_auc



def caluclate_mertics_Be_Ue(dir, models_name, virsons_name, columns_name):


    args = {}

    combine_auroc = {}
    Be_Ue_final_metrics = {}
    assembly_metrics = []
    for model_name in models_name:
        Be_Ue_final_metrics[model_name] = {}
        for virson_name in virsons_name:
            Be_Ue_final_metrics[model_name][virson_name] = {}
            for column_name in columns_name:
                file_dir = os.path.join(dir, model_name, virson_name)
                if model_name not in ['CodeBERT'] :
                    if model_name == 'LineVul_Process_func':
                        column_name = 'val_processed_func_prediction.xlsx'
                    else:
                        column_name = column_name[:-3] + "xlsx"
                file_path = os.path.join(file_dir, column_name)
                print("file_path:", file_path)
                # if os.path.exists(file_path):
                #     pass
                # else:
                #     continue

                if model_name not in ['CodeBERT'] :
                    df = pd.read_excel(file_path)
                else:
                    df = pd.read_csv(file_path)
                # print(df)

                args['file_dir'] = file_dir
                args['model_name'] = model_name
                args['virson_name'] = virson_name
                args['column_name'] = column_name
                y_pred_column_name = model_name + "_" + virson_name + "_Predictions"
                y_true = df['y_trues'].tolist()
                y_pred_proba = df['logits'].tolist()
                y_pred = df[y_pred_column_name].tolist()
                metrics = get_mertics(y_true, y_pred)
                Be_Ue_final_metrics[model_name][virson_name] = metrics
                print("\n\n\n*****************************")
                print("model_name:", model_name)
                print("virson_name:", virson_name)
                print("column_name:", column_name)
                print("metrics:", metrics)
                metrics['model_name'] = model_name
                metrics['column_name'] = column_name
                metrics['virson_name'] = virson_name
                assembly_metrics.append(metrics)
                fpr, tpr, roc_auc = draw_roc_auc(y_true, y_pred_proba, args)
                print("roc_auc:", roc_auc)
                if model_name not in combine_auroc:
                    if model_name == 'LineVul_Process_func':
                        combine_auroc[model_name] = {virson_name + " PBe + PUe": {"fpr": fpr,
                                                                                  "tpr": tpr,
                                                                                  "roc_auc": roc_auc}}
                    else:
                        combine_auroc[model_name] = {virson_name + " Be + Ue": {"fpr": fpr,
                                                                                "tpr": tpr,
                                                                                "roc_auc": roc_auc}}
                else:

                    if model_name == 'LineVul_Process_func':
                        combine_auroc[model_name][virson_name + " PBe + PUe"] = {"fpr": fpr,
                                                                                 "tpr": tpr,
                                                                                 "roc_auc": roc_auc}
                    else:
                        combine_auroc[model_name][virson_name + " Be + Ue"] = {"fpr": fpr,
                                                                               "tpr": tpr,
                                                                               "roc_auc": roc_auc}
    return combine_auroc, assembly_metrics


def calculate_mertics_Ae_Be(dir, models_name, virsons_name, columns_name, combine_auroc):


    def all_inputs_ids_process(x):
        x = str(x)
        if '.' in x:
            x = x.split(".")[0]
        return x

    args = {}

    Be_Ae_final_metrics = {}
    assembly_metrics = []
    for model_name in models_name:
        Be_Ae_final_metrics[model_name] = {}
        for virson_name in virsons_name:
            Be_Ae_final_metrics[model_name][virson_name] = {}
            for i in range(len(columns_name)):
                column_name = columns_name[i]
                if model_name not in ['CodeBERT']:
                    if model_name == 'LineVul_Process_func':
                        columns_name = ['val_processed_func_prediction.xlsx', 'val_func_after_prediction.xlsx']
                    else:

                        columns_name = ['val_func_before_prediction.xlsx', 'val_func_after_prediction.xlsx']

            file_dir = os.path.join(dir, model_name, virson_name)
            before_path = os.path.join(file_dir, columns_name[0])
            after_path = os.path.join(file_dir, columns_name[1])

            args['file_dir'] = file_dir
            args['model_name'] = model_name
            args['virson_name'] = virson_name

            if model_name not in ['CodeBERT']:
                before_df = pd.read_excel(before_path)
                after_df = pd.read_excel(after_path)
            else:
                before_df = pd.read_csv(before_path)
                after_df = pd.read_csv(after_path)
            before_df_vul = before_df[before_df.y_trues == 1]
            after_df_vul = after_df[after_df.y_trues == 1]
            predict_column = args['model_name'] + "_" + args['virson_name'] + "_Predictions"
            before_df_vul.rename(inplace=True, columns={"y_trues": "y_trues_before", "logits": "logits_before",
                                                        predict_column: predict_column + "_before"})
            after_df_vul.rename(inplace=True, columns={"y_trues": "y_trues_after", "logits": "logits_after",
                                                       predict_column: predict_column + "_after"})
            after_df_vul['y_trues_after'] = 0

            before_df_vul['all_inputs_ids'] = before_df_vul['all_inputs_ids'].apply(all_inputs_ids_process)
            after_df_vul['all_inputs_ids'] = after_df_vul['all_inputs_ids'].apply(all_inputs_ids_process)
            if model_name in ['ReVeal', 'Devign']:
                before_after = pd.merge(after_df_vul, before_df_vul, how='left', on='all_inputs_ids')
            else:
                before_after = pd.merge(before_df_vul, after_df_vul, how='left', on='all_inputs_ids')
            output_file = 'before_after_paired.csv'
            output_path = os.path.join(args['file_dir'], output_file)
            new_column_order = ['all_inputs_ids', 'y_trues_before', 'y_trues_after', 'logits_before', 'logits_after',
                                predict_column + "_before", predict_column + "_after"]
            before_after = before_after[new_column_order]
            before_after.to_csv(output_path)

            y_true = before_after['y_trues_before'].tolist() + before_after['y_trues_after'].tolist()
            y_pred = before_after[predict_column + "_before"].tolist() + before_after[
                predict_column + "_after"].tolist()
            y_pred_proba = before_after['logits_before'].tolist() + before_after['logits_after'].tolist()

            print("\n\n\n**********paired functions***************")
            print("model_name:", model_name)
            print("virson_name:", virson_name)
            print("y_true:", y_true)
            print("y_pred:", y_pred)
            print("y_pred_proba:", y_pred_proba)
            metrics = get_mertics(y_true, y_pred)
            Be_Ae_final_metrics[model_name][virson_name] = metrics
            print("metrics:", metrics)

            metrics['model_name'] = model_name
            metrics['virson_name'] = virson_name
            metrics['is_paired_func'] = 'paired_func'
            assembly_metrics.append(metrics)
            fpr, tpr, roc_auc = draw_roc_auc(y_true, y_pred_proba, args)

            print("roc_auc:", roc_auc)
            if model_name not in combine_auroc:
                combine_auroc[model_name] = {virson_name + " Be + Ae": {"fpr": fpr,
                                                                        "tpr": tpr,
                                                                        "roc_auc": roc_auc}}
            else:
                if model_name == 'LineVul_Process_func':
                    combine_auroc[model_name][virson_name + " PBe + Ae"] = {"fpr": fpr,
                                                                            "tpr": tpr,
                                                                            "roc_auc": roc_auc}
                else:
                    combine_auroc[model_name][virson_name + " Be + Ae"] = {"fpr": fpr,
                                                                           "tpr": tpr,
                                                                           "roc_auc": roc_auc}

    return  assembly_metrics


if __name__ == '__main__':

    dir = '/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/paired/'
    models_name = ['CodeBERT', 'LineVul', 'DeepDFA_Plus_LineVul', 'DeepDFA_Only', 'LineVul_Process_func', 'ReVeal', 'Devign']
    models_name = ['LineVul']
    # virsons_name = ['add_after_percent_30', 'add_after_percent_50', 'add_after_percent_70'] # original  add_after_to_before
    virsons_name = [ 'original', 'd_0', 'd_0_01', 'd_0_02','d_0_03', 'add_after_percent_30', 'add_after_percent_50','add_after_percent_70',  'add_after_to_before'] # original  add_after_to_before
    # virsons_name = ['add_after_percent_30']
    columns_name = ['val_func_before_prediction.csv']  # 'val_func_after_prediction.csv',
    combine_auroc, assembly_metrics_Be_Ue = caluclate_mertics_Be_Ue(dir, models_name, virsons_name, columns_name)
    assembly_metrics_Be_Ue = pd.DataFrame(assembly_metrics_Be_Ue)
    output_columns_name = ['model_name', 'virson_name', 'precision', 'recall', 'f1','tn','fp','fn','tp']
    assembly_metrics_Be_Ue = assembly_metrics_Be_Ue[output_columns_name]
    assembly_metrics_Be_Ue.to_csv(os.path.join(dir, 'metrics_result', 'assembly_metrics_Be_Ue.csv'))

    columns_name = ['val_func_before_prediction.csv', 'val_func_after_prediction.csv']  # con't change the order
    assembly_metrics_Be_Ae = calculate_mertics_Ae_Be(dir, models_name, virsons_name, columns_name, combine_auroc)
    assembly_metrics_Be_Ae = pd.DataFrame(assembly_metrics_Be_Ae)
    assembly_metrics_Be_Ae = assembly_metrics_Be_Ae[output_columns_name]
    assembly_metrics_Be_Ae.to_csv(os.path.join(dir, 'metrics_result', 'assembly_metrics_Be_Ae.csv'))
