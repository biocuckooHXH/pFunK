"""
Filename: SHAP_analysis.py
Author: yellower
"""

import numexpr
import pandas as pd
import shap
from keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import get_session
import math
tf.compat.v1.disable_v2_behavior()
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import train_test_split


class Config():
    epochs = 20
    feature_epochs = 30
    inte_epochs = 20
    batch_size = 1024
    lr = 0.01
    K_tasks = 2


def softmax(x):
    e_x = np.exp(x - np.max(x))  # 避免数值稳定性问题
    return e_x / e_x.sum()

# def plot_feature_importance(feature_importance_df_group):
#     # 设置图形风格
#     sns.set(style="whitegrid")
#     df = feature_importance_df_group.copy()
#     # 创建一个水平条形图
#     plt.figure(figsize=(10, 6))
#     ax = sns.barplot(x='Average SHAP', y='group', data=df, orient='h')
#
#     max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
#     # df['Average SHAP'] = df[['Average SHAP']].apply(max_min_scaler)
#     df['Average SHAP'] = df[['Average SHAP']].apply(softmax)
#     # 设置图形标题和标签
#     plt.title('Feature Importance Grouped by Group')
#     plt.xlabel('Average SHAP Value')
#     plt.ylabel('Group')
#
#     # 显示平均 SHAP 值在条形图上
#     for index, row in df.iterrows():
#         ax.text(row['Average SHAP'], index, f'{row["Average SHAP"]:.2f}', va='center')
#
#     # 显示图形
#     plt.show()
#     return df

if __name__ == '__main__':


    # function_feature = pd.read_table('./data/ten_features/paper_CPLM_K_cd_hit_function_feature_final.txt', sep='\t', index_col=0)
    # function_K_df = pd.read_table('./data/paper_CPLM_K_cd_hit_function.txt', header=None)
    # function_K_df.columns = ['Protein accession', 'Position', 'fasta_K', 'label']
    # function_K_df['id'] = function_K_df.apply(lambda x: '_'.join([x['Protein accession'], str(x['Position'])]), axis=1)
    # feature_model = load_model('E:/codes/KBHB/data/Kbhb_function_features_model/best_model_final.h5')
    # print(feature_model.outputs[0][:,0])
    # tf.gradients(feature_model.outputs[0][:,0], feature_model.inputs)
    # feature = function_K_df.merge(function_feature, on='id')
    #
    # shap.initjs()
    # # shap.explainers._deep.deep_tf.op_handlers[
    # #     "AddV2"] = shap.explainers._deep.deep_tf.passthrough  # this solves the "shap_ADDV2" problem but another one will appear
    # # shap.explainers._deep.deep_tf.op_handlers[
    # #     "FusedBatchNormV3"] = shap.explainers._deep.deep_tf.passthrough  # this solves the next problem which allows you to run the DeepExplainer.
    #
    # explainer = shap.DeepExplainer(feature_model, feature.iloc[:,5:].values)
    # num_explanations = 5
    # # num_explanations = len(feature)
    # shap_vals = explainer.shap_values(feature.iloc[:num_explanations,5:].values, check_additivity=False)
    # shap.summary_plot(shap_vals[1])
    # np.save('./Shaps/shap_vals.npy', shap_vals)


    # indentified_CPLM_all_K_concat = pd.read_csv('./data/ten_features/indentified_CPLM_all_K_concat_cd-hit_feature_merge_norm.txt',sep='\t',nrows=100)
    # indentified_CPLM_all_K_concat_sam , __, __, __ = train_test_split(indentified_CPLM_all_K_concat,indentified_CPLM_all_K_concat['label'], train_size=0.15, stratify=indentified_CPLM_all_K_concat['label'])
    #
    # shap.initjs()
    # Feature_model = load_model("./data/models/Kbhb_sites_features_model_norm.h5")

    pFunK_T = load_model("./models/pFunK-T.model")
    amino_acid_dict = {'*': 0, 'S': 1, 'L': 2, 'R': 3, 'D': 4, 'F': 5, 'Q': 6, 'V': 7, 'T': 8, 'A': 9, 'X': 10, 'N': 11,
                       'E': 12, 'W': 13, 'I': 14, 'P': 15, 'M': 16, 'Y': 17, 'K': 18, 'G': 19, 'H': 20, 'U': 21,
                       'C': 22}
    indentified_CPLM_all_K_concat = pd.read_excel('./data/indentified_CPLM_all_K_concat_cd-hit.xlsx').rename(columns={'Uniprot_ID': 'Protein accession'})
    indentified_CPLM_all_K_concat_sam, __, indentified_CPLM_all_K_concat_sam_label, __ = train_test_split(indentified_CPLM_all_K_concat,
                                                                     indentified_CPLM_all_K_concat['label'],
                                                                     train_size=0.15,
                                                                     stratify=indentified_CPLM_all_K_concat['label'])

    fasta_K_emb_list = []
    # label_list = []
    for item in indentified_CPLM_all_K_concat_sam['fasta_K']:
        fasta_K_emb_list.append([amino_acid_dict[i] for i in item])
        if len(item) != 61:
            print(len(item))

    print(len(fasta_K_emb_list))
    # indentified_CPLM_all_K_concat_sam = indentified_CPLM_all_K_concat_sam.fillna(0)
    shap.explainers._deep.deep_tf.op_handlers[
        "AddV2"] = shap.explainers._deep.deep_tf.passthrough  # this solves the "shap_ADDV2" problem but another one will appear
    shap.explainers._deep.deep_tf.op_handlers[
        "FusedBatchNormV3"] = shap.explainers._deep.deep_tf.passthrough  # this solves the next problem which allows you to run the DeepExplainer.

    explainer = shap.DeepExplainer(pFunK_T, np.array(fasta_K_emb_list))
    num_explanations = len(indentified_CPLM_all_K_concat_sam)
    print(num_explanations)
    # num_explanations = len(feature)
    shap_vals = explainer.shap_values(np.array(fasta_K_emb_list),check_additivity=False)
    shap.summary_plot(shap_vals)
    np.save('./Shaps/shap_vals_pFunK_T_trans.npy', shap_vals)

    # shap_vals = np.load('./Shaps/shap_vals_pFunK_T.npy')
    # shap_vals = np.abs(shap_vals)
    #
    # average_shap_per_sample = np.mean(shap_vals, axis=1)[0]
    # # shap.summary_plot(shap_vals, feature.iloc[:num_explanations, 5:].values, feature_names = feature.columns[5:],plot_type='bar')
    # feature_importance_df = pd.DataFrame({'Feature': indentified_CPLM_all_K_concat.columns[5:], 'Average SHAP': average_shap_per_sample})
    #
    # feature_importance_df = feature_importance_df.sort_values(by='Average SHAP', ascending=False)
    # feature_importance_df['group'] = feature_importance_df['Feature'].apply(lambda x: x.split('_')[0])
    #
    # feature_importance_df['Average SHAP'] = feature_importance_df['Average SHAP'].apply(lambda x: abs(x))
    # max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    # feature_importance_df['Average SHAP'] = feature_importance_df[['Average SHAP']].apply(max_min_scaler)
    # feature_importance_df_group = feature_importance_df.groupby('group')['Average SHAP'].mean().reset_index()
    # feature_importance_df_group['Average SHAP'] = feature_importance_df_group['Average SHAP'].apply(lambda x: abs(x))
    # #
    # top_10_features = feature_importance_df.head(10)
    # print(top_10_features)
    #
    # feature_importance_df_group = feature_importance_df_group.sort_values(by='Average SHAP', ascending=False)
    # plt.bar(feature_importance_df_group['group'], feature_importance_df_group['Average SHAP'])
    # plt.xlabel('Group')
    # plt.ylabel('Average SHAP')
    # # plt.title('Average SHAP Values by Group (Sorted)')
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.savefig('./Shaps/figures/shap_Kbhb_T.pdf')
    # plt.show()
    # plt.close()
    # #
    # plt.bar(feature_importance_df_group['group'], feature_importance_df_group['Average SHAP'], zorder=2)
    # plt.yscale('log')  # 使用对数刻度
    # # plt.xlabel('Group')
    # plt.ylabel('SHAP value (log scale)')
    # plt.xticks(rotation=45)
    #
    # # 添加网格线，并将网格线的绘制顺序设置为1（较低的值表示在下面）
    # plt.grid(True, zorder=1)
    # plt.tight_layout()
    # # 关闭次刻度线
    # plt.minorticks_off()
    # plt.savefig('./Shaps/figures/shap_Kbhb_function_log_T.pdf')
    # plt.show()
    # plt.close()

    # 使用函数展示 feature_importance_df_group
    # df = plot_feature_importance(feature_importance_df_group)
    # indentified_CPLM_all_K_concat = pd.read_excel('./data/indentity_all_K_cd-hit.xlsx', index_col=0)






