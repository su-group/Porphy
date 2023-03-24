import os
import torch
from rxnfp.models import SmilesClassificationModel
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
config = {
    "architectures": [
        "BertForMaskedLM"
    ],
    "attention_probs_dropout_prob": 0.1,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 256,
    "initializer_range": 0.02,
    "intermediate_size": 512,
    "layer_norm_eps": 1e-12,
    "max_position_embeddings": 512,
    "model_type": "bert",
    "num_attention_heads": 4,
    "num_hidden_layers": 12,
    "pad_token_id": 0,
    "type_vocab_size": 2,
}
vocab_path = '../data/vacab.txt'
args = {
    'config': config,
    'vocab_path': vocab_path,
    'train_batch_size': 16,
    'manual_seed': 42,
    'fp16': False,
    "num_train_epochs": 50,
    'max_seq_length': 300,
    'evaluate_during_training': True,
    'overwrite_output_dir': True,
    'output_dir': 'out/porphyrin-self',
    'learning_rate': 1e-5,
}

# read data
goal = 'E_gap'
df_train = pd.read_csv('data/train_MpPD.csv')
df_eval = pd.read_csv('data/val_MpPD.csv')
df_test = pd.read_csv('data/test_MpPD.csv')

path = r'outputs/best_model'

model = SmilesClassificationModel('bert', path, num_labels=1, args={
    'regression': True, "num_train_epochs": 43, 'train_batch_size': 16, 'dropout': 0.4, 'learning_rate': 1e-5
}, use_cuda=False)

model.train_model(df_train[['smiles', goal]], output_dir=r'train/model_config',
                  eval_df=df_eval)


def predict_data(model_name, data, name):
    global goal
    predict_list = model_name.predict(data['smiles'])[0]
    datas = data.copy()
    datas.insert(2, 'pre_' + goal, predict_list)
    datas.to_csv(name + '.csv', index_label=False, index=False)
    return datas


df_train_prediction = predict_data(model, df_train, 'result/train_prediction_' + goal)
df_val_prediction = predict_data(model, df_eval, 'result/val_prediction_' + goal)
df_test_prediction = predict_data(model, df_test, 'result/test_prediction_' + goal)


def draw_regression_plot(y_test, y_pred, name='test'):
    sns.set_style('darkgrid')
    plt.rc('font', family='Times New Roman')
    fig, ax = plt.subplots(figsize=(16, 16), dpi=300)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2_patch = mpatches.Patch(label="$\mathregular{R^2}$" + " = {:.4f}".format(r2), color='#0F59A4')
    rmse_patch = mpatches.Patch(label="RMSE = {:.4f} eV".format(rmse), color='#1661AB')
    mae_patch = mpatches.Patch(label="MAE = {:.4f} eV".format(mae), color='#3170A7')
    global goal
    if goal == 'E_gap':
        plt.xlim(1.6, 3.4)
        plt.ylim(1.6, 3.4)
    if goal == 'LUMO':
        plt.xlim(-3.9, -1.6)
        plt.ylim(-3.9, -1.6)
    if goal == 'HOMO':
        plt.xlim(-6.9, -4.1)
        plt.ylim(-6.9, -4.1)

    plt.tick_params(labelsize=40)

    sns.scatterplot(y_pred, y_test, alpha=0.8, color='#144A74', linewidth=0, s=400)
    # plt.scatter(y_pred, y_test, alpha=0.2, color='#2e317c')
    sns.lineplot(np.arange(-10, 10, 0.1), np.arange(-10, 10, 0.1), ls="--", color='#0F1423', alpha=0.3)
    # plt.plot(np.arange(100), np.arange(100), ls="--", c=".3")
    plt.legend(handles=[r2_patch, rmse_patch, mae_patch], fontsize=30)
    ax.set_ylabel('Measured ' + goal + ' (eV)', fontsize=35, labelpad=30)
    ax.set_xlabel('Predicted ' + goal + ' (eV)', fontsize=40, labelpad=30)
    ax.set_title(name, fontsize=50, pad=30)
    fig.savefig('result/' + name + '_regression.png')
    plt.cla()
    return


def draw_histplot(y_test, y_pred, name='test'):
    plt.figure(figsize=(16, 16), dpi=300)
    sns.color_palette("Paired", 8)
    sns.distplot(y_test, color='#20763B', label="Measured LUMO", norm_hist=False, kde=False,
                 hist_kws=dict(edgecolor='#20763B', linewidth=0),
                 bins=13)
    sns.distplot(y_pred, color='#081EAD', label="Predicted LUMO", norm_hist=False, kde=False,
                 hist_kws=dict(edgecolor='#081EAD', linewidth=0),
                 bins=13)
    plt.legend(fontsize=40)
    plt.title(name, fontsize=50, pad=30)
    plt.tick_params(labelsize=38)
    plt.xlabel("LUMO (eV)", fontsize=40, labelpad=30)
    plt.ylabel("Amount", fontsize=50, labelpad=30)
    plt.savefig('result/' + name + '_distplot.png', dpi=300)
    plt.cla()
    return


for i in ['train', 'val', 'test']:
    df_name = 'df_' + i + '_prediction'
    dataframe = eval(df_name)
    draw_regression_plot(dataframe[goal], dataframe['pre_' + goal], 'BERT_Transfer_' + goal + '_' + i)
    draw_histplot(dataframe[goal], dataframe['pre_' + goal], 'BERT_Transfer_' + goal + '_' + i)
print('finish')
