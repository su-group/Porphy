import os

import pandas as pd
import torch
from rxnfp.models import SmilesLanguageModelingModel, SmilesClassificationModel
from sklearn.model_selection import train_test_split


def predict_data(model_name, data, name):
    predict_list = model_name.predict(data['smiles'].to_list())[0]
    datas = data.copy()
    datas.insert(2, 'pre', predict_list)
    datas.to_csv(name + '.csv', index_label=False, index=False)
    return 0


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

vocab_path = 'vacab.txt'

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
    'output_dir': 'out/LUMO-self',
    'learning_rate': 1e-4,
}

# model_pre = SmilesLanguageModelingModel(model_type='bert', model_name=None, args=args,
#                                         use_cuda=torch.cuda.is_available())

print(torch.cuda.is_available())
df = pd.read_csv('PBDD_analysis.csv')
df = df[['smiles', 'HOMO']]
df_train, df_eval = train_test_split(df, train_size=0.8, random_state=42)
df_train.to_csv('split/train_PBDD.csv')
df_eval.to_csv('split/val_PBDD.csv')
df_test = pd.read_csv('split/test_full.csv')
train_file = f'split/train_PBDD.csv'
eval_file = f'split/val_PBDD.csv'
# model_pre.train_model(train_file=train_file, eval_file=eval_file)

# read data
path = r'out/LUMO-self'

model = SmilesClassificationModel('bert', path, num_labels=1, args={
    'regression': True, 'num_train_epochs': 50
}, use_cuda=torch.cuda.is_available())

model.train_model(df_train[['smiles', 'HOMO']], output_dir=r'train/model_config',
                  eval_df=df_eval)
predict_data(model, df_train, 'PBDD-train')
predict_data(model, df_eval, 'PBDD-valid')
predict_data(model, df_test, 'PBDD-test')
print('finish')
