chemprop_train --data_path PBDD_LUMO.csv --dataset_type regression --save_dir HOMO_checkpoints  --split_size 0.8 0.1 0.1 --epochs 200 --dropout 0.2  --target_columns HOMO
chemprop_predict --test_path SMILES-E_gap.csv --checkpoint_dir HOMO_checkpoints --preds_path HOMO_pretrain_predict.csv
