# pretrain:
chemprop_train --data_path PBDD.csv --dataset_type regression --save_dir PBDD_LUMO  --split_size 0.8 0.1 0.1 --epochs 200 --dropout 0.2  --target_columns LUMO
chemprop_predict --test_path MpPD.csv --checkpoint_dir PBDD_LUMO --preds_path pretrain_predict.csv
# transfer_frzn
chemprop_train --data_path MpPD.csv --dataset_type regression --save_dir LUMO_frzn --split_size 0.8 0.1 0.1 --checkpoint_frzn PBDD_LUMO/fold_0/model_0/model.pt --dropout 0.2 --target_columns LUMO --epochs 200 --freeze_first_only --gpu 0 --seed 0

chemprop_predict --test_path LUMO_frzn/fold_0/train_smiles.csv --checkpoint_dir LUMO_frzn --preds_path LUMO_frzn/fold_0/train_predict.csv
chemprop_predict --test_path LUMO_frzn/fold_0/val_smiles.csv --checkpoint_dir LUMO_frzn --preds_path LUMO_frzn/fold_0/val_predict.csv
chemprop_predict --test_path LUMO_frzn/fold_0/test_smiles.csv --checkpoint_dir LUMO_frzn --preds_path LUMO_frzn/fold_0/test_predict.csv

# transfer 
chemprop_train --data_path MpPD.csv --dataset_type regression --save_dir E_gap_transfer --split_size 0.8 0.1 0.1 --checkpoint_dir PBDD_E_gap --dropout 0.2 --target_columns E_gap --epochs 120  --gpu 0 --seed 0 

chemprop_predict --test_path E_gap_transfer/fold_0/train_smiles.csv --checkpoint_dir E_gap_transfer --preds_path E_gap_transfer/fold_0/train_predict.csv
chemprop_predict --test_path E_gap_transfer/fold_0/val_smiles.csv --checkpoint_dir E_gap_transfer --preds_path E_gap_transfer/fold_0/val_predict.csv
chemprop_predict --test_path E_gap_transfer/fold_0/test_smiles.csv --checkpoint_dir E_gap_transfer --preds_path E_gap_transfer/fold_0/test_predict.csv
s
# train without transfer learning
chemprop_train --data_path MpPD.csv --dataset_type regression --save_dir LUMO_frzn --split_size 0.8 0.1 0.1 --dropout 0.2 --target_columns LUMO --epochs 200  --gpu 0 --seed 0

chemprop_predict --test_path LUMO_frzn/fold_0/train_smiles.csv --checkpoint_dir LUMO_frzn --preds_path LUMO_frzn/fold_0/train_predict.csv
chemprop_predict --test_path LUMO_frzn/fold_0/val_smiles.csv --checkpoint_dir LUMO_frzn --preds_path LUMO_frzn/fold_0/val_predict.csv
chemprop_predict --test_path LUMO_frzn/fold_0/test_smiles.csv --checkpoint_dir LUMO_frzn --preds_path LUMO_frzn/fold_0/test_predict.csv
