chemprop_predict --test_path LUMO_frzn/fold_0/train_full.csv --checkpoint_dir LUMO_frzn --preds_path result/LUMO_frzn_train.csv
chemprop_predict --test_path LUMO_frzn/fold_0/val_full.csv --checkpoint_dir LUMO_frzn --preds_path result/LUMO_frzn_val.csv
chemprop_predict --test_path LUMO_frzn/fold_0/test_full.csv --checkpoint_dir LUMO_frzn --preds_path result/LUMO_frzn_test.csv

chemprop_predict --test_path LUMO_frzn/fold_0/train_full.csv --checkpoint_dir LUMO_transfer --preds_path result/LUMO_train.csv
chemprop_predict --test_path LUMO_frzn/fold_0/val_full.csv --checkpoint_dir LUMO_transfer --preds_path result/LUMO_val.csv
chemprop_predict --test_path LUMO_frzn/fold_0/test_full.csv --checkpoint_dir LUMO_transfer --preds_path result/LUMO_test.csv

chemprop_predict --test_path LUMO_frzn/fold_0/train_full.csv --checkpoint_dir LUMO --preds_path result/LUMO_prediction_train.csv
chemprop_predict --test_path LUMO_frzn/fold_0/val_full.csv --checkpoint_dir LUMO --preds_path result/LUMO_prediction_val.csv
chemprop_predict --test_path LUMO_frzn/fold_0/test_full.csv --checkpoint_dir LUMO --preds_path result/LUMO_prediction_test.csv

chemprop_predict --test_path LUMO_checkpoints/fold_0/train_full.csv --checkpoint_dir LUMO_checkpoints --preds_path result/LUMO_pretrain_train.csv
chemprop_predict --test_path LUMO_checkpoints/fold_0/val_full.csv --checkpoint_dir LUMO_checkpoints --preds_path result/LUMO_pretrain_val.csv
chemprop_predict --test_path LUMO_frzn/fold_0/test_full.csv --checkpoint_dir LUMO_checkpoints --preds_path result/LUMO_pretrain_test.csv
