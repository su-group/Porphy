chemprop_predict --test_path HOMO_frzn/fold_0/train_full.csv --checkpoint_dir HOMO_frzn --preds_path result/HOMO_frzn_train.csv
chemprop_predict --test_path HOMO_frzn/fold_0/val_full.csv --checkpoint_dir HOMO_frzn --preds_path result/HOMO_frzn_val.csv
chemprop_predict --test_path HOMO_frzn/fold_0/test_full.csv --checkpoint_dir HOMO_frzn --preds_path result/HOMO_frzn_test.csv

chemprop_predict --test_path HOMO_frzn/fold_0/train_full.csv --checkpoint_dir HOMO_transfer --preds_path result/HOMO_train.csv
chemprop_predict --test_path HOMO_frzn/fold_0/val_full.csv --checkpoint_dir HOMO_transfer --preds_path result/HOMO_val.csv
chemprop_predict --test_path HOMO_frzn/fold_0/test_full.csv --checkpoint_dir HOMO_transfer --preds_path result/HOMO_test.csv

chemprop_predict --test_path HOMO_frzn/fold_0/train_full.csv --checkpoint_dir HOMO --preds_path result/HOMO_prediction_train.csv
chemprop_predict --test_path HOMO_frzn/fold_0/val_full.csv --checkpoint_dir HOMO --preds_path result/HOMO_prediction_val.csv
chemprop_predict --test_path HOMO_frzn/fold_0/test_full.csv --checkpoint_dir HOMO --preds_path result/HOMO_prediction_test.csv

chemprop_predict --test_path HOMO_checkpoints/fold_0/train_full.csv --checkpoint_dir HOMO_checkpoints --preds_path result/HOMO_pretrain_train.csv
chemprop_predict --test_path HOMO_checkpoints/fold_0/val_full.csv --checkpoint_dir HOMO_checkpoints --preds_path result/HOMO_pretrain_val.csv
chemprop_predict --test_path HOMO_frzn/fold_0/test_full.csv --checkpoint_dir HOMO_checkpoints --preds_path result/HOMO_pretrain_test.csv
