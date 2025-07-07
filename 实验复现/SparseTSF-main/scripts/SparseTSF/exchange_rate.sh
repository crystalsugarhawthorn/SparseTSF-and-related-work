if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=SparseTSF

root_path_name=./dataset/
data_path_name=exchange_rate.csv
model_id_name=exchange_rate
data_name=exchange_rate

seq_len=600
for pred_len in 60 150 300 600
do
  python -u run_longExp.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --period_len 30 \
    --enc_in 8 \
    --train_epochs 30 \
    --patience 5 \
    --itr 1 --batch_size 128 --learning_rate 0.02
done

