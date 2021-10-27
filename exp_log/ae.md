

python jh_ae.py

python jh_ae.py save_x=True


python jh_ae.py save_x=True train.lr=1e-4 train.batch_size=128 channel_list=[8,16,32]

python jh_ae.py save_x=True train.lr=1e-4 train.batch_size=128 channel_list=[8,16,32] encoder=cnn_blocks
python jh_ae.py save_x=True train.lr=1e-4 train.batch_size=128 channel_list=[32,64,128] encoder=cnn_blocks


python jh_ae.py save_x=True train.lr=1e-4 train.batch_size=128 channel_list=[32,64,128] encoder=cnn_blocks criterion=bce_loss
