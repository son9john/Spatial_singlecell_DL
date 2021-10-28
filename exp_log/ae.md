

python jh_ae.py

python jh_ae.py save_x=True


python jh_ae.py save_x=True train.lr=1e-4 train.batch_size=128 channel_list=\[8,16,32\]

python jh_ae.py save_x=True train.lr=1e-4 train.batch_size=128 channel_list=\[8,16,32\] encoder=cnn_blocks
python jh_ae.py save_x=True train.lr=1e-4 train.batch_size=128 channel_list=\[32,64,128\] encoder=cnn_blocks


python jh_ae.py save_x=True train.lr=1e-4 train.batch_size=128 channel_list=\[32,64,128\] encoder=cnn_blocks criterion=bce_loss


python jh_ae.py train.lr=1e-4 train.batch_size=128 train.epoch=1 channel_list=\[32,64,128\] encoder=cnn_blocks criterion=bce_loss


python jh_ae.py train.lr=1e-4 train.batch_size=128 train.epoch=1 channel_list=\[32,64,128\] encoder=cnn_blocks decoder=dcnn_blocks criterion=bce_loss
python jh_ae.py train.lr=1e-4 train.batch_size=128 train.epoch=1 channel_list=\[32,64,128\] encoder=cnn decoder=dcnn criterion=bce_loss
python jh_ae.py train.lr=1e-4 train.batch_size=128 train.epoch=1 channel_list=\[32,64,128\] encoder=cnn decoder=upcnn criterion=bce_loss

python jh_ae.py train.lr=1e-4 train.batch_size=128 train.epoch=1 channel_list=[32,64,128] encoder=cnn decoder=upcnn criterion=bce_loss
python jh_ae.py train.lr=1e-4 train.batch_size=128 train.epoch=1 channel_list="[32,64,128]"
python jh_ae.py train.lr=1e-4 train.batch_size=128 train.epoch=1 channel_list=v1

<!-- nohup python jh_ae.py -m hydra.sweep.dir=exp/HP_search \
train.lr=1e-4 train.batch_size=128 train.epoch=200 channel_list=\[8,16,32\],\[32,64,128\],\[128,128,128\],\[8,16,32,64\],\[16,32,64,128\],\[128,128,128,128\],\[8,16,32,64,128\],\[128,128,128,128,128\] \
encoder=cnn,cnn_blocks decoder=dcnn,dcnn_blocks,upcnn \
hydra/launcher=joblib hydra.launcher.n_jobs=12 &

nohup python jh_ae.py -m hydra.sweep.dir=exp/HP_search \
train.lr=1e-4 train.batch_size=128 train.epoch=200 channel_list=\[8,16,32\],\[32,64,128\],\[128,128,128\],\[8,16,32,64\],\[16,32,64,128\],\[128,128,128,128\],\[8,16,32,64,128\],\[128,128,128,128,128\] \
encoder=cnn,cnn_blocks decoder=dcnn,dcnn_blocks,upcnn \
hydra/launcher=joblib hydra.launcher.n_jobs=12 & -->

@nipa_kks @@
nohup python jh_ae.py -m hydra.sweep.dir=exp/HP_search2 \
train.lr=1e-4 train.batch_size=128 train.epoch=200 channel_list=v1,v2,v3,v4,v5,v6,v7,v8,v9 \
encoder=cnn,cnn_blocks decoder=dcnn,dcnn_blocks,upcnn \
hydra/launcher=joblib hydra.launcher.n_jobs=6 &
- eval cuda error, loaded all data onto GPU

@nipa_kks @@
python jh_ae.py train.lr=1e-4 train.batch_size=128 train.epoch=200 encoder=cnn decoder=dcnn scorer.cfg.target_gene=Vxn
- Vxn


@nipa_kks @@
nohup python jh_ae.py -m hydra.sweep.dir=exp/HP_search \
train.lr=1e-4 train.batch_size=128 train.epoch=200 channel_list=v1,v2,v3,v4,v5,v6,v7,v8,v9 \
encoder=cnn,cnn_blocks decoder=dcnn,dcnn_blocks,upcnn \
random.seed="range(0,3)" hydra/launcher=joblib hydra.launcher.n_jobs=24 &

@nipa_kks @
nohup python jh_ae.py -m hydra.sweep.dir=exp/HP_search3 \
train.lr=1e-4 train.batch_size=128 train.epoch=200 channel_list=c256_3,c256_4,c256_5,c256_6,c512_3,c512_4,c512_5,c512_6 \
encoder=cnn,cnn_blocks decoder=dcnn,dcnn_blocks,upcnn \
random.seed="range(0,3)" hydra/launcher=joblib hydra.launcher.n_jobs=12 &
