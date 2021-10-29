

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
python jh_ae.py train.lr=1e-4 train.batch_size=128 train.epoch=0 channel_list=v1

<!-- nohup python jh_ae.py -m hydra.sweep.dir=exp/HP_search \
train.lr=1e-4 train.batch_size=128 train.epoch=200 channel_list=\[8,16,32\],\[32,64,128\],\[128,128,128\],\[8,16,32,64\],\[16,32,64,128\],\[128,128,128,128\],\[8,16,32,64,128\],\[128,128,128,128,128\] \
encoder=cnn,cnn_blocks decoder=dcnn,dcnn_blocks,upcnn \
hydra/launcher=joblib hydra.launcher.n_jobs=12 &

nohup python jh_ae.py -m hydra.sweep.dir=exp/HP_search \
train.lr=1e-4 train.batch_size=128 train.epoch=200 channel_list=\[8,16,32\],\[32,64,128\],\[128,128,128\],\[8,16,32,64\],\[16,32,64,128\],\[128,128,128,128\],\[8,16,32,64,128\],\[128,128,128,128,128\] \
encoder=cnn,cnn_blocks decoder=dcnn,dcnn_blocks,upcnn \
hydra/launcher=joblib hydra.launcher.n_jobs=12 & -->

#### HP search
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
python jh_ae.py hydra.run.dir=exp/Ttr train.lr=1e-4 train.batch_size=128 train.epoch=200 encoder=cnn decoder=upcnn scorer.cfg.target_gene=Ttr
- Ttr

@nipa_kks @@
nohup python jh_ae.py -m hydra.sweep.dir=exp/HP_search \
train.lr=1e-4 train.batch_size=128 train.epoch=200 channel_list=v1,v2,v3,v4,v5,v6,v7,v8,v9 \
encoder=cnn,cnn_blocks decoder=dcnn,dcnn_blocks,upcnn \
random.seed="range(0,3)" hydra/launcher=joblib hydra.launcher.n_jobs=24 &

@nipa_kks @@
nohup python jh_ae.py -m hydra.sweep.dir=exp/HP_search3 \
train.lr=1e-4 train.batch_size=128 train.epoch=200 channel_list=c256_3,c256_4,c256_5,c256_6,c512_3,c512_4,c512_5,c512_6 \
encoder=cnn,cnn_bn,cnn_blocks decoder=dcnn,dcnn_blocks,upcnn,upcnn_bn \
random.seed="range(0,3)" hydra/launcher=joblib hydra.launcher.n_jobs=12 &

@nipa_kks @
nohup python jh_ae.py -m hydra.sweep.dir=exp/HP_search4 \
train.lr=1e-4 train.batch_size=128 train.epoch=200 channel_list=c256_3,c256_4,c256_5,c256_6,c512_3,c512_4,c512_5,c512_6 \
encoder=cnn,cnn_bn,cnn_blocks decoder=dcnn,dcnn_blocks,upcnn,upcnn_bn \
random.seed="range(0,3)" hydra/launcher=joblib hydra.launcher.n_jobs=12 &
- fixes in code & evaluation
- encoder: cnn_bn
- decoder: upcnn_bn
- c256_6

@nipa_kks @
nohup python jh_ae.py -m hydra.sweep.dir=exp/HP_search5 \
train.lr=1e-4 train.batch_size=128 train.epoch=200 channel_list=v1,v2,v3,v4,v5,v6,v7,v8,v9 \
encoder=cnn,cnn_bn,cnn_blocks decoder=dcnn,dcnn_blocks,upcnn,upcnn_bn \
random.seed="range(0,3)" hydra/launcher=joblib hydra.launcher.n_jobs=12 &

#### Final result
@nipa_kks @
nohup python jh_ae.py hydra.run.dir=exp/Final2 \
train.lr=1e-4 train.batch_size=128 train.epoch=200 encoder=cnn decoder=dcnn channel_list=v1 \
scorer.cfg.target_gene=Ttr scorer.cfg.plot=True scorer.cfg.save_x=True &
- Ttr
- original HP, not tuned one since HP search was somewhat weird.

nohup python jh_ae.py hydra.run.dir=exp/Final2_Vxn \
train.lr=1e-4 train.batch_size=128 train.epoch=200 encoder=cnn decoder=dcnn channel_list=v1 \
scorer.cfg.target_gene=Vxn &

<!-- nohup python jh_ae.py hydra.run.dir=exp/Final1 \
train.lr=1e-4 train.batch_size=128 train.epoch=200 channel_list=c256_6 \
encoder=cnn decoder=upcnn \
scorer.cfg.plot=True scorer.cfg.save_x=True \
random.seed=0 & -->

@nipa_kks @
nohup python ssim.py hydra.run.dir=exp/SSIM_Vxn scorer.cfg.target_gene=Vxn &
nohup python ssim.py hydra.run.dir=exp/SSIM_Ttr scorer.cfg.target_gene=Ttr &


#### Final result 2
@nipa_kks
nohup python jh_ae.py hydra.run.dir=exp/Final3 \
train.lr=1e-4 train.batch_size=128 train.epoch=200 encoder=cnn_bn decoder=upcnn_bn channel_list=c256_6 scorer.cfg.plot=True scorer.cfg.save_x=True augmented=True &

@lab-server
nohup python ssim.py hydra.run.dir=exp/SSIM &



nohup python jh_ae.py train.lr=1e-4 train.batch_size=128 train.epoch=0 encoder=cnn_bn decoder=upcnn_bn channel_list=c256_6 scorer.cfg.plot=True scorer.cfg.save_x=True augmented=False &
python ssim.py


python jh_ae.py hydra.run.dir=exp/Final1 \
train.lr=1e-4 train.batch_size=128 train.epoch=1 channel_list=c256_6 \
encoder=cnn decoder=upcnn \
scorer.cfg.plot=True scorer.cfg.save_x=True
random.seed=0



python ssim.py hydra.run.dir=exp/SSIM_Ttr scorer.cfg.target_gene=Ttr
