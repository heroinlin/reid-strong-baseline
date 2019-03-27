goto start
# Experiment all tricks with center loss : 144x144-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_centerloss0_0005
# Dataset 2: dukemtmc
# imagesize: 144x144
# batchsize: 16x4
# warmup_step 10
# random erase prob 0.5
# labelsmooth: on
# last stride 1
# bnneck on
# with center loss
:start

call activate_py36.cmd
python tools/train.py --config_file=configs/softmax_triplet_with_center.yml MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('bus_id')" DATASETS.ROOT_DIR "(r'F:\Database\od_dataset\train_format')" OUTPUT_DIR "('checkpoints/bus_id/Experiment-all-tricks')"
pause