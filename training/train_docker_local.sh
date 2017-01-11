nvidia-docker run --rm --name resnet \
	-v /mnt/disk3/proj/DREAM2016_dm/DREAM2016_dm/training/metadata:/metadata \
	-v /mnt/disk3/proj/DREAM2016_dm/DREAM2016_dm/training/trainingData:/trainingData \
	-v /mnt/disk3/proj/DREAM2016_dm/DREAM2016_dm/training/preprocessedData:/preprocessedData \
	-v /mnt/disk3/proj/DREAM2016_dm/DREAM2016_dm/training/modelState:/modelState \
	-v /mnt/disk3/proj/DREAM2016_dm/DREAM2016_dm/training/scratch:/scratch \
	docker.synapse.org/syn7890435/dm-ls-train-dl:im288mv_net50_bs32 /train_small.sh
