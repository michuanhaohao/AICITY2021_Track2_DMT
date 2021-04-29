# ResNext101-IBN-a
python train.py --config_file configs/stage1/resnext101a_384.yml MODEL.DEVICE_ID "('0')"
python train_stage2_v1.py --config_file configs/stage2/resnext101a_384.yml MODEL.DEVICE_ID "('0')" OUTPUT_DIR './logs/stage2/resnext101a_384/v1'
python train_stage2_v2.py --config_file configs/stage2/resnext101a_384.yml MODEL.DEVICE_ID "('0')" OUTPUT_DIR './logs/stage2/resnext101a_384/v2'

python test.py --config_file configs/stage2/resnext101a_384.yml MODEL.DEVICE_ID "('0')" TEST.WEIGHT './logs/stage2/resnext101a_384/v1/resnext101_ibn_a_2.pth' OUTPUT_DIR './logs/stage2/resnext101a_384/v1'
python test.py --config_file configs/stage2/resnext101a_384.yml MODEL.DEVICE_ID "('0')" TEST.WEIGHT './logs/stage2/resnext101a_384/v2/resnext101_ibn_a_2.pth' OUTPUT_DIR './logs/stage2/resnext101a_384/v2'


# ResNet101-IBN-a
python train.py --config_file configs/stage1/101a_384.yml MODEL.DEVICE_ID "('1')"
python train_stage2_v1.py --config_file configs/stage2/101a_384.yml MODEL.DEVICE_ID "('1')" OUTPUT_DIR './logs/stage2/101a_384/v1'
python train_stage2_v2.py --config_file configs/stage2/101a_384.yml MODEL.DEVICE_ID "('1')" OUTPUT_DIR './logs/stage2/101a_384/v2'

python test.py --config_file configs/stage2/101a_384.yml MODEL.DEVICE_ID "('1')" TEST.WEIGHT './logs/stage2/101a_384/v1/resnet101_ibn_a_2.pth' OUTPUT_DIR './logs/stage2/101a_384/v1'
python test.py --config_file configs/stage2/101a_384.yml MODEL.DEVICE_ID "('1')" TEST.WEIGHT './logs/stage2/101a_384/v2/resnet101_ibn_a_2.pth' OUTPUT_DIR './logs/stage2/101a_384/v2'


# ResNet101-IBN-a (recrop)
python train.py --config_file configs/stage1/101a_384_recrop.yml MODEL.DEVICE_ID "('2')"
python train_stage2_v1.py --config_file configs/stage2/101a_384_recrop.yml MODEL.DEVICE_ID "('2')" OUTPUT_DIR './logs/stage2/101a_384_recrop/v1'
python train_stage2_v2.py --config_file configs/stage2/101a_384_recrop.yml MODEL.DEVICE_ID "('2')" OUTPUT_DIR './logs/stage2/101a_384_recrop/v2'

python test.py --config_file configs/stage2/101a_384_recrop.yml MODEL.DEVICE_ID "('2')" TEST.WEIGHT './logs/stage2/101a_384_recrop/v1/resnet101_ibn_a_2.pth' OUTPUT_DIR './logs/stage2/101a_384_recrop/v1'
python test.py --config_file configs/stage2/101a_384_recrop.yml MODEL.DEVICE_ID "('2')" TEST.WEIGHT './logs/stage2/101a_384_recrop/v2/resnet101_ibn_a_2.pth' OUTPUT_DIR './logs/stage2/101a_384_recrop/v2'


# ResNet101-IBN-a (spgan)
python train.py --config_file configs/stage1/101a_384_spgan.yml MODEL.DEVICE_ID "('3')"
python train_stage2_v1.py --config_file configs/stage2/101a_384_spgan.yml MODEL.DEVICE_ID "('3')" OUTPUT_DIR './logs/stage2/101a_384_spgan/v1'
python train_stage2_v2.py --config_file configs/stage2/101a_384_spgan.yml MODEL.DEVICE_ID "('3')" OUTPUT_DIR './logs/stage2/101a_384_spgan/v2'

python test.py --config_file configs/stage2/101a_384_spgan.yml MODEL.DEVICE_ID "('3')" TEST.WEIGHT './logs/stage2/101a_384_spgan/v1/resnet101_ibn_a_2.pth' OUTPUT_DIR './logs/stage2/101a_384_spgan/v1'
python test.py --config_file configs/stage2/101a_384_spgan.yml MODEL.DEVICE_ID "('3')" TEST.WEIGHT './logs/stage2/101a_384_spgan/v2/resnet101_ibn_a_2.pth' OUTPUT_DIR './logs/stage2/101a_384_spgan/v2'


# DenseNet169-IBN-a 
python train.py --config_file configs/stage1/densenet169a_384.yml MODEL.DEVICE_ID "('4')"
python train_stage2_v1.py --config_file configs/stage2/densenet169a_384.yml MODEL.DEVICE_ID "('4')" OUTPUT_DIR './logs/stage2/densenet169a_384/v1'
python train_stage2_v2.py --config_file configs/stage2/densenet169a_384.yml MODEL.DEVICE_ID "('4')" OUTPUT_DIR './logs/stage2/densenet169a_384/v2'

python test.py --config_file configs/stage2/densenet169a_384.yml MODEL.DEVICE_ID "('4')" TEST.WEIGHT './logs/stage2/densenet169a_384/v1/densenet169_ibn_a_2.pth' OUTPUT_DIR './logs/stage2/densenet169a_384/v1'
python test.py --config_file configs/stage2/densenet169a_384.yml MODEL.DEVICE_ID "('4')" TEST.WEIGHT './logs/stage2/densenet169a_384/v2/densenet169_ibn_a_2.pth' OUTPUT_DIR './logs/stage2/densenet169a_384/v2'


# ResNest101 
python train.py --config_file configs/stage1/s101_384.yml MODEL.DEVICE_ID "('5')"
python train_stage2_v1.py --config_file configs/stage2/s101_384.yml MODEL.DEVICE_ID "('5')" OUTPUT_DIR './logs/stage2/s101_384/v1'
python train_stage2_v2.py --config_file configs/stage2/s101_384.yml MODEL.DEVICE_ID "('5')" OUTPUT_DIR './logs/stage2/s101_384/v2'

python test.py --config_file configs/stage2/s101_384.yml MODEL.DEVICE_ID "('5')" TEST.WEIGHT './logs/stage2/s101_384/v1/resnest101_2.pth' OUTPUT_DIR './logs/stage2/s101_384/v1'
python test.py --config_file configs/stage2/s101_384.yml MODEL.DEVICE_ID "('5')" TEST.WEIGHT './logs/stage2/s101_384/v2/resnest101_2.pth' OUTPUT_DIR './logs/stage2/s101_384/v2'


# SeResNet101-IBN-a
python train.py --config_file configs/stage1/se_resnet101a_384.yml MODEL.DEVICE_ID "('6')"
python train_stage2_v1.py --config_file configs/stage2/se_resnet101a_384.yml MODEL.DEVICE_ID "('6')" OUTPUT_DIR './logs/stage2/se_resnet101a_384/v1'
python train_stage2_v2.py --config_file configs/stage2/se_resnet101a_384.yml MODEL.DEVICE_ID "('6')" OUTPUT_DIR './logs/stage2/se_resnet101a_384/v2'

python test.py --config_file configs/stage2/se_resnet101a_384.yml MODEL.DEVICE_ID "('6')" TEST.WEIGHT './logs/stage2/se_resnet101a_384/v1/se_resnet101_ibn_a_2.pth' OUTPUT_DIR './logs/stage2/se_resnet101a_384/v1'
python test.py --config_file configs/stage2/se_resnet101a_384.yml MODEL.DEVICE_ID "('6')" TEST.WEIGHT './logs/stage2/se_resnet101a_384/v2/se_resnet101_ibn_a_2.pth' OUTPUT_DIR './logs/stage2/se_resnet101a_384/v2'


# TransReID
python train.py --config_file configs/stage1/transreid_256.yml MODEL.DEVICE_ID "('7')"
python train_stage2_v1.py --config_file configs/stage2/transreid_256.yml MODEL.DEVICE_ID "('7')" OUTPUT_DIR './logs/stage2/transreid_256/v1'
python train_stage2_v2.py --config_file configs/stage2/transreid_256.yml MODEL.DEVICE_ID "('7')" OUTPUT_DIR './logs/stage2/transreid_256/v2'

python test.py --config_file configs/stage2/transreid_256.yml MODEL.DEVICE_ID "('7')" TEST.WEIGHT './logs/stage2/transreid_256/v1/transformer_2.pth' OUTPUT_DIR './logs/stage2/transreid_256/v1'
python test.py --config_file configs/stage2/transreid_256.yml MODEL.DEVICE_ID "('7')" TEST.WEIGHT './logs/stage2/transreid_256/v2/transformer_2.pth' OUTPUT_DIR './logs/stage2/transreid_256/v2'

