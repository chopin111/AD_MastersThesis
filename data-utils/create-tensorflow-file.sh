DATA_PATH=/media/piotr/CE58632058630695
DEST_FILENAME=data-tf-5
SRC_FILENAME=data-sorted-5
mkdir -p $DATA_PATH/$DEST_FILENAME
python3 build_image_data.py --train_directory=$DATA_PATH/$SRC_FILENAME/train --output_directory=$DATA_PATH/$DEST_FILENAME  \
--validation_directory=$DATA_PATH/$SRC_FILENAME/validate --labels_file=$DATA_PATH/$SRC_FILENAME/labels.txt   \
--train_shards=16 --validation_shards=8 --num_threads=1
