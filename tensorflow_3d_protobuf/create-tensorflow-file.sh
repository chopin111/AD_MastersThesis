mkdir -p data-tensorflow
python3 build_image_data.py --train_directory=./data/train --output_directory=./data-tensorflow  \
--validation_directory=./data/validate --labels_file=./data/labels.txt   \
--train_shards=1 --validation_shards=1 --num_threads=1
