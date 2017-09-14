mkdir -p /media/piotr/CE58632058630695/data-tensorflow-cropped-thin
python3 build_image_data.py --train_directory=/media/piotr/CE58632058630695/data-sorted/train --output_directory=/media/piotr/CE58632058630695/data-tensorflow-cropped-thin  \
--validation_directory=/media/piotr/CE58632058630695/data-sorted/validate --labels_file=/media/piotr/CE58632058630695/data-sorted/labels.txt   \
--train_shards=16 --validation_shards=8 --num_threads=1
