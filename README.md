提供一些处理mscoco的代码
```
note：
1. mscoco共有80类，但是其annotation标注的category_id范围为[1,90]，中间有间断。具体见labels.txt
2. coco的annotation json文件把所有图像的标注信息都写到一起了（annotation_examples目录有示例）
```

###拆分annotation json文件，每张图像单独存储为一个json文件。
`参考https://github.com/weiliu89/caffe/tree/ssd/data/coco`
1. Download Images and Annotations from MSCOCO. By default, we assume the data is stored in $HOME/data/coco

2. Get the coco code. We will call the directory that you cloned coco into $COCO_ROOT
```bash
git clone https://github.com/weiliu89/coco.git
cd coco
git checkout dev
```

3. Build the coco code.
```python
cd PythonAPI
python setup.py build_ext --inplace
```

4. Split the annotation to many files per image and get the image size info.
```python
# Check scripts/batch_split_annotation.py and change settings accordingly.
python scripts/batch_split_annotation.py
# Create the minival2014_name_size.txt and test-dev2015_name_size.txt in $CAFFE_ROOT/data/coco
python scripts/batch_get_image_size.py
```
### 制作训练数据格式
1. For CAFFE: create the LMDB file
```bash
cd $CAFFE_ROOT
# Create the minival.txt, testdev.txt, test.txt, train.txt in data/coco/
python data/coco/create_list.py
# You can modify the parameters in create_data.sh if needed.
# It will create lmdb files for minival, testdev, test, and train with encoded original image:
#   - $HOME/data/coco/lmdb/coco_minival_lmdb
#   - $HOME/data/coco/lmdb/coco_testdev_lmdb
#   - $HOME/data/coco/lmdb/coco_test_lmdb
#   - $HOME/data/coco/lmdb/coco_train_lmdb
# and make soft links at examples/coco/
./data/coco/create_data.sh
```
2. For Tensorflow: create tfrecord
```python
python tf_convert_coco.py
```