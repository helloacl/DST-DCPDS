# DST-DCPDS

We develop our code baed on CHAN](https://github.com/smartyfh/CHAN-DST.git).  


## Requirements

* python3.6

* Install python packages:
~~~
pip install -r requirements_train.text
~~~

## Usages
### Data preprocessing
We conduct experiments on the following datasets:

* MultiWOZ 2.0 [Download](https://www.repository.cam.ac.uk/bitstream/handle/1810/280608/MULTIWOZ2.zip?sequence=3&isAllowed=y) to get `MULTIWOZ2.zip`
* MultiWOZ 2.1 [Download](https://www.repository.cam.ac.uk/bitstream/handle/1810/294507/MULTIWOZ2.1.zip?sequence=1&isAllowed=y) to get `MULTIWOZ2.1.zip`

We use the same preprocessing steps for both datasets. For example, preprocessing MultiWOZ 2.1:
```bash
cd /home/user/DST-DCPDS
# download multiwoz 2.1 dataset
# preprocess datasets for training DST and STP jointly
$ unzip -j MULTIWOZ2.1.zip -d data/multiwoz2.1-update/original
$ cd data/multiwoz2.1-update/original
$ mv ontology.json ..
$ python convert_to_glue_format.py
```

### Pretrained BERT Download
 * Model [Download](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz) to get `bert-base-uncased.tar.gz`
 * Vocab [Download](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt) to get `bert-base-uncased-vocab.txt`
```
cd /home/user/DST-DCPDS
# rename model file
mv bert-base-uncased.tar.gz bert-base-uncased.model
# mv file to target file
mv bert-base-uncased.model /home/user/DST-DCPDS/data/pytorch_bert/.pytorch_pretrained_bert/.
mv bert-base-uncased-vocab.txt /home/user/DST-DCPDS/data/pytorch_bert/.pytorch_pretrained_bert/.
```

### Train
Take MultiWOZ2.1 as an example:

- training for DST-DCPDS for both-level
```bash
bash run_multiwoz2.1_all.sh
```

For `teacher-forcing` set the `mix_teaching_force` to `0`

For `uniform scheduled sampling` set the `mix_teaching_force` to `1` 







  
