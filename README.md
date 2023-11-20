# A delve into color space methods for microplastic classification

This code analyze different images fusion schemes to improve microplastics classification. 
The method aims at merging the paired dataset images (amplitude and phase grayscale images) into a single, three-channels picture. 

The HMPD dataset can be downloaded [at this link](https://cnrsc-my.sharepoint.com/:u:/g/personal/marco_delcoco_cnr_it/Ed_vtJKpJ7xBtQBzQ8sjEgABjg8RbYHoQxzxzlCoqiy9JA?e=siQehx?download=1) and is presented by the authors in [this repository](https://github.com/beppe2hd/HMPD) [[1]](https://link.springer.com/chapter/10.1007/978-3-031-43153-1_11).
 
To run the experiments using the HSL data:

- install the required packages

 ``` pip install -r requirements.txt ```

- install the _hsluv package_
  
``` pip install hsluv ```

- generate the HLS data
  
``` python combine.py --type HLS ```

- modify the yaml config file to use the desired type of data and backbone network. For example, to use HSL and DenseNet model, create the following yaml file:
``` 
num_classes: 2
channel: 'HLS'
num_epochs: 25
learning_rate: 0.00001
train_CNN: True
batch_size: 32
shuffle: True
pin_memory: True
num_workers: 8
transform_resize: [100, 100]
transform_crop: [80, 80]
transform_normalize_mean: [0.5, 0.5, 0.5]
transform_normalize_var: [0.5, 0.5, 0.5]
listofNetwork:
  'densenet121': 'models.densenet121()'
```

-run the training:
``` python banckmark_staticFolds.py --device gpu --name newtest --config ./config/costuomConf.yaml --dataset <basepath>/Microplastiche/images --gt <basepath_folding_files> ``` 

-run the test:
``` python report.py --name <newtest> ``` 
-Check the obtained results in the tests/<newtest>/results.json file 

