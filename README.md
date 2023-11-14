# A delve into color space methods for microplastic classification

This code analyze different images fusion schemes to improve microplastics classification. 
The method aims at merging the paired dataset images (amplitude and phase grayscale images) into a single, three-channels picture. 
To run the experiments, please read the instruction.md file.

The HMPD dataset can be downloaded here: [DOWNLOAD HDMP](https://cnrsc-my.sharepoint.com/:u:/g/personal/marco_delcoco_cnr_it/Ed_vtJKpJ7xBtQBzQ8sjEgABjg8RbYHoQxzxzlCoqiy9JA?e=siQehx?download=1) and is presented by the authors in [link](https://github.com/beppe2hd/HMPD) [[1]](https://link.springer.com/chapter/10.1007/978-3-031-43153-1_11).
 
To run the experiments using the HSL data:

- install hsluv package:
  
``` pip install hsluv ```

- generate the HLS data
  
``` python combine.py --type HLS ```

- 
