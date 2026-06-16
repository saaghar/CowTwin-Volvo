# distilled-fsod

## Installation
Clone this repo:
```bash
git clone https://esw-github.33858.volvo.net/A519359/distilled-fsod.git
cd distilled-fsod
```
Clone CD-ViTO code into the repo:
```bash
git clone https://github.com/lovelyqian/CDFSOD-benchmark
```
Create a Python 3.9 environment:
```bash
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e CDFSOD-benchmark
```

## Running CD-ViTO
Follow the `README.md` and `DATASETS.md` steps from the CDFSOD-benchmark repository. 

If you struggle to follow the steps from the CDFSOD repository, you can try the following:
* In `./lib/categries.py`, add your custom dataset to the `datasets_name` tuple and add the labels to `CLASS_NAME`.
* In `./detectron2/data/datasets/builtin.py`, add you custom dataset to the `datasets_name` tuple.
* In `./build_prototypes.sh` and `main_results`, add your custom dataset to `data_list` and `datalist`, respectively.
* Create a dataset folder with annotations, train and test folders. Use the other benchmark datasets as examples of how to structure your dataset directory.

If you're having problems with `torch.load` in some files, you can either add the parameter `weights_only=False` or downgrade torch to an older version, as in the requirements of the CDFSOD repo.

### Training
* Create the prototypes using the provided script
* Create YAML files for training. Use the structure from the other datasets, but change the directory names and K-shot as needed.
* Make sure you have the appropriate weights for the background classes downloaded.

## Contact
If you have any further questions, contact Anders Björklund at anders.bjorklund@volvo.com.
