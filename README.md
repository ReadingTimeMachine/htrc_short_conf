# Repo for the Short Conference Paper for the AEOLIAN (Artificial Intelligence for Cultural Organizations) Conference

## Installation Notes

### General Notes
Make sure you have the right conda environment installed!  See [athing]()


### Detectron2 Notes
Of note: PyTorch should be installed with `conda install pytorch torchvision torchaudio -c pytorch`, and detectron2 should then be installed following the [github instructions](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md#build-detectron2-from-source) via pip with:

```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# (add --user if you don't have permission)

# Or, to install it from a local clone:
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

# On macOS, you may need to prepend the above commands with a few environment variables:
CC=clang CXX=clang++ ARCHFLAGS="-arch x86_64" python -m pip install ...
```

This can also be installed on a google collab [see this collab notebook example]().


Also, specific for this version of detectron2, you need the weights trained on [the PubLayNet dataset](https://github.com/JPLeoRX/detectron2-publaynet).  You can do this in the command line with:

```
wget https://raw.githubusercontent.com/hpanwar08/detectron2/master/configs/DLA_mask_rcnn_R_101_FPN_3x.yaml
wget https://raw.githubusercontent.com/hpanwar08/detectron2/master/configs/DLA_mask_rcnn_R_50_FPN_3x.yaml
wget https://raw.githubusercontent.com/hpanwar08/detectron2/master/configs/DLA_mask_rcnn_X_101_32x8d_FPN_3x.yaml
wget https://raw.githubusercontent.com/facebookresearch/detectron2/main/configs/Base-RCNN-FPN.yaml
```

You will also need to download the trained weights [which you can get from the link in the GitHub repo](https://keybase.pub/jpleorx/detectron2-publaynet/).  Just make sure you match up the model training file name with the one you want to use for the config file.


## Order of operations:

### 1. Download papers from HTRC

So far the list is:
 * https://babel.hathitrust.org/cgi/pt?id=uiug.30112101602172&view=1up&seq=16&skin=2021
 * https://babel.hathitrust.org/cgi/pt?id=osu.32435023323769&view=1up&seq=199
 
### 2. Run OCR (if needed)

 * `ocr_and_image_processing_batch.py`
 
### 3. Annotate with MakeSense.ai

 * use `pull_check_makesense.ipynb` to give "first guess" using prior model
 
### 4. Process annotations

 * `process_annotations_batch.py`
 
Note: for this to work, you need to have java > 8 installed (should be in the conda installation process).

### 5. Generate features

 * `generate_features.py`
 
 
### 6. Find boxes and post-process

 * Reading Time Machine: `post_process_tfrecords.py`
 * detectron2: `run_detectron2_batch.py` (see the `run_detectron2.ipynb` for more details)
 
 
### 7. Check out metrics

 * `explore_calculate_metrics.ipynb`







**NOTE:** there is no re-training of the model at this point (in between Generate features and Post-process there could potentially be a re-training of the model).




 
 
 
# TODO

 - [ ] save model files SOMEWHERE for easy access
 - [ ] add conda install environment file
 - [ ] add detectron2 colab example file