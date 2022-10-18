# Repo for the Short Conference Paper for the AEOLIAN (Artificial Intelligence for Cultural Organizations) Conference

Make sure you have the right conda environment installed!  See [athing]()

## Order of operations:

### 1. Download papers from HTRC

So far the list is:
 * https://babel.hathitrust.org/cgi/pt?id=uiug.30112101602172&view=1up&seq=16&skin=2021
 
### 2. Run OCR (if needed)

 * `ocr_and_image_processing_batch.py`
 
### 3. Annotate with MakeSense.ai

 * use `pull_check_makesense.ipynb` to give "first guess" using prior model
 
### 4. Process annotations

 * `process_annotations_batch.py`
 
Note: for this to work, you need to have java > 8 installed (should be in the conda installation process).

### 5. Generate features

 * `generate_features.py`
 
 
### 6. Post-process

 * `post_process_tfrecords.py`
 
 
### 7. Check out metrics

 * `explore_calculate_metrics.ipynb`







**NOTE:** there is no re-training of the model at this point (in between Generate features and Post-process there could potentially be a re-training of the model).




 
 
 
# TODO

 - [ ] save model files SOMEWHERE for easy access
 - [ ] add conda install environment file