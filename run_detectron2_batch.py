# batch version of the thing!

pages_dir = '/Users/jnaiman/Dropbox/wwt_image_extraction/FigureLocalization/BenchMarks/Pages_htrc/RandomSingleFromPDFIndexed/'

detectron2_params_dir = '/Users/jnaiman/Dropbox/wwt_image_extraction/FigureLocalization/BenchMarks/model_params/detectron2/'

# make sure these two are consistent!!!
detectron2_conf = 'DLA_mask_rcnn_X_101_32x8d_FPN_3x.yaml'
detectron2_weights = 'mask_rcnn_R_101_FPN_3x/model_final_trimmed.pth'



# Places of storage for htrc stuffs
ocr_results_dir = '/Users/jnaiman/Dropbox/wwt_image_extraction/FigureLocalization/BenchMarks/OCR_processing_htrc/'
save_binary_dir = '/Users/jnaiman/Dropbox/wwt_image_extraction/FigureLocalization/StoredFeatures/MegaYolo_htrc/'
make_sense_dir = '/Users/jnaiman/Dropbox/wwt_image_extraction/FigureLocalization/BenchMarks/Annotations_htrc/MakeSenseAnnotations/'
images_jpeg_dir = '/Users/jnaiman/Dropbox/wwt_image_extraction/FigureLocalization/BenchMarks/Pages_htrc/RandomSingleFromPDFIndexed/'
full_article_pdfs_dir = '/Users/jnaiman/Dropbox/wwt_image_extraction/FigureLocalization/BenchMarks/data/PMC_htrc/pdfs/'
metric_results_dir = '/Users/jnaiman/Dropbox/wwt_image_extraction/FigureLocalization/MetricsResults/htrc/'


# ----------------------------------------------------------------------

import config

import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import pickle

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from glob import glob
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import yt
yt.enable_parallelism()

from annotation_utils import get_all_ocr_files, collect_ocr_process_results, \
   get_makesense_info_and_years, get_years
from post_processing_utils import parse_annotations_to_labels, \
   get_true_boxes, get_ocr_results, get_image_process_boxes, clean_overlapping_squares, \
   clean_merge_pdfsquares, clean_merge_heurstic_captions, add_heuristic_captions, \
   clean_found_overlap_with_ocr, clean_true_overlap_with_ocr, clean_merge_squares, \
   clean_big_captions, clean_match_fig_cap, expand_true_boxes_fig_cap, \
   expand_found_boxes_fig_cap, expand_true_area_above_cap, expand_found_area_above_cap


# ------------------------------------------------------------------------
binary_dirs = None

if save_binary_dir is None: save_binary_dir = config.save_binary_dir
if ocr_results_dir is None: ocr_results_dir = config.ocr_results_dir
if make_sense_dir is None: make_sense_dir = config.make_sense_dir
if images_jpeg_dir is None: images_jpeg_dir = config.images_jpeg_dir
# from config file
annotation_dir = save_binary_dir + config.ann_name + str(config.IMAGE_H) + 'x' + str(config.IMAGE_W) + '_ann/'

if binary_dirs is None: binary_dirs = 'binaries/'
feature_dir = save_binary_dir + binary_dirs


LABELS, labels, slabels, \
  CLASS, annotations, Y_full = parse_annotations_to_labels(annotation_dir, 
                                                           '', 
                                                           benchmark=True)

# ------------------------------------------------------------------------

nProcs = config.nProcs

classes = ['text', 'title', 'list', 'table', 'figure']
MetadataCatalog.get("dla_val").thing_classes = classes

def setup_cfg(configfile, weightsfile, processingType='cpu', confidence_threshold=0.5):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(configfile)
    argsList = ['MODEL.WEIGHTS',weightsfile, 'MODEL.DEVICE',processingType]
    #print(argsList)
    cfg.merge_from_list(argsList)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    cfg.freeze()
    return cfg

# count:
start_time = time.time()
if yt.is_root(): print('START RUN: ', time.ctime(start_time))

# read in the requested setup
config_file = detectron2_params_dir + detectron2_conf
weights_file = detectron2_params_dir + detectron2_weights

cfg = setup_cfg(config_file, weights_file)

# make predictor
predictor = DefaultPredictor(cfg)

# get pages
pages = glob(pages_dir + '*')

my_storage = {}

wsInds = np.arange(0,len(pages))

iMod = 10

for sto, ipage in yt.parallel_objects(wsInds, nProcs, storage=my_storage):
    if ipage%iMod == 0: print('on', ipage, 'of', len(wsInds)-1)
    
    img = np.array(Image.open(pages[ipage]).convert('RGB'))
    outputs = predictor(img)
    
    instances = outputs["instances"].to("cpu")
    pred_boxes = instances.pred_boxes
    scores = instances.scores
    pred_classes = instances.pred_classes
    
#     # also get true boxes
#     # there is a lot of mess here that gets and formats all true boxes and 
#     #. all of the OCR data
#     a = pages[ipage].split('/')[-1]
#     a = a[:a.rfind('.')]
#     a = annotation_dir+a+'.xml'
    
#     imgs_name, pdfboxes, pdfrawboxes,years_ind, truebox = get_true_boxes(a,LABELS,
#                                                        [],[],
#                                                        annotation_dir=annotation_dir,
#                                                        feature_dir=feature_dir,
#                                                        check_for_file=False)
    
#     # get OCR results and parse them, open image for image processing
#     backtorgb,image_np,rotatedImage,rotatedAngleOCR,bbox_hocr,\
#       bboxes_words,bbsq,cbsq, rotation,bbox_par = get_ocr_results(imgs_name, dfMakeSense,df,
#                                                                  image_np=image.numpy(),
#                                                                   images_jpeg_dir=images_jpeg_dir)

#     # probably do this earlier and pass it...
#     #ff = imgs_name[0].split('/')[-1].split('.npz')[0]
#     #dfMS = dfMakeSense.loc[dfMakeSense['filename']==ff]

#     truebox2 = expand_true_boxes_fig_cap(truebox.copy(), rotatedImage, LABELS)
#     truebox3 = expand_true_area_above_cap(truebox2, rotatedImage, LABELS)
    
    
    height,width = img.shape[0], img.shape[1]
    #print(height,width)
    sto.result = [pages[ipage],pred_boxes, scores, pred_classes,height,width]
    
if yt.is_root():
    pages, boxes, scores, clas = [],[],[],[]
    height,width = [],[]
  
    for ns, v in sorted(my_storage.items()):
        if v is not None:
            pages.append(v[0])
            boxes.append(v[1])
            scores.append(v[2])
            clas.append(v[3])
            height.append(v[4])
            width.append(v[5])

        
    # do a little test save here - locations of squares and figure caption boxes
    pickle_file_name = metric_results_dir + 'detectron2.pickle'
    with open(pickle_file_name, 'wb') as ff:
        pickle.dump([pages,boxes,scores,clas,classes,height,width], ff)
        
    print("DONE at", time.ctime(time.time()))