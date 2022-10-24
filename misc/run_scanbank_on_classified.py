# this will run deepfigures/ScanBank on the classified images
# COPY THIS FILE THE host-input DIRECTORY!!!

# need to be in docker image:
# sudo docker run  -it --volume /Users/jnaiman/Dropbox/wwt_image_extraction/FigureLocalization/BenchMarks/deepfigures-results:/work/host-output --volume /Users/jnaiman/Dropbox/wwt_image_extraction/FigureLocalization/BenchMarks/data/htrc/pdfs:/work/host-input sampyash/vt_cs_6604_digital_libraries:deepfigures_gpu_0.0.5 /bin/bash

# then do python host-input/run_deepfigures_on_classified.py


pdf_storage = '/Users/jnaiman/Dropbox/wwt_image_extraction/FigureLocalization/BenchMarks/data/htrc/pdfs/'
#testList = 'testList_scanbank.csv'


overwrite = False




from deepfigures.extraction import pipeline
import pickle
import time
import shutil
import pandas as pd
import os
from glob import glob
#import shutil

starttime = time.time()

output_directory = 'host-output/'
input_directory = 'host-input/'
    
# check for duplicate downloaded PDFs (some scanned pages come from same article PDF)
# ws1 = pd.read_csv(input_directory+testList,names=['filename'])['filename'].values.tolist()
# ws = []
# for w in ws1:
#     ws.append(pdf_storage+w.split('/')[-1].split('_p')[0]+'.pdf')
# df = pd.DataFrame({'ws':ws})
# df = df.drop_duplicates('ws')
# ws = df['ws'].values

pdfs = glob('~/host_input/'+'*.pdf')
print(pdfs)

# copy this file to the pdf directory
####shutil.copyfile('./run_scanbank_on_classified.py', pdf_storage+'run_scanbank_on_classified.py')



iMod = 10

for i in range(len(ws)):
    if i%iMod == 0: print('on', i, 'of', len(ws))

    pdf_path = input_directory + ws[i].split('/')[-1]
    figure_extractor = pipeline.FigureExtractionPipeline()
    # remove if there -- assume overwriting
    if os.path.exists(output_directory+ws[i].split('/')[-1].split('.pdf')[0]) and overwrite:
        shutil.rmtree(output_directory+ws[i].split('/')[-1].split('.pdf')[0]) 
    try:
        figure_extractor.extract(pdf_path, output_directory+ws[i].split('/')[-1].split('.pdf')[0])
    except:
        print('*****PASSING ON:', i, ws[i])
        pass

print('done in', (time.time()-starttime)/60., 'minutes')