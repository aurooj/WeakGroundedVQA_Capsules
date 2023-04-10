For feature extraction, you could have a look at extract_features.py file. You might need to modify it to get an image_info file (explained below).  

Data files for GQA were provided by MAC network's original code repository. The file formats are discussed here: https://cs.stanford.edu/people/dorarad/gqa/download.html

gqa_spatial.h5 basically has extracted grid features of size (2048,7,7). 

config.imgsInfoFilename reads a dictionary which has image_id to index_in_feature_file mapping in the following format:
{"1": {"index": 0},
 "2": {"index": 1}, 
"3": {"index": 2},
.
.
.
}
where, key is the image id, and index is the extracted feature index in gqa_spatial.h5 file. 

To test on VQA, you can create data files in the similar format and use them for training/testing.
