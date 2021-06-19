# Found a Reason for me? Weakly-supervised Grounded Visual Question Answering using Capsules [CVPR2021]

## Abstract
The problem of grounding VQA tasks has seen an increased attention in the research community recently, with most attempts usually focusing on solving this task by using pretrained object detectors which require bounding box annotations for detecting relevant objects in the vocabulary, which may not always be feasible for real-life large-scale applications.
In this paper, we focus on a more relaxed setting: the grounding of relevant visual entities in a weakly supervised manner by training on the VQA task alone. To address this problem, we propose a visual capsule module with a query-based selection mechanism of capsule features, that allows the model to focus on relevant regions based on the textual cues about visual information in the question. We show that integrating the proposed capsule module in existing VQA systems significantly improves their performance on the weakly supervised grounding task. Overall, we demonstrate the effectiveness of our approach on two state-of-the-art VQA systems, stacked NMN and MAC, on the CLEVR-Answers benchmark, our new evaluation set based on CLEVR scenes with groundtruth bounding boxes for objects that are relevant for the correct answer, as well as on GQA, a real world VQA dataset with compositional questions. We show that the systems with the proposed capsule module are consistently outperforming the respective baseline systems in terms of answer grounding while achieving comparable performance on VQA task.

[[Paper](https://www.crcv.ucf.edu/wp-content/uploads/2018/11/Found-a-Reason-for-me.pdf)] [[Supplementary](https://www.crcv.ucf.edu/wp-content/uploads/2018/11/Found-a-Reason-for-me_Supp.pdf)] [[Presentation Video](https://www.crcv.ucf.edu/wp-content/uploads/2018/11/cvpr_2021_5min.mp4)] [[Poster](https://www.crcv.ucf.edu/wp-content/uploads/2018/11/cvpr21_poster_v2.pdf)]

<!---
### Qualitative Results
!to be added.
--->


### Requirements
We use tensorflow, cuda version 10.1 for our experiments. 

We recommend creating a conda environment to install libraries.
Follow instructions from [SNMN](https://github.com/ronghanghu/snmn) for SNMN, and [MAC](https://github.com/stanfordnlp/mac-network/tree/master) code repos to setup the environments. 

Todo: mention version dependencies

### Datasets
We use two datasets in this work: **GQA** and **CLEVR-Answers**

#### CLEVR-Answers
CLEVR-Answers is an extended version of [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/) dataset for evaluation on answer grounding task. 
We used the [CLEVR dataset generation framework](https://github.com/facebookresearch/clevr-dataset-gen/blob/master/question_generation/README.md) to generate new questions with the bounding box labels for the answers. Each data sample now consists of question, image, answer label and bounding box labels for answer objects.
We provide these labels for CLEVR training and validation sets. 
We call this dataset **CLEVR-Answers** and can be downloaded from [here](https://1drv.ms/u/s!AtxSFigVVA5JhPUP9Pb7xcBFQ5m7rQ?e=wQuzf7).

Following is the file structure for CLEVR-Answers:
```
CLEVR_Answers
|____CLEVR_train_questions_new.json
|____CLEVR_train_question2bboxes.json
|____CLEVR_val_questions_new.json
|____CLEVR_val_question2bboxes.json

```
To have a standard train-val-test setup, we separate 1K training images with 10K question-answer pairs for validation of hyperparameters. We call this set "train-val".
The original validation set is used as test set in all our experiments.

The split of training data into "new-train" and "train-val" is provided [here](https://1drv.ms/u/s!AtxSFigVVA5JhPUQa3fKWdCKZFcyWA?e=x8ryKH).
todo: add file format description

#### GQA
GQA dataset can be downloaded from [here](https://cs.stanford.edu/people/dorarad/gqa/download.html). 
We used the balanced version of GQA for our experiments. 
GQA provides the bounding box annotations for both question and answer objects. We evaluate grounding on this dataset for different grounding ground truths: Question (Q), full answer (FA), short answer (A), and both question and answer objects (All). The bboxes information for each groundtruth type is saved in the same format as CLEVR-Answers. These files can be downloaded from this [link](https://1drv.ms/u/s!AtxSFigVVA5JhPUVHmUpAC7oI7wE5A?e=BX9sLA).

Following is the file structure for GQA:
```
GQA
|____gqa_val_all_question2bboxes.json
|____gqa_val_answer_question2bboxes.json
|____gqa_val_fullAnswer_question2bboxes.json
|____gqa_val_question_question2bboxes.json

```

### Baselines
We integrate our capsule module into two baselines: SNMN and MAC. 
MAC network was trained on both [CLEVR-Answers](https://github.com/stanfordnlp/mac-network/tree/master) and [GQA](https://github.com/stanfordnlp/mac-network/tree/gqa) datasets.

### MAC-Caps
Code for MAC-Caps is shared under directory `mac-capsules`. We report our best results on GQA with 32 capsules. 

#### download GQA features

```
cd data
wget http://nlp.stanford.edu/data/gqa/spatialFeatures.zip
unzip spatialFeatures.zip
cd ../
python merge.py --name spatial 
```

Download data as instructed in MAC original code repo and copy it under the `mac-capsules/data/` folder. 
Download GQA data files from [here](#-gqa) in the `mac-capsules/data/` folder.



#### Training
##### Note
`--WriteDim` depends on the number of capsules. 
`--NUM_VIS_CAPS_L1` denotes the number of primary capsules.
`--NUM_VIS_CAPS_L2 ` denotes the number of visual capsules.
For all experiments, we keep the same number of capsules in primary layer and visual capsule layer i.e., `NUM_VIS_CAPS_L1==NUM_VIS_CAPS_L2==C`. `--WriteDim` therefore is calculated as `Cx(KxK+1)`, where K is the pose dim with pose matrix of size KxK; Activations denote the additional dimension. 

```Hence, 
for C=16, --writeDim=16x17=272
for C=24, --writeDim=24x17=408
for C=32, --writeDim=32x17=544
```
Run the following command to start training MAC-Capsules with 32 capsules for network length 4 on gqa dataset:
```
python main.py --expName "gqaExperiment-Spatial" --train --testedNum 10000  --epochs 25 --netLength 4 @configs/gqa/gqa_spatial.txt --writeDim 544 
--NUM_VIS_CAPS_L1 32 --NUM_VIS_CAPS_L2 32
```
Code and details coming soon...

### Citation
If this work and/or dataset is useful for your research, please cite our paper.

```bibtex
@InProceedings{Urooj_2021_CVPR,
    author    = {Urooj, Aisha and Kuehne, Hilde and Duarte, Kevin and Gan, Chuang and Lobo, Niels and Shah, Mubarak},
    title     = {Found a Reason for me? Weakly-supervised Grounded Visual Question Answering using Capsules},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {8465-8474}
}
```

### Questions?
Please contact 'aishaurooj@gmail.com'


