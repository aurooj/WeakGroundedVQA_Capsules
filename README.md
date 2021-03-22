# Found a Reason for me? Found a Reason for me? Weakly-supervised Grounded Visual Question Answering using Capsules [CVPR2021]

## Abstract
The problem of grounding VQA tasks has seen an increased attention in the research community recently, with most attempts usually focusing on solving this task by using pretrained object detectors which require bounding box annotations for detecting relevant objects in the vocabulary, which may not always be feasible for real-life large-scale applications.
In this paper, we focus on a more relaxed setting: the grounding of relevant visual entities in a weakly supervised manner by training on the VQA task alone. To address this problem, we propose a visual capsule module with a query-based selection mechanism of capsule features, that allows the model to focus on relevant regions based on the textual cues about visual information in the question. We show that integrating the proposed capsule module in existing VQA systems significantly improves their performance on the weakly supervised grounding task. Overall, we demonstrate the effectiveness of our approach on two state-of-the-art VQA systems, stacked NMN and MAC, on the CLEVR-Answers benchmark, our new evaluation set based on CLEVR scenes with groundtruth bounding boxes for objects that are relevant for the correct answer, as well as on GQA, a real world VQA dataset with compositional questions. We show that the systems with the proposed capsule module are consistently outperforming the respective baseline systems in terms of answer grounding while achieving comparable performance on VQA task.

Code and details coming soon...



