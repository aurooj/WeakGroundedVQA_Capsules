The scripts `postprocess.py` and `vis.py` are used to generate detections from attention maps produced from MAC-network.

You will need to update the file paths for data and prediction files according to your file structure.

The script postprocess.py will generate a file named `grounding_results_corrected_{tier}_0.5_vqa_gt.json` (tier='val')

I used the script `compute_scores_updated_gqa.py` to get the detection scores (overlap and iou in terms of P, R, F1).
