import json
import gc
import os
import argparse
import numpy as np
import utils
from tqdm import tqdm
from vis import visualize, attention_interpolation

eps = 1e-15

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default="/data/logs/gqa/gqa_spatial_baseline", help="dir path to prediction files.")
parser.add_argument('--out_dir', type=str, default="/data/logs/gqa/gqa_spatial_baseline", help="dir path to save output files")
parser.add_argument('--data_dir', type=str, default="/data/visualreasoning/mac-with-data/mac-network-gqa-capsules", help='path to data needed to process output')
parser.add_argument('--exp_name', type=str, default="gqaExperiment-Spatial")
parser.add_argument('--featureType', type=str, default="spatial", help="'spatial' for spatial features, 'object'")
parser.add_argument('--tier',           default = "val",                     type = str,    help = "Tier, e.g. train, val")
parser.add_argument('--scenes',         default="{tier}_sceneGraphs.json",   type = str,    help = "Scene graphs file name format.")
parser.add_argument('--predictions',    default="{tier}_predictions.json",   type = str,    help = "Answers file name format.")
parser.add_argument('--attentions',     default="{tier}_attentions.json",    type = str,    help = "Attentions file name format.")
parser.add_argument('--attnStep',     default="last",    type = str,   choices=['mean', 'last', 'all'], help = "From which reasoing step, want to save attention")
parser.add_argument('--objdata_dir', type=str, default="/data/visualreasoning/mac-with-data/mac-network-gqa-capsules/mac-network-w-capsules/data/")
parser.add_argument('--dataset', type=str, default="gqa", choices=['gqa', 'clevr'])
parser.add_argument('--chunk_size', type=int, default=20000, help='chunk size to process prediction bboxes into chunks--useful to avoid OOM')
parser.add_argument('--grounding',      action="store_true",        help = "True to compute grounding score (If model uses attention).")

args = parser.parse_args()

if args.dataset == 'gqa':
    h,w = 7, 7
elif args.dataset == 'clevr':
    h,w = 14, 14
else:
    print('dataset not supported.')

def loadFile(name):
    # load standard json file
    if os.path.isfile(name):
        with open(name) as file:
            data = json.load(file)
    # load file chunks if too big
    elif os.path.isdir(name.split(".")[0]):
        data = {}
        chunks = glob.glob('{dir}/{dir}_*.{ext}'.format(dir = name.split(".")[0], ext = name.split(".")[1]))
        for chunk in chunks:
            with open(chunk) as file:
                data.update(json.load(file))
    else:
        raise Exception("Can't find {}".format(name))
    return data

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# Load scene graphs
print("Loading scene graphs...")
scenes = loadFile(os.path.join(args.data_dir, 'sceneGraphs',args.scenes.format(tier = args.tier)))

in_path = os.path.join(args.input_dir, args.exp_name)
save_path = os.path.join(args.out_dir, args.exp_name)

os.makedirs(os.path.join(save_path, args.attnStep), exist_ok=True)

pred_file = args.tier+'Predictions-'+args.exp_name+'.json'
print('Loading predictions from {} at path {}...'.format(pred_file, in_path))
preds = utils.load_json(os.path.join(in_path, pred_file))
print('loaded preditions successfully!')

print('processing prediction file for evaluation script...')

def getRegion(obj, img_w, img_h):
    x0 = float(obj["x"]) / img_w
    y0 = float(obj["y"]) / img_h
    x1 = float(obj["x"] + obj["w"]) / img_w
    y1 = float(obj["y"] + obj["h"]) / img_h
    return x0, y0, x1, y1


def get_ts2detections_gqa(pred):
    img_id = pred['imageId']['id']
    img_w = scenes[img_id]['width']
    img_h = scenes[img_id]['height']
    qid = pred['questionId']
    question = pred['questionStr']
    answer = pred["answer"]
    im_path = os.path.join(args.data_dir, 'images', img_id + '.jpg')
    attns = pred['attentions']['kb']
    max_step = len(attns)
    attns = [np.array(attn).reshape((7, 7)) for attn in attns]
    attns = np.stack(attns, axis=0)
    t2detections = visualize(max_step=max_step, im_path=im_path, attns=attns, question=question, answer=answer, h=img_h,
                             w=img_w)
    return qid, t2detections


def get_grounding_result_gqa(pred_lst):
    # global qid2t2detections
    qid2t2detections = {}
    results = [get_ts2detections_gqa(pred) for pred in tqdm(pred_lst)] #list of tuples (qid, t2detection)
    qid2t2detections = {tup[0]:tup[1] for tup in results}
    # for pred in tqdm(preds):
    #     get_ts2detections(pred)
    # print(len(qid2t2detections.items()))
    # utils.save_json(qid2t2detections, os.path.join(save_path,
    #                                                'grounding_results_corrected_{tier}_0.5_vqa_gt_{chunk}.json'.format(
    #                                                    tier=args.tier, chunk=chunk)))
    # qid2t2detections = []
    # results = []
    return qid2t2detections

def get_ts2detections_clevr(pred):
    img_id = pred['imageId']['id']
    img_w = 480
    img_h = 320
    qid = pred['questionId']
    question = pred['questionStr']
    answer = pred["answer"]
    im_path = os.path.join(args.data_dir, 'images', args.tier, img_id + '.png')
    attns = pred['attentions']['kb']
    max_step = len(attns)
    attns = [np.array(attn).reshape((14, 14)) for attn in attns]
    attns = np.stack(attns, axis=0)
    t2detections = visualize(max_step=max_step, im_path=im_path, attns=attns, question=question, answer=answer, h=img_h,
                             w=img_w)
    return qid, t2detections

def get_grounding_result_clevr(pred_lst):
    # global qid2t2detections
    qid2t2detections = {}
    results = [get_ts2detections_clevr(pred) for pred in tqdm(pred_lst)] #list of tuples (qid, t2detection)
    qid2t2detections = {tup[0]:tup[1] for tup in results}
    # for pred in tqdm(preds):
    #     get_ts2detections(pred)
    # print(len(qid2t2detections.items()))
    # utils.save_json(qid2t2detections, os.path.join(save_path,
    #                                                'grounding_results_corrected_{tier}_0.5_vqa_gt_{chunk}.json'.format(
    #                                                    tier=args.tier, chunk=chunk)))
    # qid2t2detections = []
    # results = []
    return qid2t2detections


if args.featureType == 'spatial':
    spatial_attn = []
    predictions = []
    if args.attnStep in ['mean', 'last']:
        for pred in tqdm(preds):

            if args.grounding:
                if args.attnStep == 'last':
                    attn = np.array(pred['attentions']['kb'][-1]).reshape((h, w)).tolist()
                elif args.attnStep == 'mean':
                    # print(np.shape(np.mean(np.array(pred['attentions']['kb']), axis=0)))
                    attn = np.mean(np.array(pred['attentions']['kb']), axis=0).reshape((7, 7)).tolist()

                spatial_attn.append(
                    {
                        'questionId': pred['questionId'],
                        'attention': attn
                    }
                )
            predictions.append(
                {
                    'questionId': pred['questionId'],
                    'prediction': pred['prediction']
                }
            )
        utils.save_json(spatial_attn, os.path.join(save_path, args.attnStep, args.attentions.format(tier=args.tier)))
        utils.save_json(predictions, os.path.join(save_path, args.attnStep, args.predictions.format(tier=args.tier)))
    else:
        if args.attnStep == 'all':
            num_steps = len(preds[0]['attentions']['kb'])
            for t in tqdm(range(num_steps)):
                spatial_attn = []
                predictions = []

                for pred in preds:
                    if args.grounding:
                        spatial_attn.append(
                            {
                                'questionId': pred['questionId'],
                                'attention':  np.array(pred['attentions']['kb'][t]).reshape((h,w)).tolist()
                            }
                        )
                    predictions.append(
                        {
                            'questionId': pred['questionId'],
                            'prediction': pred['prediction']
                        }
                    )
                os.makedirs(os.path.join(save_path, args.attnStep, t), exist_ok=True)
                utils.save_json(spatial_attn, os.path.join(save_path, args.attnStep, t, args.attentions.format(tier=args.tier)))
                utils.save_json(predictions, os.path.join(save_path, args.attnStep, t, args.predictions.format(tier=args.tier)))
                del spatial_attn, predictions
        else:
            print('unknown argument: {}'.format(args.attnStep))
            exit()

    ##################### post process for object detections #######################
    print('done with processing files and saved!\n ')

    if args.grounding:
        print('Now get attention maps for object detections...')
        if args.dataset == 'gqa':
            #read questions file to get img width and height
            #divide preds file in chunks
            preds_list = list(chunks(preds, args.chunk_size))
            qid2t2detections_full = {}
            for chunk, pred_lst in enumerate(preds_list):
                qid2t2detections = get_grounding_result_gqa(pred_lst)
                qid2t2detections_full.update(qid2t2detections)
                print('processed grouding results for chunk {}/{}'.format(chunk, len(preds_list)))
            print('saving grounding results..')
            utils.save_json(qid2t2detections_full, os.path.join(save_path,
                                                           'grounding_results_corrected_{tier}_0.5_vqa_gt.json'.format(
                                                               tier=args.tier)))
            print('saved grouding results successfully to {}!'.format(save_path))
        elif args.dataset == 'clevr':
            preds_list = list(chunks(preds, args.chunk_size))
            qid2t2detections_full = {}
            for chunk, pred_lst in enumerate(preds_list):
                qid2t2detections = get_grounding_result_clevr(pred_lst)
                qid2t2detections_full.update(qid2t2detections)
                print('processed grouding results for chunk {}/{}'.format(chunk, len(preds_list)))
            print('saving grounding results..')
            utils.save_json(qid2t2detections_full, os.path.join(save_path,
                                                                'grounding_results_corrected_{tier}_0.5_vqa_gt.json'.format(
                                                                    tier=args.tier)))
            print('saved grouding results successfully to {}!'.format(save_path))

        else:
            print("ERROR: invalid dataset. Select from allowed choices: 'gqa' or 'clevr'. ")

elif args.featureType == 'object':
    print('reading obj info..')
    #read obj info file
    obj_info_dir = os.path.join(args.objdata_dir, 'gqa_objects_merged_info_with_bboxes.json')
    if os.path.isfile(obj_info_dir):
        obj_info = utils.load_json(obj_info_dir)
    else:
        print('file not found, loading obj feat and info file to obtain bboxes...')
        import h5py
        #read obj feat file and obj info file and merge boxes
        h = h5py.File(
            os.path.join(args.objdata_dir, 'gqa_objects.h5'),
            'r')
        bboxes = h['bboxes']
        obj_info = utils.load_json(
            os.path.join(args.objdata_dir, 'gqa_objects_merged_info.json'))

        for key, value in tqdm(obj_info.items()):
            obj_info[key]['bboxes'] = bboxes[obj_info[key]['index']].tolist()
        utils.save_json(obj_info,
                        os.path.join(args.objdata_dir, 'gqa_objects_merged_info_with_bboxes.json')
                        )
        print('merged and saved obj info file.')

    if args.attnStep in ['mean', 'last']:
        spatial_attn = []
        predictions = []
        for pred in tqdm(preds):


            if args.grounding:
                if args.attnStep == 'last':
                    attn = np.array(pred['attentions']['kb'][-1][:pred['objectsNum']]).tolist()
                elif args.attnStep == 'mean':
                    attn = np.mean(np.array(pred['attentions']['kb'][:][:pred['objectsNum']]), axis=-1).tolist()
                pred_bboxes = obj_info[pred['imageId']['id']]['bboxes']
                width, height = obj_info[pred['imageId']['id']]['width'], obj_info[pred['imageId']['id']]['height']
                pboxes_normalized = []
                for pbox in pred_bboxes:
                    obj = {
                        'x': pbox[0],
                        'y': pbox[1],
                        'w': pbox[2],
                        'h': pbox[3]
                    }
                    pboxes_normalized.append(getRegion(obj, width, height))

                    # pbox = [pb+eps for pb in pbox if pb==0.0]
                attn_with_boxes = [[*bb, atn] for bb, atn in zip(pboxes_normalized, attn)]
                spatial_attn.append(
                    {
                        'questionId': pred['questionId'],
                        'attention': attn_with_boxes
                    }
                )
            predictions.append(
                {
                    'questionId': pred['questionId'],
                    'prediction': pred['prediction']
                }
            )

        utils.save_json(spatial_attn, os.path.join(save_path, args.attnStep, args.attentions.format(tier=args.tier)))
        utils.save_json(predictions, os.path.join(save_path, args.attnStep, args.predictions.format(tier=args.tier)))
    else:
        if args.attnStep == 'all':
            num_steps = len(preds[0]['attentions']['kb'])
            for t in range(num_steps):
                spatial_attn = []
                predictions = []

                for pred in preds:
                    attn = np.array(pred['attentions']['kb'][t][:pred['objectsNum']]).tolist()
                    pred_bboxes = obj_info[pred['imageId']['id']]['bboxes']
                    #normalize bounding boxes
                    width, height = obj_info[pred['imageId']['id']]['width'], obj_info[pred['imageId']['id']]['height']
                    for pbox in pred_bboxes:
                        pbox[2] = pbox[0] + pbox[2]  # x+w
                        pbox[3] = pbox[1] + pbox[3]  # y+h
                        pbox[0] /= width  # x0
                        pbox[2] /= width  # x1
                        pbox[1] /= height  # y0
                        pbox[3] /= height  # y1
                        # pbox = [pb + eps for pb in pbox if pb == 0.0]
                    attn_with_boxes = [[*bb, atn] for bb, atn in zip(pred_bboxes, attn)]


                    spatial_attn.append(
                        {
                            'questionId': pred['questionId'],
                            'attention':  attn_with_boxes
                        }
                    )
                    predictions.append(
                        {
                            'questionId': pred['questionId'],
                            'prediction': pred['prediction']
                        }
                    )
                os.makedirs(os.path.join(save_path, args.attnStep, t), exist_ok=True)
                utils.save_json(spatial_attn, os.path.join(save_path, args.attnStep, t, args.attentions.format(tier=args.tier)))
                utils.save_json(predictions, os.path.join(save_path, args.attnStep, t, args.predictions.format(tier=args.tier)))
                del spatial_attn, predictions
        else:
            print('unknown argument: {}'.format(args.attnStep))
            exit()






