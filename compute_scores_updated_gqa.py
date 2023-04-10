import argparse
import os
import json
import numpy as np
from tqdm import tqdm
import utils
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
exp_name = 'vqa_gt_layout'

parser = argparse.ArgumentParser()
parser.add_argument('--gt_dir', type=str, default="/data/visualreasoning/mac-with-data/mac-network-gqa-capsules")
parser.add_argument('--question_dir', type=str, default="/data/visualreasoning/mac-with-data/mac-network-gqa-capsules/questions1.2")
parser.add_argument('--pred_dir', type=str, default="/data/logs/gqa/gqa_spatial_baseline")
parser.add_argument('--save_dir', type=str, default="gqaExperiment-Spatial")
parser.add_argument('--result_folder', type=str, default="gqaExperiment-Spatial")
parser.add_argument('--attn_thresh', type=float, default=0.5)
parser.add_argument('--iou_thresh', type=float, default=0.5)
parser.add_argument('--split', type=str, default="val")
parser.add_argument('--test_iter', type=int, default=200000)
parser.add_argument('--labelType', type=str, default="all", choices=['question', 'answer', 'fullAnswer', 'all'])
args = parser.parse_args()

# iters = ['00000010', '00000050', '00000100', '00000500', '00001000',
#          '00010000', '00020000', '00200000']
iters = ['%08d'%args.test_iter]
attn_threshold = [args.attn_thresh]
iou_threshold = [args.iou_thresh]#np.arange(5, 10) * 0.1
results_dict = {k:{} for k in attn_threshold}
gt_dir = os.path.join(args.gt_dir, "gqa_{}_{}_question2bboxes.json".format(args.split, args.labelType))
questions = utils.load_json(os.path.join(args.question_dir, "{}_balanced_questions.json".format(args.split)))
# pred_dir = "/tmp/pycharm_project_103/exp_clevr_snmn/results/vqa_gt_layout/00160000/grounding_results_val.json"
#load gt bboxes
gt = utils.load_json(gt_dir)
results_log = ["Detection results log\n"]

def get_processed_scores(results):
    processed_results = {}
    f1_scores = []
    p_scores = []
    r_scores = []
    TP, FP, FN = [], [], []
    ts_ = []
    best_cases = []
    for qid, res in results.items():
        ts2scores = {}
        flag = False
        max_match = {'p': 0.0, 'r': 0.0, 'fscore': 0.0, 'TP': 0, 'FP':0, 'FN':0, 'ts':-1}
        for ts, score in res.items():
            tp, fp, fn = score['TP'], score['FP'], score['FN']
            prec = utils.precision(tp, fp)
            recall = utils.recall(tp, fn)
            f1_score = utils.f1_score(prec, recall)
            if f1_score > max_match['fscore']:
                max_match['fscore'] = f1_score
                max_match['p'] = prec
                max_match['r'] = recall
                max_match['score'] = score
                max_match['TP'] = tp
                max_match['FP'] = fp
                max_match['FN'] = fn
                max_match['ts'] = ts
                flag = False

        if not flag and res!={}:
            #if f1-score was always zero because tp was zero

            last_ts = list(res.keys())[-1]
            max_match['TP'] = res[last_ts]['TP']
            max_match['FP'] = res[last_ts]['FP']
            max_match['FN'] = res[last_ts]['FN']
            max_match['ts'] = last_ts

                # TP.append(tp)
                # FP.append(fp)
                # FN.append(fn)
            # ts2scores[ts] = {'p': prec, 'r': recall, 'f_score': f1_score, 'score': score}
        if max_match['fscore'] > 0.8:
            best_cases.append([qid, max_match['fscore']])
        f1_scores.append(max_match['fscore'])
        p_scores.append(max_match['p'])
        r_scores.append(max_match['r'])
        TP.append(max_match['TP'])
        FP.append(max_match['FP'])
        FN.append(max_match['FN'])
        ts_.append(max_match['ts'])
        processed_results[qid] = ts2scores
    precision = utils.precision(sum(TP), sum(FP))
    recall = utils.recall(sum(TP), sum(FN))
    f1_score = utils.f1_score(precision, recall)
    utils.save_json(best_cases,  './best_cases.json')
    return f1_score, precision, recall, processed_results, ts_

def get_processed_scores_by_family(results):
    questions = utils.load_json(os.path.join(args.question_dir, "CLEVR_{}_questions.json".format(args.split)))
    questions = utils.get_question2data_dict(questions)
    processed_results = { }
    f1_scores = []
    p_scores = []
    r_scores = []
    TP, FP, FN = [], [], []
    ts_ = []
    best_cases = []
    for qid, res in results.items():
        ts2scores = {}
        flag = False
        q_fam = questions[int(qid)]['program'][-1]['type']
        if q_fam not in processed_results.keys():
            processed_results[q_fam] = {}
            processed_results[q_fam]['TP']=[]
            processed_results[q_fam]['FP'] = []
            processed_results[q_fam]['FN'] = []
            processed_results[q_fam]['p'] = 0.0
            processed_results[q_fam]['r'] = 0.0
            processed_results[q_fam]['fscore'] = 0.0
        max_match = {'p': 0.0, 'r': 0.0, 'fscore': 0.0, 'TP': 0, 'FP':0, 'FN':0, 'ts':-1}
        for ts, score in res.items():
            tp, fp, fn = score['TP'], score['FP'], score['FN']
            prec = utils.precision(tp, fp)
            recall = utils.recall(tp, fn)
            f1_score = utils.f1_score(prec, recall)
            if f1_score > max_match['fscore']:
                max_match['fscore'] = f1_score
                max_match['p'] = prec
                max_match['r'] = recall
                max_match['score'] = score
                max_match['TP'] = tp
                max_match['FP'] = fp
                max_match['FN'] = fn
                max_match['ts'] = ts
                flag = True
                # TP.append(tp)
                # FP.append(fp)
                # FN.append(fn)

            # ts2scores[ts] = {'p': prec, 'r': recall, 'f_score': f1_score, 'score': score}
        if not flag and res!={}:
            #if f1-score was always zero because tp was zero

            last_ts = list(res.keys())[-1]
            max_match['TP'] = res[last_ts]['TP']
            max_match['FP'] = res[last_ts]['FP']
            max_match['FN'] = res[last_ts]['FN']
            max_match['ts'] = last_ts

        f1_scores.append(max_match['fscore'])
        if max_match['fscore'] > 0.7:
            best_cases.append([qid,max_match['fscore']])

        p_scores.append(max_match['p'])
        r_scores.append(max_match['r'])
        TP.append(max_match['TP'])
        FP.append(max_match['FP'])
        FN.append(max_match['FN'])
        ts_.append(max_match['ts'])
        processed_results[q_fam]['TP'].append(max_match['TP'])
        processed_results[q_fam]['FP'].append(max_match['FP'])
        processed_results[q_fam]['FN'].append(max_match['FN'])
        # processed_results[qid] = ts2scores
    q_fam_grouped = {'count':['count'],
                     'exist':['exist'],
                     'compare_number':['equal_integer', 'less_than', 'greater_than'],
                     'compare_attr':['equal_size', 'equal_color', 'equal_material', 'equal_shape'],
                     'query_attr':['query_size', 'query_color', 'query_material', 'query_shape']
                     }
    q_fam_grouped_res = {k:{'TP':0, 'FP':0, 'FN':0, 'fscore':0.0, 'p':0.0, 'r':0.0}
                         for k in q_fam_grouped.keys()}
    for q_fam in processed_results.keys():
        if q_fam in q_fam_grouped.keys(): #count and exist
            q_fam_grouped_res[q_fam]['TP'] = sum(processed_results[q_fam]['TP']) #utils.precision(sum(processed_results[q_fam]['TP']), sum(processed_results[q_fam]['FP']))
            q_fam_grouped_res[q_fam]['FP'] = sum(processed_results[q_fam]['FP']) #utils.recall(sum(processed_results[q_fam]['TP']), sum(processed_results[q_fam]['FN']))
            q_fam_grouped_res[q_fam]['FN'] = sum(processed_results[q_fam]['FN']) #utils.f1_score(processed_results[q_fam]['p'], processed_results[q_fam]['r'])

        elif q_fam in q_fam_grouped['compare_number']:
            q_fam_grouped_res['compare_number']['TP'] += sum(processed_results[q_fam]['TP'])
            q_fam_grouped_res['compare_number']['FP'] += sum(processed_results[q_fam]['FP'])
            q_fam_grouped_res['compare_number']['FN'] += sum(processed_results[q_fam]['FN'])
        elif q_fam in q_fam_grouped['compare_attr']:
            q_fam_grouped_res['compare_attr']['TP'] += sum(processed_results[q_fam]['TP'])
            q_fam_grouped_res['compare_attr']['FP'] += sum(processed_results[q_fam]['FP'])
            q_fam_grouped_res['compare_attr']['FN'] += sum(processed_results[q_fam]['FN'])
        elif q_fam in q_fam_grouped['query_attr']:
            q_fam_grouped_res['query_attr']['TP'] += sum(processed_results[q_fam]['TP'])
            q_fam_grouped_res['query_attr']['FP'] += sum(processed_results[q_fam]['FP'])
            q_fam_grouped_res['query_attr']['FN'] += sum(processed_results[q_fam]['FN'])

    for q_fam in q_fam_grouped_res.keys():
        q_fam_grouped_res[q_fam]['p'] = utils.precision(q_fam_grouped_res[q_fam]['TP'], q_fam_grouped_res[q_fam]['FP'])
        q_fam_grouped_res[q_fam]['r'] = utils.recall(q_fam_grouped_res[q_fam]['TP'], q_fam_grouped_res[q_fam]['FN'])
        q_fam_grouped_res[q_fam]['fscore'] = utils.f1_score(q_fam_grouped_res[q_fam]['p'], q_fam_grouped_res[q_fam]['r'])

        res_str = "%s:\t\tPrec: %.4f,\tRecall: %.4f,\tF1-score:%.4f" % (
            q_fam.upper(),
            q_fam_grouped_res[q_fam]['p'],
            q_fam_grouped_res[q_fam]['r'],
            q_fam_grouped_res[q_fam]['fscore'])
        results_log.append(res_str)

        print("%s:\t\tPrec: %.4f,\tRecall: %.4f,\tF1-score:%.4f" % (
            q_fam.upper(),
            q_fam_grouped_res[q_fam]['p'],
            q_fam_grouped_res[q_fam]['r'],
            q_fam_grouped_res[q_fam]['fscore'])
          )
    precision = None #utils.precision(sum(TP), sum(FP))
    recall = None #utils.recall(sum(TP), sum(FN))
    f1_score = None #utils.f1_score(precision, recall)
    return f1_score, precision, recall, processed_results, ts_

def get_processed_scores_by_template(results):
    questions = utils.load_json(os.path.join(args.question_dir, "CLEVR_{}_questions.json".format(args.split)))
    questions = utils.get_question2data_dict(questions)
    processed_results = { }
    f1_scores = []
    p_scores = []
    r_scores = []
    TP, FP, FN = [], [], []
    ts_ = []
    for qid, res in results.items():
        ts2scores = {}
        flag = False
        q_fam = questions[int(qid)]['template_filename'][:-5] #get rid of '.json'
        if q_fam not in processed_results.keys():
            processed_results[q_fam] = {}
            processed_results[q_fam]['TP']=[]
            processed_results[q_fam]['FP'] = []
            processed_results[q_fam]['FN'] = []
            processed_results[q_fam]['p'] = 0.0
            processed_results[q_fam]['r'] = 0.0
            processed_results[q_fam]['fscore'] = 0.0
        max_match = {'p': 0.0, 'r': 0.0, 'fscore': 0.0, 'TP': 0, 'FP':0, 'FN':0, 'ts':-1}
        for ts, score in res.items():
            tp, fp, fn = score['TP'], score['FP'], score['FN']
            prec = utils.precision(tp, fp)
            recall = utils.recall(tp, fn)
            f1_score = utils.f1_score(prec, recall)
            if f1_score > max_match['fscore']:
                max_match['fscore'] = f1_score
                max_match['p'] = prec
                max_match['r'] = recall
                max_match['score'] = score
                max_match['TP'] = tp
                max_match['FP'] = fp
                max_match['FN'] = fn
                max_match['ts'] = ts
                flag = True
                # TP.append(tp)
                # FP.append(fp)
                # FN.append(fn)

            # ts2scores[ts] = {'p': prec, 'r': recall, 'f_score': f1_score, 'score': score}
        if not flag and res!={}:
            #if f1-score was always zero because tp was zero

            last_ts = list(res.keys())[-1]
            max_match['TP'] = res[last_ts]['TP']
            max_match['FP'] = res[last_ts]['FP']
            max_match['FN'] = res[last_ts]['FN']
            max_match['ts'] = last_ts

        f1_scores.append(max_match['fscore'])
        p_scores.append(max_match['p'])
        r_scores.append(max_match['r'])
        TP.append(max_match['TP'])
        FP.append(max_match['FP'])
        FN.append(max_match['FN'])
        ts_.append(max_match['ts'])
        processed_results[q_fam]['TP'].append(max_match['TP'])
        processed_results[q_fam]['FP'].append(max_match['FP'])
        processed_results[q_fam]['FN'].append(max_match['FN'])
        # processed_results[qid] = ts2scores
    q_fam_grouped = set([v['template_filename'][:-5] for k, v in questions.items()])
    q_fam_grouped_res = {k:{'TP':0, 'FP':0, 'FN':0, 'fscore':0.0, 'p':0.0, 'r':0.0}
                         for k in q_fam_grouped}
    for q_fam in processed_results.keys():
        if q_fam in q_fam_grouped:
            q_fam_grouped_res[q_fam]['TP'] = sum(processed_results[q_fam]['TP']) #utils.precision(sum(processed_results[q_fam]['TP']), sum(processed_results[q_fam]['FP']))
            q_fam_grouped_res[q_fam]['FP'] = sum(processed_results[q_fam]['FP']) #utils.recall(sum(processed_results[q_fam]['TP']), sum(processed_results[q_fam]['FN']))
            q_fam_grouped_res[q_fam]['FN'] = sum(processed_results[q_fam]['FN']) #utils.f1_score(processed_results[q_fam]['p'], processed_results[q_fam]['r'])


    for q_fam in q_fam_grouped_res.keys():
        q_fam_grouped_res[q_fam]['p'] = utils.precision(q_fam_grouped_res[q_fam]['TP'], q_fam_grouped_res[q_fam]['FP'])
        q_fam_grouped_res[q_fam]['r'] = utils.recall(q_fam_grouped_res[q_fam]['TP'], q_fam_grouped_res[q_fam]['FN'])
        q_fam_grouped_res[q_fam]['fscore'] = utils.f1_score(q_fam_grouped_res[q_fam]['p'], q_fam_grouped_res[q_fam]['r'])

        res_str = "%s:\t\tPrec: %.4f,\tRecall: %.4f,\tF1-score:%.4f"%(
            q_fam.upper(),
            q_fam_grouped_res[q_fam]['p'],
            q_fam_grouped_res[q_fam]['r'],
            q_fam_grouped_res[q_fam]['fscore'])

        results_log.append(res_str)

        print("%s:\t\tPrec: %.4f,\tRecall: %.4f,\tF1-score:%.4f"%(
            q_fam.upper(),
            q_fam_grouped_res[q_fam]['p'],
            q_fam_grouped_res[q_fam]['r'],
            q_fam_grouped_res[q_fam]['fscore'])
        )

    precision = None #utils.precision(sum(TP), sum(FP))
    recall = None #utils.recall(sum(TP), sum(FN))
    f1_score = None #utils.f1_score(precision, recall)
    return f1_score, precision, recall, processed_results, ts_


for it in iters:
    for atn_th in attn_threshold:
        print("loading pred file...")

        # pred_dir = "/media/data/snmn/exp_clevr_snmn/results/%s/%s/grounding_results_corrected_val_%s_vqa_gt.json"%(exp_name, it, str(atn_th))
        # pred_dir = "/data/logs/masked_vcaps/grounding_results_corrected_train_val_0.5_vqa_gt.json"
        # load predictions
        preds = utils.load_json(os.path.join(args.pred_dir, args.result_folder,
                                              'grounding_results_corrected_{}_0.5_vqa_gt.json'.format(args.split)))
        # preds = utils.get_results_dict(preds)
        print("Processing attn threshold of {}, \nlen of gt file: {}, len of pred file:{}".
              format(atn_th, len(gt.items()), len(preds)))
        for iou_th in iou_threshold:
        #     count = 23826
        #
            results_dict[atn_th][iou_th] = {'mean_fscore':0, 'mean_p':0, 'mean_r':0}
            results_interarea, results_iou = {}, {}
            vis_num = 0
            for qid, v in tqdm(gt.items()):  # iterate over questions
                ts2results_interarea = {}
                ts2results_iou = {}
                try:
                    #try catch block to deal with subset of data
                    pred_bboxes = preds["{0:08d}".format(int(qid))]
                    # pred_bboxes = preds[int(qid)]["{0:08d}".format(int(qid))]
                except:
                    continue

                if v == {}:
                    # count += 1
                    # compute TP, FP and FN
                    for ts, pboxes in pred_bboxes.items():
                        num_pred = len(pboxes)
                        num_gt_boxes = len(v.items())
                        ts2results_interarea[ts] = {"TP": 0,
                                          "FP": num_pred,  #all predictions are false detections
                                          "FN": 0}
                        ts2results_iou[ts] = {"TP": 0,
                                                    "FP": num_pred,  # all predictions are false detections
                                                    "FN": 0}
                    results_interarea[qid] = ts2results_interarea
                    results_iou[qid] = ts2results_iou
                    continue

                ts2TP = {k:0 for k in pred_bboxes.keys()}
                ts2TP2 = {k: 0 for k in pred_bboxes.keys()}
                # im = np.array(Image.open('/media/data/CLEVR_v1.0/images/train_val/{}'.format(questions['questions'][int(qid)]['image_filename'])),
                #               dtype=np.uint8)

                # uncomment to visualize for debugging
                # fig, axes = plt.subplots(nrows=len(pred_bboxes.items()), ncols=1, squeeze=False)
                #
                # axes = axes.ravel()
        #
        #         # todo:put an assert to make sure qid is same for gt and pred
                for obj_id, bbox in v.items():
                    gt_bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        #
                    for ts, pboxes in pred_bboxes.items():
                        # axes[int(ts)].imshow(im)
                        # rect_g = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='g',
                        #                            facecolor='none')
                        # axes[int(ts)].add_patch(rect_g)
        #
                        for pbox in pboxes:
                            pbox_ = [pbox['bbox'][1], pbox['bbox'][0],
                                     pbox['bbox'][3],
                                     pbox['bbox'][2]]

        #                     # uncomment to visualize for debugging
        #                     rect = patches.Rectangle((pbox['bbox'][1], pbox['bbox'][0]),
        #                                              pbox['bbox'][3]-pbox['bbox'][1], pbox['bbox'][2]-pbox['bbox'][0], linewidth=1, edgecolor='r',
        #                                              facecolor='none')
        #                     axes[int(ts)].add_patch(rect)
        #
                            if utils.intersects(gt_bbox, pbox_):
                                iou, interArea = utils.bb_intersection_over_union(gt_bbox, pbox_)
                                # print(iou)
                                if interArea >= iou_th:
                                    ts2TP[ts] += 1
                                if iou >= iou_th:
                                    ts2TP2[ts] += 1
                                # print(intersects(gt_bbox, pbox_))

                #compute TP, FP and FN
                for ts, pboxes in pred_bboxes.items():
                    num_pred = len(pboxes)
                    num_gt_boxes = len(v.items())
                    FP_per_ts = max(num_pred - ts2TP[ts], 0) #for cases where 1 big predicted bbox covers multiple gt boxes
                    FN_per_ts = num_gt_boxes - ts2TP[ts]
                    ts2results_interarea[ts] = {"TP": ts2TP[ts],
                                       "FP": FP_per_ts,
                                       "FN": FN_per_ts}
                    text_ = "Intersection over detection: TP:{}, FP:{}, FN:{}\n".format(ts2TP[ts], FP_per_ts, FN_per_ts)
                    FP_per_ts = max(num_pred - ts2TP2[ts], 0) #for cases where 1 big predicted bbox covers multiple gt boxes
                    FN_per_ts = num_gt_boxes - ts2TP2[ts]
                    ts2results_iou[ts] = {"TP": ts2TP2[ts],
                                                "FP": FP_per_ts,
                                                "FN": FN_per_ts}
                    # axes[int(ts)].axis('off')
                    # axes[int(ts)].text(0.5, -0.1, text_ + "IOU: TP:{}, FP:{}, FN:{}".format(ts2TP2[ts], FP_per_ts, FN_per_ts))

                results_interarea[qid] = ts2results_interarea
                results_iou[qid] = ts2results_iou
                # item = questions['questions'][int(qid)]

                # uncomment to visualize for debugging
                # q = item['question'].split(" ")
                # # axes[-1].axis('off')
                # plt.title( " ".join(q[:len(q) // 2]) + "\n" + " ".join(q[len(q) // 2:]) + "\n answer:" + str(item['answer']))
                # plt.show()
                # if vis_num < 100:
                #     plt.savefig("/media/data/snmn/exp_clevr_snmn/results/vqa_gt_layout/00160000/scores/bbox{0:08d}.png".format(int(qid)), bbox_inches='tight')
                #     vis_num +=1

            # utils.save_json([results_interarea, results_iou], "/media/data/snmn/exp_clevr_snmn/results/%s/%s/iou_based_scores_corr_%s_%s.json" % (exp_name, it, atn_th, iou_th))

            # results = utils.load_json("/media/data/snmn/exp_clevr_snmn/results/%s/%s/iou_based_scores_corr_%s_%s.json" % (exp_name, it, atn_th, iou_th))
            results = [results_interarea, results_iou]
            f1_score, precision, recall, processed_results, ts_iod = get_processed_scores(results[0])

            res_str = "\nExperiment: %s, overlap, it:%s, F1:%.4f, P:%.4f, R:%.4f"%(exp_name, it, f1_score, precision, recall)
            results_log.append(res_str)

            print("\nExperiment: %s, overlap, it:%s, F1:%.4f, P:%.4f, R:%.4f"%(exp_name, it, f1_score, precision, recall))

            results_dict[atn_th][iou_th]['mean_fscore'] = [f1_score]  # [sum(f1_scores)/len(questions['questions']),
            # sum(f1_scores)/(len(questions['questions'])-count)]
            results_dict[atn_th][iou_th]['mean_p'] = [precision]  # [sum(p_scores) / len(questions['questions']),
            # sum(p_scores) / (len(questions['questions']) - count)]
            results_dict[atn_th][iou_th]['mean_r'] = [recall]  # [sum(r_scores) / len(questions['questions']),
            # sum(r_scores) / (len(questions['questions']) - count)]
            results_dict[atn_th][iou_th]['processed_results'] = [processed_results]

            f1_score, precision, recall, processed_results, ts_iou = get_processed_scores(results[1])

            res_str = "\nExperiment: %s, iou, it:%s, F1:%.4f, P:%.4f, R:%.4f"%(exp_name, it, f1_score, precision, recall)
            results_log.append(res_str)

            print("\nExperiment: %s, iou, it:%s, F1:%.4f, P:%.4f, R:%.4f"%(exp_name, it, f1_score, precision, recall))
            results_dict[atn_th][iou_th]['mean_fscore'].append(f1_score) #[sum(f1_scores)/len(questions['questions']),
                                                   #sum(f1_scores)/(len(questions['questions'])-count)]
            results_dict[atn_th][iou_th]['mean_p'].append(precision)#[sum(p_scores) / len(questions['questions']),
                                                   #sum(p_scores) / (len(questions['questions']) - count)]
            results_dict[atn_th][iou_th]['mean_r'].append(recall)#[sum(r_scores) / len(questions['questions']),
                                                   #sum(r_scores) / (len(questions['questions']) - count)]
            results_dict[atn_th][iou_th]['processed_results'].append(processed_results)

            results_log.append('\n\nOverlap by question family:')
            print('\n\nOverlap by question family:')
            f1_score, precision, recall, processed_results, ts_iod = get_processed_scores_by_family(results[0])

            results_log.append('\n\nIOU by question family:')
            print('\n\nIOU by question family:')
            f1_score, precision, recall, processed_results, ts_iod = get_processed_scores_by_family(results[1])

            results_log.append('\n\nOverlap by question template:\n')
            print('\n\nOverlap by question template:\n')
            f1_score, precision, recall, processed_results, ts_iod = get_processed_scores_by_template(results[0])

            results_log.append('\n\nIOU by question template:\n')
            print('\n\nIOU by question template:\n')
            f1_score, precision, recall, processed_results, ts_iod = get_processed_scores_by_template(results[1])

    save_path = os.path.join(args.save_dir, args.result_folder)
    os.makedirs(save_path, exist_ok=True)
    print("saving results to %s"%save_path)
    utils.save_json(results_dict, os.path.join(save_path,'results_dict.json'))
    with open(os.path.join(save_path, "detection_summary.log"), "a") as f:
        f.write("\n".join(results_log) + "\n")
    # utils.save_json(results_dict, "/media/data/snmn/exp_clevr_snmn/results/%s/%s/results_dict.json"%(exp_name, it))
    # utils.save_json({"ts_iou":ts_iou, "ts_iod":ts_iod}, "/media/data/snmn/exp_clevr_snmn/results/%s/%s/ts_gt_layout.json"%(exp_name, it))
    # print("random print statement to put breakpoint...:D")
