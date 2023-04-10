import os
import numpy as np
import matplotlib;
from matplotlib import patches
import matplotlib.pyplot as plt
import skimage.io
from skimage.feature import blob_doh
from skimage.color import rgb2gray, gray2rgb
import skimage.measure as skmeas
import utils

root = "../CLEVR_v1.0/images/val"

def attention_interpolation(im, att):
    # softmax = _att_softmax(att)
    att = att.squeeze()
    att_reshaped = skimage.transform.resize(att, im.shape[:2], order=3)
    # normalize the attention
    # make sure the 255 alpha channel is at least 3x uniform attention
    att_reshaped /= np.max(att_reshaped)#, 1. / att.size)
    att_reshaped = att_reshaped[..., np.newaxis]
    object_labels = skmeas.label(rgb2gray(att_reshaped.squeeze()) >= 0.5)
    # make the attention area brighter than the rest of the area
    try:
        vis_im = att_reshaped * im + (1-att_reshaped) * im * .45
    except:
        im = gray2rgb(im)
        vis_im = att_reshaped * im + (1 - att_reshaped) * im * .45
    vis_im = vis_im.astype(im.dtype)

    # blobs_doh = blob_doh(rgb2gray(att_reshaped.squeeze())>=0.5, max_sigma=30)
    some_props = skmeas.regionprops(object_labels)
    bboxes = [{'bbox':one_prop['bbox'], 'centroid':one_prop['centroid']} for one_prop in some_props]
    # blobs_list = [blobs_doh, blobs_doh, blobs_doh]
    # colors = ['yellow', 'lime', 'red']
    # titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
    #           'Determinant of Hessian']
    # sequence = zip(blobs_list, colors, titles)
    #
    # fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
    # ax = axes.ravel()
    #
    # for idx, (blobs, color, title) in enumerate(sequence):
    #     ax[idx].set_title(title)
    #     ax[idx].imshow(rgb2gray(vis_im))
    #     for blob in blobs:
    #         y, x, r = blob
    #         c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
    #         ax[idx].add_patch(c)
    #     ax[idx].set_axis_off()
    #
    # plt.tight_layout()
    # plt.show()
    # plt.savefig('./new5.png', dpi=100)
    return vis_im, bboxes

def visualize(max_step, im_path, attns, question=None, answer=None, h=320, w=480):
    # fig, axes = plt.subplots(nrows=max_step, ncols=1, squeeze=False, figsize=(4, 20))
    # axes = axes.ravel()
    # attns = attns.squeeze().detach().cpu().numpy()
    # attns = attns[:, :-1]

    t2detections = {}
    img = skimage.io.imread(im_path)
    img = skimage.transform.resize(img, (h, w))
    for i in range(max_step):
        img_with_att, boxes_detected = attention_interpolation(img, attns[i])
        # axes[i].imshow(img_with_att)
        # for bb in boxes_detected:
        #     bbox = bb['bbox']
        #     rect_g = patches.Rectangle((bbox[1], bbox[0]), bbox[3]-bbox[1], bbox[2]-bbox[0],
        #                                 linewidth=0.5, edgecolor='r', facecolor='none')
        #
        #     axes[i].add_patch(rect_g)
        #     axes[i].axis('off')
        t2detections[i] = boxes_detected
    # q_len = len(question)
    # plt.title(question[:q_len//2]+"\n"+question[q_len//2:]+str(answer), {'fontsize':10})
    # plt.show()
    return t2detections

def visualizeSingleMap(im_path, attn, h=320, w=480):
    img = skimage.io.imread(im_path)
    img = skimage.transform.resize(img, (h, w))
    img_with_att, boxes_detected = attention_interpolation(img, attn)
    #todo: add bbox plotting code to img_with_att and save
    pass


def visualize_batch(batch, dic):
    q2detect_list = []

    for i in range(batch['b_size']):
        im_path = os.path.join(root, batch['imgfile'][i])
        q_len = batch['q_len'][i]
        question = utils.token2words(batch['question'][i][:q_len], dict=dic['word_dic'])
        answer = utils.token2words([batch['answer'][i]], dict=dic['answer_dic'])
        qid = batch['qid'][i]
        t2detections = visualize(batch['max_step'], im_path, batch['attns'][i], question, answer)
        q2detect_list.append({qid:t2detections})
    return q2detect_list
