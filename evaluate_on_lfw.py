import os
import numpy as np
import tensorflow as tf
from scipy import interpolate
from network import *
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from datasets import tf_dataset_from_paths
from evaluation_metrics import evaluate
from sklearn import metrics
from scipy.optimize import brentq

LFW_PAIRS_PATH = r'/HOME/data/lfw/pairs.txt'
LFW_DIR = r'/HOME/data/lfw/processed'

def get_paths(lfw_dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
            issame = True
        elif len(pair) == 4:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
            path_list += (path0, path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_list, issame_list


def add_extension(path):
    if os.path.exists(path + '.jpg'):
        return path + '.jpg'
    elif os.path.exists(path + '.png'):
        return path + '.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)


def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)

def evaluate_LFW(model,embedding_size,use_flipped_images = False,N_folds=5,distance_metric=1,verbose=1):
    pairs = read_pairs(os.path.expanduser(LFW_PAIRS_PATH))
    paths, actual_issame = get_paths(os.path.expanduser(LFW_DIR), pairs)
    ds = tf_dataset_from_paths(paths, flip=False)
    embeddings = np.zeros([len(paths), embedding_size])
    j = 0
    if verbose>=2:
        print("Feed forward all pairs")
    for batch in ds:
        batch_embeddings = model(batch).numpy()
        embeddings[j:j + len(batch)] = batch_embeddings
        j += len(batch)
    if use_flipped_images:
        if verbose >= 2:
            print("Feed forward all pairs - flipped")
        flip_ds = tf_dataset_from_paths(paths, flip=True)
        flip_embeddings = np.zeros([len(paths), embedding_size])
        j = 0
        for batch in flip_ds:
            batch_embeddings = model(batch).numpy()
            flip_embeddings[j:j + len(batch)] = batch_embeddings
            j += len(batch)

        full_embeddings = np.zeros((len(paths), embedding_size * 2))
        full_embeddings[:, :embedding_size] = embeddings
        full_embeddings[:, embedding_size:] = flip_embeddings
    if verbose>=2:
        print("Calculating metrics")

    if use_flipped_images:
        tpr, fpr, accuracy, val, val_std, far,best_thresholds = evaluate((embeddings + flip_embeddings) / 2, actual_issame,
                                                         nrof_folds=N_folds,
                                                         distance_metric=distance_metric)
    else:
        tpr, fpr, accuracy, val, val_std, far,best_thresholds = evaluate(embeddings, actual_issame, nrof_folds=N_folds,
                                                         distance_metric=distance_metric)
    if verbose:
        print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
        print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
        print('threshold : %2.5f+-%2.5f' % (np.mean(best_thresholds), np.std(best_thresholds)))
        auc = metrics.auc(fpr, tpr)
        print('Area Under Curve (AUC): %1.3f' % auc)
        eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
        print('Equal Error Rate (EER): %1.3f' % eer)
    return accuracy


if __name__ == '__main__':
    embedding_size = 128
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'

    # Create the base model from the pre-trained model MobileNet V2
    model = get_MobileNet_backbone_network(embedding_size=128, fc_layer_size=512, l2_norm=True, trainable_base=True)
    model.load_weights('/HOME/FaceNet/checkpoints/vgg_train_celeb_val/20200710-133833/best_epoch_153_weights.hdf5')
    # model.load_weights('/HOME/FaceNet/checkpoints/celeb_finetune/20200711-094426/best_epoch_19_weights.hdf5')
    # Run forward pass to calculate embeddings
    # Read the file containing the pairs used for testing
    evaluate_LFW(model,embedding_size,use_flipped_images=False,distance_metric=1,verbose=2,N_folds=10)