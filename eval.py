import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import os
import matplotlib.pyplot as plt
from network import *
from datasets import TripletDataset,get_dists
from loss import batch_all_triplet_loss,_pairwise_distances


os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

data_dir = r'/HOME/data/celeb_a/processed/'
metadata_path = r"/media/rrtammyfs/labDatabase/celeb_a/identity_CelebA.txt"
vgg_data_dir = r'/HOME/data/vggface2/processed'

dataset = TripletDataset(4,2)
# dataset.load_celeb_a_dataset(data_dir,metadata_path)
dataset.load_vgg_face(vgg_data_dir)

# dataset.split_train_val_data()
dataset.gen_tf_dataset()

# Create the base model from the pre-trained model MobileNet V2
model = get_MobileNet_backbone_network(embedding_size=128, fc_layer_size=512, l2_norm=True, trainable_base=True)
model.load_weights('/HOME/FaceNet/checkpoints/vgg_train_celeb_val/20200710-133833/best_epoch_153_weights.hdf5')
model.summary()

a,b = next(iter(dataset.train_ds))
embeddings = model(a)
loss,frac =  batch_all_triplet_loss(b,embeddings,0.2)
