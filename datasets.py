import tensorflow as tf
import pandas as pd
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import os
AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_SIZE = 160
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

def process_image(file_path,label=None):
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  if label is not None:
    return img,label
  else:
      return img

def set_shapes(image, label=None):
    image.set_shape(IMG_SHAPE)
    if label is not None:
        label.set_shape([])
        return image, label
    else:
        return image

def augment_img(image,label,flip_lr=True,quality=False):
    if flip_lr:
        image = tf.image.random_flip_left_right(image)

    if quality:
        image = tf.image.random_jpeg_quality(image,80,100)
    return image, label


def k_gen(df,k):
    for label in np.random.permutation(df.label.unique()):
        same_label_samples = df[df.label == label].sample(n=k, replace=False)
        for i in range(k):
            image = same_label_samples.index[i]
            label = same_label_samples.label[i]
            yield image, label
    return

def tf_dataset_from_paths(paths,flip=False,bs=128):
    list_ds = tf.data.Dataset.from_tensor_slices(np.array(paths))
    list_ds = list_ds.map(process_image, num_parallel_calls=AUTOTUNE)
    list_ds = list_ds.map(set_shapes, num_parallel_calls=AUTOTUNE)
    list_ds = list_ds.batch(bs)
    if flip:
        list_ds = list_ds.map(tf.image.flip_left_right)
    list_ds = list_ds.prefetch(buffer_size=AUTOTUNE)
    return list_ds

class TripletDataset:
    def __init__(self,labels_per_batch,labels_per_image):
        self.train_data = None
        self.val_data = None
        self.labels_per_image = labels_per_image
        self.labels_per_batch = labels_per_batch
        self.train_ds = None
        self.val_ds = None
    def load_celeb_a_dataset(self,data_dir,csv_path,append=False,load_as_validation=False):
        self.labels_per_image = self.labels_per_image
        #Prepare and load data for celeb_a dataset
        data = pd.read_csv(csv_path,delimiter=' ',names=['filename','label'],index_col=0)
        images = glob(data_dir + '*.jpg')
        if len(images)==0:
            print("Error loading images! please check path!")
        #Filter the data to only existing images
        data = data.loc[[os.path.basename(x) for x in images]]
        #Set full path
        data.index = data_dir + data.index
        #Filter labels with less than min_labels_per_image
        labels_count = data['label'].value_counts()
        low_label_count = labels_count[labels_count < self.labels_per_image]
        data = data[~data['label'].isin(low_label_count.index)]
        #Catgorify the data
        if append:
            data.label = data.label.astype('category').cat.codes + 20000000
            self.train_data = self.train_data.append(data)
            self.train_data.label = self.train_data.label.astype('category').cat.codes
        else:
            data.label = data.label.astype('category').cat.codes
        if load_as_validation:
            self.val_data = data
        else:
            self.train_data = data
    def load_vgg_face(self,data_dir,append=False,load_as_validation=False):
        images = glob(os.path.join(data_dir,'*/*.jpg'))
        labels = [path.split('/')[-2] for path in images]
        data = pd.DataFrame(labels, index=images, columns=['label'])
        #Filter labels with less than min_labels_per_image
        labels_count = data['label'].value_counts()
        low_label_count = labels_count[labels_count < self.labels_per_image]
        data = data[~data['label'].isin(low_label_count.index)]
        if append:
            data.label = data.label.astype('category').cat.codes + 10000000
            self.train_data = self.train_data.append(data)
            self.train_data.label = self.train_data.label.astype('category').cat.codes
        else:
            data.label = data.label.astype('category').cat.codes

        if load_as_validation:
            self.val_data = data
        else:
            self.train_data = data

    def split_train_val_data(self,val_split=0.2):
        np.random.seed(1234)
        rand_labels = np.random.permutation(self.train_data.label.nunique())
        self.val_data = self.train_data[self.train_data.label.isin(rand_labels[:int(val_split*len(rand_labels))])]
        self.train_data = self.train_data[self.train_data.label.isin(rand_labels[int(val_split*len(rand_labels)):])]

    def gen_tf_dataset(self,prepare_for_training=True,repeat=False,augment=True):
        self.train_gen = partial(k_gen,df = self.train_data, k = self.labels_per_image)
        self.val_gen = partial(k_gen,df = self.val_data,k = self.labels_per_image)

        self.train_ds = tf.data.Dataset.from_generator(self.train_gen, (tf.string, tf.int32))
        self.val_ds = tf.data.Dataset.from_generator(self.val_gen, (tf.string, tf.int32))

        if prepare_for_training:
            self.train_ds = self.train_ds.map(process_image, num_parallel_calls=AUTOTUNE)
            self.train_ds = self.train_ds.map(set_shapes, num_parallel_calls=AUTOTUNE)
            self.train_ds = self.train_ds.batch(self.get_batch_size())
            if augment:
                self.train_ds = self.train_ds.map(augment_img, num_parallel_calls=AUTOTUNE)
            self.train_ds = self.train_ds.prefetch(buffer_size=AUTOTUNE)
            if repeat:
                self.train_ds = self.train_ds.repeat()

            self.val_ds = self.val_ds.map(process_image, num_parallel_calls=AUTOTUNE)
            self.val_ds = self.val_ds.map(set_shapes, num_parallel_calls=AUTOTUNE)
            self.val_ds = self.val_ds.batch(self.get_batch_size())
            self.val_ds = self.val_ds.prefetch(buffer_size=AUTOTUNE)
            if repeat:
                self.val_ds = self.val_ds.repeat()

    def get_batch_size(self):
        return self.labels_per_image*self.labels_per_batch

    def get_steps_per_epoch(self):
        if self.val_data is not None:
            return self.train_data.label.nunique()//self.labels_per_batch, self.val_data.label.nunique()//self.labels_per_batch
        else:
            return self.train_data.label.nunique() // self.labels_per_batch, 0
