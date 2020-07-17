
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
from glob import glob
import os
from tqdm.auto import tqdm

raw_data_dir = r'/media/rrtammyfs/labDatabase/celeb_a/faces'
processed_data_dir = r'/media/rrtammyfs/labDatabase/celeb_a/faces/processed/'
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'


list_imgs = glob(os.path.join(raw_data_dir,"*/*.jpg"))
mtcnn = MTCNN(margin=10, select_largest=True, post_process=False)#, device='cuda:0')
for img_path in tqdm(list_imgs):
  img = plt.imread(img_path)
  face = mtcnn(img)
  if face is not None:
    os.makedirs(os.path.join(processed_data_dir, img_path.split('/')[-2]),
                exist_ok=True)
    face = face.permute(1, 2, 0).int().numpy()
    plt.imsave(os.path.join(processed_data_dir, img_path.split('/')[-2],
                            img_path.split('/')[-1]), face.astype(np.uint8))