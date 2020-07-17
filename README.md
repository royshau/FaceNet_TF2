
# FaceNet TF2
An implementation of the FaceNet paper in TensorFlow2.

Download of the external data is needed ([LFW](http://vis-www.cs.umass.edu/lfw/), [VGGFace2](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/),[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)).
After the data is downloaded and unzipped, use "preprocess_data.py" to preprocess the datasets (may take a long time. TODO: batch process images)

The model is trained using the train.py script. You can change various model and training configurations in the file.
To evaluate on the LFW dataset, use "evaluate_on_lfw.py". 

TODOS:

 - [ ] Batch preprocessing
 - [ ] Improve code documentation and README
 - [ ] A simple script (maybe using Colab?) to run inference on own images.

## Acknowlegments
The project was created for "Selected Topics in Image Processing" course in Ben Gurion University of the lecturer Yitzhak Yitzhaky.

The project uses great other FaceNet and triplet loss implementations available in GitHub:

 - [Triplet loss in TensorFlow](https://github.com/omoindrot/tensorflow-triplet-loss)  - Triplet loss implementation heavily relies on the TF1 implementation and the excellent blog post by Olivier Moindrot.
 - [Face Recognition using Tensorflow](https://github.com/davidsandberg/facenet) - The evaluation script on LFW is used and modified to use TF2. Ideas from the documentation and code are also used.