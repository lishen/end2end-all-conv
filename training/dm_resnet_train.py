from sklearn.cross_validation import train_test_split
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import SGD
import os
from meta import UNIMAGED_INT, DMMetaManager
from dm_image import DMImageDataGenerator
from dm_resnet import ResNetBuilder

# Setup training and validation data.
img_wid = 1152
img_hei = 896
batch_size = 4
random_seed = os.getenv('RANDOM_SEED', 12345)
meta_man = DMMetaManager(exam_tsv='metadata/exams_metadata.tsv', 
                         img_tsv='metadata/images_crosswalk.tsv', 
                         img_folder='preprocessedData/jpg_org/', 
                         img_extension='jpg')
img_list, lab_list = meta_man.get_flatten_img_list()
img_train, img_val, lab_train, lab_val = train_test_split(
    img_list, lab_list, test_size=.2, random_state=random_seed, stratify=lab_list)
img_gen = DMImageDataGenerator(
    samplewise_center=True, 
    samplewise_std_normalization=True, 
    horizontal_flip=True, 
    vertical_flip=True)
train_generator = img_gen.flow_from_img_list(
    img_train, lab_train, target_size=(img_wid, img_hei), 
    batch_size=batch_size, seed=random_seed)
val_generator = img_gen.flow_from_img_list(
    img_val, lab_val, target_size=(img_wid, img_hei), 
    batch_size=batch_size, seed=random_seed)

# Model training.
model = ResNetBuilder.build_dm_resnet_68((1, img_wid, img_hei), 1, .0)
sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=True)
model.compile(optimizer=sgd, loss='binary_crossentropy', 
              metrics=['accuracy', 'precision', 'recall'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
model.fit_generator(train_generator, 
                    samples_per_epoch=32, 
                    nb_epoch=20, 
                    validation_data=val_generator, 
                    nb_val_samples=len(img_val), 
                    callbacks=[reduce_lr])

# weight decay
# momentum
# mini-batch size
# multiple GPUs
# optimizer
# early stopping

