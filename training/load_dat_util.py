from meta import DMMetaManager
meta_man = DMMetaManager(img_folder='preprocessedData/png_288x224/', img_extension='png')
exam_list = meta_man.get_flatten_exam_list()
img_list = meta_man.get_flatten_img_list()
from dm_image import DMImageDataGenerator
img_gen = DMImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
img_gen.mean = 7772.
img_gen.std = 12187.
datgen_exam = img_gen.flow_from_exam_list(exam_list, target_size=(288, 224), batch_size=8, shuffle=False, seed=123)
datgen_image = img_gen.flow_from_img_list(img_list[0], img_list[1], target_size=(288, 224), batch_size=32, shuffle=False, seed=123)
import numpy as np

