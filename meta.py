import pandas as pd
from os import path

class DMMetaManager(object):
    '''Class for reading meta data and feeding them to training
    '''

    def __init__(self, exam_tsv='/metadata/exams_metadata.tsv', 
                 img_tsv='/metadata/images_crosswalk.tsv', 
                 img_folder='/trainingData', 
                 img_extension='dcm'):
        '''Constructor for DMMetaManager
        Args:
            exam_tsv ([str]): path to the exam meta .tsv file. Default is set 
                    to the default location when run in a docker container.
            img_tsv ([str]): path to the image meta .tsv file. Default is set 
                    to the default location when run in a docker container.
            img_folder ([str]): path to the folder where the images are stored.
                    Default is set to the default location when run in a docker 
                    container.
            img_extension ([str]): image file extension. Default is 'dcm'.
        '''
        exam_df = pd.read_csv(exam_tsv, sep="\t")
        img_df = pd.read_csv(img_tsv, sep="\t")
        exam_df_indexed = exam_df.set_index(['subjectId', 'examIndex'])
        img_df_indexed = img_df.set_index(['subjectId', 'examIndex'])
        self.exam_img_df = exam_df_indexed.join(img_df_indexed)
        def mod_file_path(name):
            '''Change file name extension and append folder path.
            '''
            return path.join(img_folder, 
                             path.splitext(name)[0] + '.' + img_extension)
        self.exam_img_df['filename'] = \
            self.exam_img_df['filename'].apply(mod_file_path)


    def get_flatten_img_list(self, meta=False):
        '''Get image-level training data list
        Args:
            meta ([bool]): whether to return meta info or not. Default is 
                    False.
        '''
        img = []
        lab = []
        for idx, dat in self.exam_img_df.iterrows():
            img_name = dat['filename']
            laterality = dat['laterality']
            cancer = dat['cancerL'] if laterality == 'L' else dat['cancerR']
            ##!! if breast is not imaged (marked '.'), set cancer status to 0.
            ##!! this definitely needs to be better dealt with later.
            cancer = 0 if cancer == '.' else int(cancer) 
            img.append(img_name)
            lab.append(cancer)
        return (img, lab)


    def get_flatten_breast_list(self, meta=False):
        '''Get breast-level training data list
        '''
        pass

    def get_last_breast_list(self, meta=False):
        '''Get the last exam breast-level training data list
        '''
        pass












