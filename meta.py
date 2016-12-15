import pandas as pd
from os import path

UNIMAGED_INT = -99  # A global definition for the int represention for unimaged breast.

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
            # cancer = 0 if cancer == '.' else int(cancer)
            # No need to worry about the rows where cancer='.' because the 
            # filenames must correspond to the other breast. The labels of 
            # '.' will not appear in the training data.
            cancer = int(cancer)
            img.append(img_name)
            lab.append(cancer)
        return (img, lab)


    def _get_info_per_exam(self, exam):
        '''Get training-related info for each exam as a dict
        Args:
            exam (DataFrame): data for an exam.
        Returns:
            A dict containing info for each breast: the cancer status for 
            each breast and the image paths to the CC and MLO views. If an 
            image is missing, the corresponding path is None. 
        Notes:
            In current implementation, only CC and MLO views are included. 
            All other meta info are not included.
        '''
        info = {'L': {}, 'R': {}}
        cancerL = exam['cancerL'].iloc[0]
        cancerL = int(cancerL) if cancerL != '.' else UNIMAGED_INT
        cancerR = exam['cancerR'].iloc[0]
        cancerR = int(cancerR) if cancerR != '.' else UNIMAGED_INT
        info['L']['cancer'] = cancerL
        info['R']['cancer'] = cancerR
        exam_indexed = exam.set_index(['laterality', 'view', 'imageIndex'])
        try:
            info['L']['CC'] = exam_indexed.loc['L'].loc['CC'][['filename']]
        except KeyError:
            info['L']['CC'] = None
        try:
            info['R']['CC'] = exam_indexed.loc['R'].loc['CC'][['filename']]
        except KeyError:
            info['R']['CC'] = None
        try:
            info['L']['MLO'] = exam_indexed.loc['L'].loc['MLO'][['filename']]
        except KeyError:
            info['L']['MLO'] = None
        try:
            info['R']['MLO'] = exam_indexed.loc['R'].loc['MLO'][['filename']]
        except KeyError:
            info['R']['MLO'] = None
        return info


    def subj_generator(self):
        '''A generator for the data of each subject
        Returns:
            A tuple of (subject ID, the corresponding records of the subject).
        '''
        subj_list = self.exam_img_df.index.levels[0]
        for subj_id in subj_list:
            yield (subj_id, self.exam_img_df.loc[subj_id])


    def exam_generator(self):
        '''A generator for the data of each exam
        Returns:
            A tuple of (subject ID, exam Index, the corresponding records of 
            the exam).
        Notes:
            All exams are flattened.
        '''
        for subj_id, subj_dat in self.subj_generator():
            for ex_idx in subj_dat.index.unique():
                yield (subj_id, ex_idx, subj_dat.loc[ex_idx])


    def last_exam_generator(self):
        '''A generator for the data of the last exam of each subject
        Returns:
            A tuple of (subject ID, exam Index, the corresponding records of 
            the exam).
        '''
        for subj_id, subj_dat in self.subj_generator():
            last_idx = subj_dat.index.max()
            yield (subj_id, last_idx, subj_dat.loc[last_idx])


    def get_flatten_exam_list(self, meta=False):
        '''Get exam-level training data list
        Returns:
            A list of all exams for all subjects. Each element is a tuple of 
            (subject ID, exam Index, a dict of extracted info for the exam).
        '''
        exam_list = []
        for subj_id, ex_idx, exam_dat in self.exam_generator():
            exam_list.append( (subj_id, ex_idx, 
                               self._get_info_per_exam(exam_dat)) )
        return exam_list


    def get_last_exam_list(self, meta=False):
        '''Get the last exam training data list
        Returns:
            A list of the last exams for each subject. Each element is a tuple 
            of (subject ID, exam Index, a dict of extracted info for the exam).
        '''
        exam_list = []
        for subj_id, ex_idx, exam_dat in self.last_exam_generator():
            exam_list.append( (subj_id, ex_idx, 
                               self._get_info_per_exam(exam_dat)) )
        return exam_list












