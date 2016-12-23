import pandas as pd
import numpy as np
from os import path


class DMMetaManager(object):
    '''Class for reading meta data and feeding them to training
    '''

    def __init__(self, 
                 img_tsv='./metadata/images_crosswalk.tsv', 
                 exam_tsv=None, 
                 img_folder='./trainingData', 
                 img_extension='dcm'):
        '''Constructor for DMMetaManager
        Args:
            img_tsv ([str]): path to the image meta .tsv file. 
            exam_tsv ([str]): path to the exam meta .tsv file. Default is None
                    because this file is not available to SC1. 
            img_folder ([str]): path to the folder where the images are stored.
            img_extension ([str]): image file extension. Default is 'dcm'.
        '''

        def mod_file_path(name):
            '''Change file name extension and append folder path.
            '''
            return path.join(img_folder, 
                             path.splitext(name)[0] + '.' + img_extension)

        img_df = pd.read_csv(img_tsv, sep="\t")
        try:
            img_df_indexed = img_df.set_index(['subjectId', 'examIndex'])
        except KeyError:
            img_df_indexed = img_df.set_index(['subjectId'])
        if exam_tsv is not None:
            exam_df = pd.read_csv(exam_tsv, sep="\t")
            exam_df_indexed = exam_df.set_index(['subjectId', 'examIndex'])
            self.exam_img_df = exam_df_indexed.join(img_df_indexed)
            self.exam_img_df['filename'] = \
                self.exam_img_df['filename'].apply(mod_file_path)
        else:
            img_df_indexed['filename'] = \
                img_df_indexed['filename'].apply(mod_file_path)
            self.img_df_indexed = img_df_indexed


    def get_flatten_img_list(self, meta=False):
        '''Get image-level training data list
        Args:
            meta ([bool]): whether to return meta info or not. Default is 
                    False.
        '''
        img = []
        lab = []
        try:
            for idx, dat in self.exam_img_df.iterrows():
                img_name = dat['filename']
                laterality = dat['laterality']
                cancer = dat['cancerL'] if laterality == 'L' else dat['cancerR']
                # No need to worry about the rows where cancer='.' because the 
                # labels of '.' will not appear in the image list.
                # For example, if cancerL = '.', laterality must be 'R' and 
                # cancerR can't be '.', and vice versa.
                cancer = int(cancer) if cancer != '*' else np.nan
                img.append(img_name)
                lab.append(cancer)
        except AttributeError:
            for idx, dat in self.img_df_indexed.iterrows():
                img_name = dat['filename']
                try:
                    cancer = int(dat['cancer'])
                except KeyError:
                    cancer = np.nan
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
        exam_indexed = exam.set_index(['laterality', 'view', 'imageIndex'])
        # Determine cancer status.
        try:
            cancerL = exam_indexed['cancerL'].iloc[0]
            cancerR = exam_indexed['cancerR'].iloc[0]
            cancerL = np.nan if cancerL == '.' or cancerL == '*' else int(cancerL)
            cancerR = np.nan if cancerR == '.' or cancerR == '*' else int(cancerR)
        except KeyError:
            try:
                cancerL = int(exam_indexed.loc['L']['cancer'].iloc[0])
            except KeyError:
                cancerL = np.nan
            try:
                cancerR = int(exam_indexed.loc['R']['cancer'].iloc[0])
            except KeyError:
                cancerR = np.nan
        info['L']['cancer'] = cancerL
        info['R']['cancer'] = cancerR
        # Obtain file names for different views.
        def view_fnames(exam_df, breast, view_list):
            '''Obtain the file names for a view list for a breast
            Returns:
                a DataFrame object contains the file names.
            '''
            df_list = []
            for view in view_list:
                try:
                    df_list.append(exam_df.loc[breast].loc[view][['filename']])
                except KeyError:
                    df_list.append(None)
            try:
                return pd.concat(df_list)
            except ValueError:
                return None

        # View      Description
        # *Undetermined yet.
        # AT        axillary tail  *
        # CC        craniocaudal
        # CCID      craniocaudal (implant displaced)
        # CV        cleavage  *
        # FB        from below
        # LM        90 lateromedial
        # LMO       lateromedial oblique
        # ML        90 mediolateral
        # MLID      90 mediolateral (implant displaced)
        # MLO       mediolateral oblique
        # MLOID     mediolateral oblique (implant displaced)
        # RL        rolled lateral  *
        # RM        rolled medial  *
        # SIO       superior inferior oblique
        # XCCL      exaggerated craniocaudal lateral
        # XCCM      exaggerated craniocaudal medial

        cc_like_list = ['CC', 'CCID', 'FB', 'LM', 'ML', 'MLID', 'XCCL', 'XCCM']
        mlo_like_list = ['MLO', 'LMO', 'MLOID', 'SIO']
        info['L']['CC'] = view_fnames(exam_indexed, 'L', cc_like_list)
        info['R']['CC'] = view_fnames(exam_indexed, 'R', cc_like_list)
        info['L']['MLO'] = view_fnames(exam_indexed, 'L', mlo_like_list)
        info['R']['MLO'] = view_fnames(exam_indexed, 'R', mlo_like_list)

        return info


    def subj_generator(self):
        '''A generator for the data of each subject
        Returns:
            A tuple of (subject ID, the corresponding records of the subject).
        '''
        try:
            df = self.exam_img_df
        except AttributeError:
            df = self.img_df_indexed
        try:
            subj_list = df.index.levels[0]
        except AttributeError:
            subj_list = df.index.unique()
        for subj_id in subj_list:
            yield (subj_id, df.loc[subj_id])


    def exam_generator(self):
        '''A generator for the data of each exam
        Returns:
            A tuple of (subject ID, exam Index, the corresponding records of 
            the exam).
        Notes:
            All exams are flattened. When examIndex is unavailable, the 
            returned exam index is equal to the subject ID.
        '''
        for subj_id, subj_dat in self.subj_generator():
            for ex_idx in subj_dat.index.unique():
                yield (subj_id, ex_idx, subj_dat.loc[ex_idx])


    def last_exam_generator(self):
        '''A generator for the data of the last exam of each subject
        Returns:
            A tuple of (subject ID, exam Index, the corresponding records of 
            the exam).
        Notes:
            When examIndex is unavailable, the returned exam index is equal to 
            the subject ID.
        '''
        for subj_id, subj_dat in self.subj_generator():
            last_idx = subj_dat.index.max()
            yield (subj_id, last_idx, subj_dat.loc[last_idx])


    def get_subj_list(self, meta=False):
        '''Get subject-level training data list
        Returns:
            A list of all subjects. Each element is a tuple of (subject ID, 
            [ (exam Index, extracted exam info), ..., () ] ).
        '''
        subj_list = []
        for subj_id, subj_dat in self.subj_generator():
            subj_exam_list = []
            for ex_idx in subj_dat.index.unique():  # uniq exam indices.
                exam_info = self._get_info_per_exam(subj_dat.loc[ex_idx])
                subj_exam_list.append( (ex_idx, exam_info) )
            subj_list.append( (subj_id, subj_exam_list) )
        return subj_list


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












