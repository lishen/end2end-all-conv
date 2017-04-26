import pandas as pd
import numpy as np
from os import path


class DMMetaManager(object):
    '''Class for reading meta data and feeding them to training
    '''

    def __init__(self, 
                 img_tsv='./metadata/images_crosswalk.tsv', 
                 exam_tsv='./metadata/exams_metadata.tsv', 
                 img_folder='./trainingData', 
                 img_extension='dcm',
                 exam_df=None):
        '''Constructor for DMMetaManager
        Args:
            img_tsv ([str]): path to the image meta .tsv file. 
            exam_tsv ([str]): path to the exam meta .tsv file. Default is None
                    because this file is not available to SC1. 
            img_folder ([str]): path to the folder where the images are stored.
            img_extension ([str]): image file extension. Default is 'dcm'.
        '''

        if exam_df is None:
            def mod_file_path(name):
                '''Change file name extension and append folder path.
                '''
                return path.join(img_folder, 
                                 path.splitext(name)[0] + '.' + img_extension)

            img_df = pd.read_csv(img_tsv, sep="\t", na_values=['.', '*'])
            try:
                img_df_indexed = img_df.set_index(['subjectId', 'examIndex'])
            except KeyError:
                img_df_indexed = img_df.set_index(['subjectId'])
            if exam_tsv is not None:
                exam_df = pd.read_csv(exam_tsv, sep="\t", na_values=['.', '*'])
                exam_df_indexed = exam_df.set_index(['subjectId', 'examIndex'])
                self.exam_img_df = exam_df_indexed.join(img_df_indexed)
                self.exam_img_df['filename'] = \
                    self.exam_img_df['filename'].apply(mod_file_path)
            else:
                img_df_indexed['filename'] = \
                    img_df_indexed['filename'].apply(mod_file_path)
                self.img_df_indexed = img_df_indexed
        else:
            self.set_exam_df(exam_df)

        # Setup CC and MLO view categorization.
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
        self.view_cat_dict = {
            'CC': 'CC', 'CCID': 'CC', 'FB': 'CC', 'LM': 'CC', 
            'ML': 'CC', 'MLID': 'CC', 'XCCL': 'CC', 'XCCM': 'CC',
            'MLO': 'MLO', 'LMO': 'MLO', 'MLOID': 'MLO', 'SIO': 'MLO'}


    def get_exam_df(self):
        '''Get exam dataframe
        '''
        try:
            return self.exam_img_df
        except AttributeError:
            return self.img_df_indexed


    def set_exam_df(self, exam_df):
        '''Set exam dataframe from external object
        '''
        if 'cancerL' in exam_df.columns:
            self.exam_img_df = exam_df
            try:
                del self.img_df_indexed
            except AttributeError:
                pass
        else:
            self.img_df_indexed = exam_df
            try:
                del self.exam_img_df
            except AttributeError:
                pass


    def get_flatten_img_list(self, subj_list=None, meta=False):
        '''Get image-level training data list
        Args:
            meta ([bool]): whether to return meta info or not. Default is 
                    False.
        '''
        img = []
        lab = []
        for subj_id, ex_idx, exam_dat in self.exam_generator(subj_list):
            for idx, dat in exam_dat.iterrows():
                img_name = dat['filename']
                laterality = dat['laterality']
                try:
                    cancer = dat['cancerL'] if laterality == 'L' else dat['cancerR']
                    try:
                        cancer = int(cancer)
                    except ValueError:
                        cancer = np.nan
                except KeyError:
                    try:
                        cancer = int(dat['cancer'])
                    except KeyError:
                        cancer = np.nan
                img.append(img_name)
                lab.append(cancer)

        return (img, lab)


    def get_info_per_exam(self, exam, flatten_img_list=False, cc_mlo_only=False):
        '''Get training-related info for each exam as a dict
        Args:
            exam (DataFrame): data for an exam.
            flatten_img_list ([bool]): whether or not return a flatten image 
                    list for each breast.
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
            try:
                cancerL = int(cancerL)
            except ValueError:
                cancerL = np.nan
            try:
                cancerR = int(cancerR)
            except ValueError:
                cancerR = np.nan
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
        if flatten_img_list:
            for breast in exam_indexed.index.levels[0]:
                info[breast]['img'] = exam_indexed.loc[breast]['filename'].tolist()
        elif cc_mlo_only:
            try:
                info['L']['CC'] = exam_indexed.loc['L'].loc['CC']['filename'].tolist()
            except KeyError:
                info['L']['CC'] = None
            try:
                info['R']['CC'] = exam_indexed.loc['R'].loc['CC']['filename'].tolist()
            except KeyError:
                info['R']['CC'] = None
            try:
                info['L']['MLO'] = exam_indexed.loc['L'].loc['MLO']['filename'].tolist()
            except KeyError:
                info['L']['MLO'] = None
            try:
                info['R']['MLO'] = exam_indexed.loc['R'].loc['MLO']['filename'].tolist()
            except KeyError:
                info['R']['MLO'] = None
        else:
            info['L']['CC'] = None
            info['R']['CC'] = None
            info['L']['MLO'] = None
            info['R']['MLO'] = None
            for breast in exam_indexed.index.levels[0]:
                for view in exam_indexed.loc[breast].index.levels[0]:
                    if view not in self.view_cat_dict:
                        continue  # skip uncategorized view for now.
                    view_ = self.view_cat_dict[view]
                    fname_df = exam_indexed.loc[breast].loc[view][['filename']]
                    if fname_df.empty:
                        continue
                    if info[breast][view_] is None:
                        info[breast][view_] = fname_df
                    elif view == 'CC' or view == 'MLO':
                        # Make sure canonical views are always on top.
                        info[breast][view_] = fname_df.append(info[breast][view_])
                    else:
                        info[breast][view_] = info[breast][view_].append(fname_df)

        return info


    def subj_generator(self, subj_list=None):
        '''A generator for the data of each subject
        Args:
            subj_list ([list]): a subset list of subject ids.
        Returns:
            A tuple of (subject ID, the corresponding records of the subject).
        '''
        try:
            df = self.exam_img_df
        except AttributeError:
            df = self.img_df_indexed
        if subj_list is None:
            try:
                subj_list = df.index.levels[0]
            except AttributeError:
                subj_list = df.index.unique()
        for subj_id in subj_list:
            yield (subj_id, df.loc[subj_id])


    def exam_generator(self, subj_list=None):
        '''A generator for the data of each exam
        Returns:
            A tuple of (subject ID, exam Index, the corresponding records of 
            the exam).
        Notes:
            All exams are flattened. When examIndex is unavailable, the 
            returned exam index is equal to the subject ID.
        '''
        for subj_id, subj_dat in self.subj_generator(subj_list):
            for ex_idx in subj_dat.index.unique():
                yield (subj_id, ex_idx, subj_dat.loc[ex_idx])


    def last_exam_generator(self, subj_list=None):
        '''A generator for the data of the last exam of each subject
        Returns:
            A tuple of (subject ID, exam Index, the corresponding records of 
            the exam).
        Notes:
            When examIndex is unavailable, the returned exam index is equal to 
            the subject ID.
        '''
        for subj_id, subj_dat in self.subj_generator(subj_list):
            last_idx = subj_dat.index.max()
            yield (subj_id, last_idx, subj_dat.loc[last_idx])


    def flatten_2_exam_generator(self, subj_list=None):
        '''A generator for the data of the flatten 2 exams of each subject
        Returns:
            A tuple of (subject ID, current exam Index, current exam data,
            prior exam Index, prior exam data). If no prior exam is present, 
            will return None.
        Notes:
            This generates all the pairs of the current and the prior exams. 
            The function is meant for SC2.
        '''
        for subj_id, subj_dat in self.subj_generator(subj_list):
            nb_exam = len(subj_dat.index.unique())
            if nb_exam == 1:
                yield (subj_id, 1, subj_dat.loc[1], None, None)
            else:
                for prior_idx in xrange(1, nb_exam):
                    curr_idx = prior_idx + 1
                    yield (subj_id, curr_idx, subj_dat.loc[curr_idx], 
                           prior_idx, subj_dat.loc[prior_idx])


    def last_2_exam_generator(self, subj_list=None):
        '''A generator for the data of the last 2 exams of each subject
        Returns:
            A tuple of (subject ID, last exam Index, last exam data,
            2nd last exam Index, 2nd last exam data). If no prior exam is 
            present, will return None.
        Notes:
            The function is meant for SC2.
        '''
        for subj_id, subj_dat in self.subj_generator(subj_list):
            nb_exam = len(subj_dat.index.unique())
            if nb_exam == 1:
                yield (subj_id, 1, subj_dat.loc[1], None, None)
            else:
                curr_idx = nb_exam
                prior_idx = curr_idx - 1
                yield (subj_id, curr_idx, subj_dat.loc[curr_idx], 
                       prior_idx, subj_dat.loc[prior_idx])


    def get_flatten_2_exam_dat(self, subj_list=None, pred_tsv=None):
        '''Get the info about the flatten 2 exams as a dataframe
        Returns: 
            a tuple of (df, labs) where df is a dataframe of exam pair info 
            for breasts; labs is the corresponding cancer labels.
        '''
        rec_list = []
        lab_list = []
        if pred_tsv is not None:
            pred_df = pd.read_csv(pred_tsv, sep="\t")
            pred_df = pred_df.set_index(['subjectId', 'examIndex', 'laterality'])

        for subj_id, curr_idx, curr_dat, prior_idx, prior_dat in \
                self.flatten_2_exam_generator(subj_list):
            left_record, right_record = \
                DMMetaManager.get_info_exam_pair(curr_dat, prior_dat)
            if pred_tsv is not None:
                nb_days = left_record['daysSincePreviousExam']
                curr_left_score = pred_df.loc[subj_id].loc[curr_idx].loc['L']['confidence']
                curr_right_score = pred_df.loc[subj_id].loc[curr_idx].loc['R']['confidence']
                try:
                    prior_left_score = pred_df.loc[subj_id].loc[prior_idx].loc['L']['confidence']
                    prior_right_score = pred_df.loc[subj_id].loc[prior_idx].loc['R']['confidence']
                    diff_left_score = (curr_left_score - prior_left_score)/nb_days*365
                    diff_right_score = (curr_right_score - prior_right_score)/nb_days*365
                except TypeError:
                    prior_left_score = np.nan
                    prior_right_score = np.nan
                    diff_left_score = np.nan
                    diff_right_score = np.nan
                left_record = left_record\
                        .assign(curr_score=curr_left_score)\
                        .assign(prior_score=prior_left_score)\
                        .assign(diff_score=diff_left_score)
                right_record = right_record\
                        .assign(curr_score=curr_right_score)\
                        .assign(prior_score=prior_right_score)\
                        .assign(diff_score=diff_right_score)
            rec_list.append(left_record)
            rec_list.append(right_record)

            try:
                left_cancer = int(curr_dat['cancerL'].iloc[0])
            except ValueError:
                # left_cancer = np.nan
                left_cancer = 0
            try:
                right_cancer = int(curr_dat['cancerR'].iloc[0])
            except ValueError:
                # right_cancer = np.nan
                right_cancer = 0
            lab_list.append(left_cancer)
            lab_list.append(right_cancer)

        df = pd.concat(rec_list, ignore_index=True)
        labs = np.array(lab_list)
        
        return df, labs


    def get_subj_dat_list(self, subj_list=None, meta=False):
        '''Get subject-level training data list
        Returns:
            A list of all subjects. Each element is a tuple of (subject ID, 
            [ (exam Index, extracted exam info), ..., () ] ).
        '''
        subj_dat_list = []
        for subj_id, subj_dat in self.subj_generator(subj_list):
            subj_exam_list = []
            for ex_idx in subj_dat.index.unique():  # uniq exam indices.
                exam_info = self.get_info_per_exam(subj_dat.loc[ex_idx])
                subj_exam_list.append( (ex_idx, exam_info) )
            subj_dat_list.append( (subj_id, subj_exam_list) )
        return subj_dat_list


    def get_subj_labs(self):
        '''Get subject IDs and their last exam labels
        '''
        subj_list = []
        lab_list = []
        for subj_id, ex_idx, exam_dat in self.last_exam_generator():
            subj_list.append(subj_id)
            try:
                cancerL = (exam_dat['cancerL'] == 1).sum() > 0
                cancerR = (exam_dat['cancerR'] == 1).sum() > 0
                lab_list.append(1 if cancerL or cancerR else 0)
            except KeyError:
                try:
                    cancer = (exam_dat['cancer'] == 1).sum() > 0
                    lab_list.append(1 if cancer else 0)
                except KeyError:
                    lab_list.append(np.nan)
        return subj_list, lab_list


    def get_flatten_exam_list(self, subj_list=None, meta=False, 
                              flatten_img_list=False, cc_mlo_only=False):
        '''Get exam-level training data list
        Returns:
            A list of all exams for all subjects. Each element is a tuple of 
            (subject ID, exam Index, a dict of extracted info for the exam).
        '''
        exam_list = []
        for subj_id, ex_idx, exam_dat in self.exam_generator(subj_list):
            exam_list.append(
                (subj_id, ex_idx, 
                 self.get_info_per_exam(
                    exam_dat, flatten_img_list=flatten_img_list, 
                    cc_mlo_only=cc_mlo_only))
            )
        return exam_list


    @staticmethod
    def get_info_exam_pair(curr_dat, prior_dat):
        '''Extract meta info from current and prior exams
        Returns: 
            a tuple of (left_df, right_df), where left_df and right_df are both
            dataframes containing meta info about the current and prior exams.
        '''
        # number of days since last exam.
        nb_days = curr_dat['daysSincePreviousExam'].iloc[0]
        # prior cancer invasive or not.
        try:
            left_prior_inv = prior_dat['invL'].iloc[0]
        except TypeError:
            left_prior_inv = np.nan
        try:
            right_prior_inv = prior_dat['invR'].iloc[0]
        except TypeError:
            right_prior_inv = np.nan
        # current, prior and diff bmi.
        curr_bmi = curr_dat['bmi'].iloc[0]
        try:
            prior_bmi = prior_dat['bmi'].iloc[0]
            diff_bmi = (curr_bmi - prior_bmi)/nb_days*365
        except TypeError:
            prior_bmi = np.nan
            diff_bmi = np.nan
        # implantation.
        implantNow = curr_dat['implantNow'].iloc[0]
        if implantNow == 2:
            left_implantNow = 1
            right_implantNow = 0
        elif implantNow == 1:
            left_implantNow = 0
            right_implantNow = 1
        elif implantNow == 4:
            left_implantNow = 1
            right_implantNow = 1            
        elif implantNow == 5:
            left_implantNow = .5
            right_implantNow = .5
        else:
            left_implantNow = np.nan
            right_implantNow = np.nan
        try:
            implantPrior = prior_dat['implantNow'].iloc[0]
            if implantPrior == 2:
                left_implantPrior = 1
                right_implantPrior = 0
            elif implantPrior == 1:
                left_implantPrior = 0
                right_implantPrior = 1
            elif implantPrior == 4:
                left_implantPrior = 1
                right_implantPrior = 1
            elif implantPrior == 5:
                left_implantPrior = .5
                right_implantPrior = .5
            else:
                left_implantPrior = np.nan
                right_implantPrior = np.nan
        except TypeError:
            left_implantPrior = np.nan
            right_implantPrior = np.nan                
        # previous breast cancer history.
        previousBcLaterality = curr_dat['previousBcLaterality'].iloc[0]
        if previousBcLaterality == 2:
            left_previousBcHistory = 1
            right_previousBcHistory = 0
        elif previousBcLaterality == 1:
            left_previousBcHistory = 0
            right_previousBcHistory = 1
        elif previousBcLaterality == 3:
            left_previousBcHistory = .5
            right_previousBcHistory = .5
        elif previousBcLaterality == 4:
            left_previousBcHistory = 1
            right_previousBcHistory = 1
        else:
            left_previousBcHistory = 0
            right_previousBcHistory = 0
        # breast reduction history.
        reduxLaterality = curr_dat['reduxLaterality'].iloc[0]
        if reduxLaterality == 2:
            left_reduxHistory = 1
            right_reduxHistory = 0
        elif reduxLaterality == 1:
            left_reduxHistory = 0
            right_reduxHistory = 1
        elif reduxLaterality == 4:
            left_reduxHistory = 1
            right_reduxHistory = 1
        else:
            left_reduxHistory = 0
            right_reduxHistory = 0
        # hormone replacement therapy.
        curr_hrt = curr_dat['hrt'].iloc[0]
        curr_hrt = np.nan if curr_hrt == 9 else curr_hrt
        try:
            prior_hrt = prior_dat['hrt'].iloc[0]
            prior_hrt = np.nan if prior_hrt == 9 else prior_hrt
        except TypeError:
            prior_hrt = np.nan
        # anti-estrogen therapy.
        curr_antiestrogen = curr_dat['antiestrogen'].iloc[0]
        curr_antiestrogen = np.nan if curr_antiestrogen == 9 else curr_antiestrogen
        try:
            prior_antiestrogen = prior_dat['antiestrogen'].iloc[0]
            prior_antiestrogen = np.nan if prior_antiestrogen == 9 else prior_antiestrogen
        except TypeError:
            prior_antiestrogen = np.nan
        # first degree relative with BC.
        firstDegreeWithBc = curr_dat['firstDegreeWithBc'].iloc[0]
        firstDegreeWithBc = np.nan if firstDegreeWithBc == 9 else firstDegreeWithBc
        # first degree relative with BC under 50.
        firstDegreeWithBc50 = curr_dat['firstDegreeWithBc50'].iloc[0]
        firstDegreeWithBc50 = np.nan if firstDegreeWithBc50 == 9 else firstDegreeWithBc50
        # race.
        race = curr_dat['race'].iloc[0]
        race = np.nan if race == 9 else race

        # put all info input a dict.
        left_record = {
            'daysSincePreviousExam': nb_days,
            'prior_inv': left_prior_inv,
            'age': curr_dat['age'].iloc[0],
            'implantEver': curr_dat['implantEver'].iloc[0],
            'implantNow': left_implantNow,
            'implantPrior': left_implantPrior,
            'previousBcHistory': left_previousBcHistory,
            'yearsSincePreviousBc': curr_dat['yearsSincePreviousBc'].iloc[0],
            'reduxHistory': left_reduxHistory,
            'curr_hrt': curr_hrt,
            'prior_hrt': prior_hrt,
            'curr_antiestrogen': curr_antiestrogen,
            'prior_antiestrogen': prior_antiestrogen,
            'curr_bmi': curr_bmi,
            'prior_bmi': prior_bmi,
            'diff_bmi': diff_bmi,
            'firstDegreeWithBc': firstDegreeWithBc,
            'firstDegreeWithBc50': firstDegreeWithBc50,
            'race': race
        }
        right_record = {
            'daysSincePreviousExam': nb_days,
            'prior_inv': right_prior_inv,
            'age': curr_dat['age'].iloc[0],
            'implantEver': curr_dat['implantEver'].iloc[0],
            'implantNow': right_implantNow,
            'implantPrior': right_implantPrior,
            'previousBcHistory': right_previousBcHistory,
            'yearsSincePreviousBc': curr_dat['yearsSincePreviousBc'].iloc[0],
            'reduxHistory': right_reduxHistory,
            'curr_hrt': curr_hrt,
            'prior_hrt': prior_hrt,
            'curr_antiestrogen': curr_antiestrogen,
            'prior_antiestrogen': prior_antiestrogen,
            'curr_bmi': curr_bmi,
            'prior_bmi': prior_bmi,
            'diff_bmi': diff_bmi,
            'firstDegreeWithBc': firstDegreeWithBc,
            'firstDegreeWithBc50': firstDegreeWithBc50,
            'race': race
        }
        return (pd.DataFrame(left_record, index=[0]), 
                pd.DataFrame(right_record, index=[0]))


    @staticmethod
    def exam_labs(exam_list):
        return [ 1 if e[2]['L']['cancer']==1 or e[2]['R']['cancer']==1 else 0 
                 for e in exam_list ]

    @staticmethod
    def flatten_exam_labs(exam_list):
        labs = []
        for e in exam_list:
            lc = e[2]['L']['cancer']
            rc = e[2]['R']['cancer']
            lc = lc if not np.isnan(lc) else 0
            rc = rc if not np.isnan(rc) else 0
            labs.append(lc)
            labs.append(rc)
        return labs

    @staticmethod
    def exam_list_summary(exam_list):
        '''Return a summary dataframe for an exam list
        '''
        subj_list = []
        exid_list = []
        l_cc_list = []
        l_mlo_list = []
        r_cc_list = []
        r_mlo_list = []
        l_can_list = []
        r_can_list = []
        def nb_fname(df):
            return 0 if df is None else df.shape[0]
        for e in exam_list:
            subj_list.append(e[0])
            exid_list.append(e[1])
            l_cc_list.append(nb_fname(e[2]['L']['CC']))
            l_mlo_list.append(nb_fname(e[2]['L']['MLO']))
            r_cc_list.append(nb_fname(e[2]['R']['CC']))
            r_mlo_list.append(nb_fname(e[2]['R']['MLO']))
            l_can_list.append(e[2]['L']['cancer'])
            r_can_list.append(e[2]['R']['cancer'])
        summary_df = pd.DataFrame(
            {'subj': subj_list, 'exam': exid_list, 
             'L_CC': l_cc_list, 'L_MLO': l_mlo_list,
             'R_CC': r_cc_list, 'R_MLO': r_mlo_list,
             'L_cancer': l_can_list, 'R_cancer': r_can_list})
        return summary_df


    def get_last_exam_list(self, subj_list=None, meta=False, 
                           flatten_img_list=False, cc_mlo_only=False):
        '''Get the last exam training data list
        Returns:
            A list of the last exams for each subject. Each element is a tuple 
            of (subject ID, exam Index, a dict of extracted info for the exam).
        '''
        exam_list = []
        for subj_id, ex_idx, exam_dat in self.last_exam_generator(subj_list):
            exam_list.append(
                (subj_id, ex_idx, 
                 self.get_info_per_exam(
                    exam_dat, flatten_img_list=flatten_img_list, 
                    cc_mlo_only=cc_mlo_only))
            )
        return exam_list


    @staticmethod
    def subset_img_labs(img_list, lab_list, neg_vs_pos_ratio, seed=12345):
        rng = np.random.RandomState(seed)
        img_list = np.array(img_list)
        lab_list = np.array(lab_list)
        pos_idx = np.where(lab_list==1)[0]
        neg_idx = np.where(lab_list==0)[0]
        nb_neg_desired = int(len(pos_idx)*neg_vs_pos_ratio)
        if nb_neg_desired < len(neg_idx):
            sampled_neg_idx = rng.choice(neg_idx, nb_neg_desired, replace=False)
            all_idx = np.concatenate([pos_idx, sampled_neg_idx])
            img_list = img_list[all_idx].tolist()
            lab_list = lab_list[all_idx].tolist()
            return img_list, lab_list
        else:
            return img_list.tolist(), lab_list.tolist()


    @staticmethod
    def subset_exam_list(exam_list, neg_vs_pos_ratio, seed=12345):
        rng = np.random.RandomState(seed)
        exam_labs = np.array(DMMetaManager.exam_labs(exam_list))
        pos_idx = np.where(exam_labs==1)[0]
        neg_idx = np.where(exam_labs==0)[0]
        nb_neg_desired = int(len(pos_idx)*neg_vs_pos_ratio)
        if nb_neg_desired < len(neg_idx):
            sampled_neg_idx = rng.choice(neg_idx, nb_neg_desired, replace=False)
            all_idx = np.concatenate([pos_idx, sampled_neg_idx])
            sample_mask = np.zeros(len(exam_list), dtype='bool')
            sample_mask[all_idx] = True
            sampled_exam_list = [ exam for i,exam in enumerate(exam_list) 
                                  if sample_mask[i]]
            return sampled_exam_list
        else:
            return exam_list


    @staticmethod
    def subset_subj_list(subj_list, subj_labs, neg_vs_pos_ratio, seed=12345):
        rng = np.random.RandomState(seed)
        subj_list = np.array(subj_list)
        subj_labs = np.array(subj_labs)
        pos_idx = np.where(subj_labs==1)[0]
        neg_idx = np.where(subj_labs==0)[0]
        nb_neg_desired = int(len(pos_idx)*neg_vs_pos_ratio)
        if nb_neg_desired < len(neg_idx):
            sampled_neg_idx = rng.choice(neg_idx, nb_neg_desired, replace=False)
            all_idx = np.concatenate([pos_idx, sampled_neg_idx])
            subj_list = subj_list[all_idx].tolist()
            subj_labs = subj_labs[all_idx].tolist()
            return subj_list, subj_labs
        else:
            return subj_list.tolist(), subj_labs.tolist()









