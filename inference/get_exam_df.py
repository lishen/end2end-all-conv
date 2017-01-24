from meta import DMMetaManager

man = DMMetaManager(img_tsv='/metadata/images_crosswalk.tsv', 
                    exam_tsv='/metadata/exams_metadata.tsv', 
                    img_folder='/trainingData', 
                    img_extension='dcm')
df = man.get_exam_df()
df.to_pickle('/modelState/exam_df.pkl')
