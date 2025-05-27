#%%
# create ev files which replace original during 0 into 0.1
import os
import pandas as pd
behav_dir = '/Users/xiaoqianxiao/projects/NARSAD/MRI/source_data/behav'
file_dic = {'task-Narsad_phase2_half_events.csv': 'single_trial_task-Narsad_phase2_half_events.csv',
            'task-Narsad_phase3_half_events.csv': 'single_trial_task-Narsad_phase3_half_events.csv',
            'task-Narsad_phase-3_sub-202_half_events.csv': 'single_trial_task-Narsad_phase-3_sub-202_half_events.csv'}
for ori_file, new_file in file_dic.items():
    file = ori_file
    file_path = os.path.join(behav_dir, file)
    new_file_path = os.path.join(behav_dir, new_file)
    df = pd.read_csv(file_path, sep='\t')
    df.loc[df['duration'] == 0, 'duration'] = 0.1
    df.to_csv(new_file_path, index=False, sep='\t')