import pandas as pd
import numpy as np
import re



def load_df(filename_opensmile, filename_praat, filename_librosa, filename_behavior, filename_output):
    """Loads and merges all audio-files and the behavioral file."""
    
    behavior_df=pd.read_csv(filename_behavior, sep=';', decimal=",", na_values=['-999'], encoding='latin-1') 
    behavior_df['vpn']=behavior_df.Subject 
    
    librosa_df=pd.read_csv(filename_librosa, sep=',', na_values=['?'], encoding='latin-1')  
    librosa_features=['spectrum_0', 'spectrum_1', 'spectrum_2','spectrum_3', 'spectrum_4',  
             'spectrum_5', 'spectrum_6', 'spectrum_7', 'spectrum_8', 'spectrum_9',
             'spectrum_10', 'spectrum_11', 'spectrum_12',  'spectrum_13', 'spectrum_14',
             'spectrum_15', 'spectrum_16', 'spectrum_17', 'spectrum_18', 'spectrum_19',
             'spectrum_20', 'spectrum_21',  'spectrum_22', 'spectrum_23', 'spectrum_24', 
             'spectrum_25', 'spectrum_26', 'spectrum_27', 'spectrum_28', 'spectrum_29', 
             'spectrum_30', 'spectrum_31', 'spectrum_32', 'spectrum_33', 'spectrum_34',
             'spectrum_35', 'spectrum_36', 'spectrum_37', 'spectrum_38', 'spectrum_39', 'energy', 'length', 'vpn']
    librosa_df.columns=librosa_features
    #librosa_df['vpn']=pd.to_numeric(librosa_df.vpn.str[2:4])
    librosa_df["vpn"] = librosa_df["vpn"].apply(_vpnum).astype("Int64")
    librosa_df['frame']=librosa_df.groupby('vpn').cumcount() 
    librosa_df['time']=librosa_df.groupby('vpn')['frame'].max()
    
    praat_features=['meanF0Hz', 'stdevF0Hz', 'HNR', 'localJitter',
           'localabsoluteJitter', 'rapJitter', 'ppq5Jitter', 'ddpJitter',
           'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer',
           'apq11Shimmer', 'ddaShimmer', 'median_pitch']
    praat_female_df=pd.read_csv(filename_praat + '_female.csv', sep=',', na_values=['?'], encoding='latin-1')  
    #praat_female_df['vpn']=pd.to_numeric(praat_female_df.voiceID.str[24:26])
    #praat_female_df["vpn"] = praat_female_df["voiceID"].str.extract(r"VP(\d+)", flags=re.IGNORECASE)[0].astype("Int64")
    praat_female_df["vpn"] = praat_female_df["voiceID"].apply(_vpnum).astype("Int64")
    praat_male_df=pd.read_csv(filename_praat + '_male.csv', sep=',', na_values=['?'], encoding='latin-1')  
    #praat_male_df['vpn']=pd.to_numeric(praat_male_df.voiceID.str[22:24]) 
    #praat_male_df["vpn"]   = praat_male_df["voiceID"].str.extract(r"VP(\d+)", flags=re.IGNORECASE)[0].astype("Int64") 
    praat_male_df["vpn"] = praat_male_df["voiceID"].apply(_vpnum).astype("Int64")
    praat_df=pd.concat([praat_female_df, praat_male_df], axis=0)

    librosa_praat=pd.merge(librosa_df.groupby('vpn').mean().reset_index(), praat_df, on =['vpn'])
    
    smile_df=pd.read_csv(filename_opensmile, sep=',', na_values=['?'], encoding='latin-1')  
    opensimle_features=smile_df.columns
    
    audio_df=pd.merge(librosa_praat, smile_df, on=['vpn'])
    
    final_df=pd.merge(audio_df, behavior_df, on=['vpn'])
    
    librosa_features.remove('vpn')
    opensimle_features=list(opensimle_features.drop(['file', 'vpn', 'start', 'end']))
    audio_features=praat_features + opensimle_features+librosa_features
    final_df[audio_features]=final_df[audio_features].apply(pd.to_numeric, errors='coerce')   
    final_df['Cond']=final_df['Cond'].apply(pd.to_numeric, errors='coerce')   

    final_df.to_csv(filename_output +'.csv')
    print (len(final_df))
    print (len(librosa_df))
    print (len(praat_df))
    print (len(behavior_df))
    
    return final_df, librosa_features, praat_features, opensimle_features


def calc_var(df):
    df['Cortisol_Max']=np.max(df[['cort_baseline', 'cort_01_min', 'cort_20_min']], axis=1)
    df['Cortisol_Min']=np.min(df[['cort_baseline', 'cort_01_min', 'cort_20_min']], axis=1)
    df['Cortisol_MinMax']=(df.Cortisol_Max)-(df.Cortisol_Min)
    df['Cortisol_React']=(df.Cortisol_Max)-(df.cort_baseline)
    df['PANA_Delta_NA']=df.PANAS_1post_NA-df.PANAS_1pre_NA
    df['PANA_Delta_PA']=df.PANAS_1post_PA-df.PANAS_1pre_PA
    df['sAA_Max']=np.max(df[['sAA_baseline', 'sAA_01_min', 'sAA_20_min']], axis=1)
    df['sAA_React']=(df.sAA_Max)-(df.sAA_baseline)
    return df

def _vpnum(x):
    if pd.isna(x):
        return None
    s = str(x).strip()

    # 1) VP + optional spaces/_/- + digits  (VP21, VP 21, VP_21, VP-021)
    m = re.search(r"VP[\s_-]*0*(\d+)", s, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))

    # 2) filename starts with digits (62_Fin...)
    m = re.match(r"\s*0*(\d+)", s)
    if m:
        return int(m.group(1))

    # 3) path contains /62_... (…/females/62_Fin...)
    m = re.search(r"[\\/]\s*0*(\d+)_", s)
    if m:
        return int(m.group(1))

    return None
