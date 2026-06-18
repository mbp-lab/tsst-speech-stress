#Measure pitch of all wav files in directory
import glob
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import os
import librosa
import opensmile
import glob
import re

# This is the function to measure voice pitch
def measurePitch(voiceID, f0min, f0max, unit):
    sound = parselmouth.Sound(voiceID) # read the sound
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max) #create a praat pitch object
    meanF0 = call(pitch, "Get mean", 0, 0, unit) # get mean pitch
    pitch = sound.to_pitch(pitch_floor=f0min, pitch_ceiling=f0max)  # Same as you above, but nicer Python syntax
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values==0] = np.nan
    median_pitch = np.nanmedian(pitch_values)


    stdevF0 = call(pitch, "Get standard deviation", 0 ,0, unit) # get standard deviation
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    

    return meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer, median_pitch


def praat_analysis(input_path, output_filename, f0min, f0max):
        
   
    file_list = []
    mean_F0_list = []
    sd_F0_list = []
    hnr_list = []
    localJitter_list = []
    localabsoluteJitter_list = []
    rapJitter_list = []
    ppq5Jitter_list = []
    ddpJitter_list = []
    localShimmer_list = []
    localdbShimmer_list = []
    apq3Shimmer_list = []
    aqpq5Shimmer_list = []
    apq11Shimmer_list = []
    ddaShimmer_list = []
    median_pitch_list = []

    # Go through all the wave files in the folder and measure pitch
    for wave_file in glob.glob(input_path):
        (meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, 
         ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, 
         aqpq5Shimmer, apq11Shimmer, ddaShimmer, median_pitch) = measurePitch(wave_file, f0min, f0max, "Hertz")

        file_list.append(wave_file) # make an ID list
        mean_F0_list.append(meanF0) # make a mean F0 list
        sd_F0_list.append(stdevF0) # make a sd F0 list
        hnr_list.append(hnr)
        localJitter_list.append(localJitter)
        localabsoluteJitter_list.append(localabsoluteJitter)
        rapJitter_list.append(rapJitter)
        ppq5Jitter_list.append(ppq5Jitter)
        ddpJitter_list.append(ddpJitter)
        localShimmer_list.append(localShimmer)
        localdbShimmer_list.append(localdbShimmer)
        apq3Shimmer_list.append(apq3Shimmer)
        aqpq5Shimmer_list.append(aqpq5Shimmer)
        apq11Shimmer_list.append(apq11Shimmer)
        ddaShimmer_list.append(ddaShimmer)
        median_pitch_list.append(median_pitch)

    df = pd.DataFrame(np.column_stack([file_list, mean_F0_list, 
                                       sd_F0_list, hnr_list, localJitter_list, localabsoluteJitter_list,
                                       rapJitter_list, ppq5Jitter_list, ddpJitter_list, localShimmer_list,
                                       localdbShimmer_list, apq3Shimmer_list, aqpq5Shimmer_list, 
                                       apq11Shimmer_list, ddaShimmer_list, median_pitch_list]),
                                   columns=['voiceID', 'meanF0Hz', 'stdevF0Hz', 'HNR',
                                            'localJitter', 'localabsoluteJitter', 'rapJitter', 
                                            'ppq5Jitter', 'ddpJitter', 'localShimmer', 
                                            'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer', 
                                            'apq11Shimmer', 'ddaShimmer', 'median_pitch'])  #add these lists to pandas in the right 

    # Write out the updated dataframe
    df.to_csv(output_filename, index=False)



    
def get_librosa_features(main_dir, output_filename):
    
    main_dir=os.path.join(main_dir)
    df=pd.DataFrame() 
    print(os.listdir(main_dir))
        
    #for i, audio_dir in enumerate(os.listdir(main_dir)):
    for subdir, dirs, files in os.walk(main_dir):
        for file in files:
            #print os.path.join(subdir, file)
            path = subdir + os.sep + file
            print (path)
            
            data, sampling_rate = librosa.load(path)
            print(sampling_rate)
            rmse = librosa.feature.rms(y=data) 
            vpn_df = pd.DataFrame(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T) 
           
            print (len(rmse.T))
            print (len(vpn_df))
            
            vpn_df['rmse']=rmse.T
            vpn_df['length']=len(vpn_df)
            vpn_df['vpn']=str(file)
          
            filename=vpn_df.vpn.str[0:4]
            df=pd.concat([df, vpn_df], axis=0)

    df.to_csv(output_filename, index=False)


def get_opensmile_features_regex(input_dir_path, output_filename='opensmile_features.csv'):
    #Neue version holt sich die VPN Nummer mit regex, damit es flexibler ist 

    smile = opensmile.Smile(
        #feature_set=opensmile.FeatureSet.eGeMAPSv01b,
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
        num_channels=1
    )
    df=pd.DataFrame()

    #main_dir='./final_processed_wav'
    main_dir=input_dir_path

    main_dir=os.path.join(main_dir)
    print(os.listdir(main_dir))

    for subdir, dirs, files in os.walk(main_dir):
        for file in files:
            path = subdir + os.sep + file
            vpn_df = pd.DataFrame(smile.process_file(path)).reset_index()

            print(len(vpn_df))

            folder = os.path.basename(subdir)
            if folder == 'males':
                m = re.search(r'\d+', file)
                print(m)
                vpn_df["vpn"] = _vpnum(file)


            if folder == 'females':
                m = re.search(r'\d+', file)
                print(m)
                vpn_df["vpn"] = _vpnum(file)

            df=pd.concat([df, vpn_df], axis=0)

    df.to_csv(output_filename, index=False)



def _vpnum(x):
    if pd.isna(x):
        return None
    s = str(x)

    m = re.search(r"VP(\d+)", s, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))

    m = re.search(r"^(\d+)", s)  # fallback: führende Zahl
    return int(m.group(1)) if m else None


def merge_all_df_together(filename_praat, filename_librosa, filename_behavior, filename_output):
    """
    Merges all tables using the participant id extracted from:
      - librosa:  column 'vpn' (e.g., 'VP17_TSST.wav')
      - praat:   column 'voiceID' (path containing 'VPxx')
      - behavior: column 'Subject' (numeric -> 'VPxx')

    Output is saved as filename_output + '.csv' (or exact name if already endswith .csv)
    Returns merged dataframe.
    """

    #load the data
    feature_data = pd.read_csv(filename_librosa, sep=",", encoding="latin-1")
    behav_data = pd.read_csv(filename_behavior, sep=";", encoding="latin-1")

    female_praat = pd.read_csv(filename_praat + "_female.csv", sep=",", encoding="latin-1")
    male_praat = pd.read_csv(filename_praat + "_male.csv", sep=",", encoding="latin-1")

    praat_data = pd.concat([female_praat, male_praat], ignore_index=True)
    praat_data.to_csv(f"{filename_praat}.csv", index=False)

    # make vpn numeric to merge
    feature_data["vpn"] = feature_data["vpn"].apply(_vpnum)
    praat_data["vpn"] = praat_data["voiceID"].apply(_vpnum)
    behav_data["vpn"] = behav_data["Subject"].astype(float).astype("Int64")

    #one row per participant
    feature_agg = (
        feature_data
        .groupby("vpn")
        .mean(numeric_only=True)
        .reset_index()
    )

    praat_agg = (
        praat_data
        .groupby("vpn")
        .mean(numeric_only=True)
        .reset_index()
    )

    # merge all dataframes
    df = behav_data.merge(feature_agg, on="vpn", how="left")
    df = df.merge(praat_agg, on="vpn", how="left")

    out_path = f"{filename_output}.csv"
    df.to_csv(out_path, index=False)

    return df