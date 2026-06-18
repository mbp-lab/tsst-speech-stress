from pydub import AudioSegment
import os
import numpy as np
import soundfile as sf
import librosa

def merge_intervals(intervals, eps_ms=50):
    
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e + eps_ms:  
            merged[-1][1] = max(last_e, e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]

def subtract_intervals(keep, cut, eps_ms=0):

    if not keep:
        return []
    if not cut:
        return keep

    cut = [(max(0, s - eps_ms), e + eps_ms) for s, e in cut]
    cut = merge_intervals(cut, eps_ms=0)

    out = []
    j = 0
    for ks, ke in keep:
        cur = ks
        while j < len(cut) and cut[j][1] <= ks:
            j += 1
        jj = j
        while jj < len(cut) and cut[jj][0] < ke:
            cs, ce = cut[jj]
            if cs > cur:
                out.append((cur, min(cs, ke)))
            cur = max(cur, ce)
            if cur >= ke:
                break
            jj += 1
        if cur < ke:
            out.append((cur, ke))
    # kleine Fragmente entfernen 
    out = [(s, e) for s, e in out if e > s]
    return out


def cut_segment(input_path, output_path):
    audio = AudioSegment.from_wav(input_path)

    start_ms = 7 * 60 * 1000          
    end_ms = start_ms + (9 * 60 * 1000)  
    trimmed_audio = audio[start_ms:end_ms]

    trimmed_audio.export(output_path, format="wav")
    return output_path



def get_vp_speaker(diarization_output):
    
    # finde speaker mit längster Sprechdauer
    speaker_durations = {}
    for turn, speaker in diarization_output.speaker_diarization:
        duration = turn.end - turn.start
        if speaker in speaker_durations:
            speaker_durations[speaker] += duration
        else:
            speaker_durations[speaker] = duration
    
    vp_speaker = max(speaker_durations, key=speaker_durations.get)
    return vp_speaker

def get_vp_speaker_nemo(segments):
    speaker_durations = {}
    for seg in segments:
        start_s, end_s, speaker = seg.split()
        start_s, end_s = float(start_s), float(end_s)
        duration = end_s - start_s
        if speaker in speaker_durations:
            speaker_durations[speaker] += duration
        else:
            speaker_durations[speaker] = duration
    
    vp_speaker = max(speaker_durations, key=speaker_durations.get)
    return vp_speaker


def cut_only_vp_parts_and_remove_overlaps(audio_path, diarization_output, vp_speaker, out_path, eps_ms=50):

    audio = AudioSegment.from_file(audio_path)
    total_ms = len(audio)

    vp_intervals = []
    other_intervals = []

    for turn, speaker in diarization_output.speaker_diarization:
        start_ms = max(0, int(turn.start * 1000))
        end_ms = min(total_ms, int(turn.end * 1000))
        if end_ms <= start_ms:
            continue

        if speaker == vp_speaker:
            vp_intervals.append((start_ms, end_ms))
        else:
            other_intervals.append((start_ms, end_ms))

    vp_intervals = merge_intervals(vp_intervals, eps_ms=eps_ms)
    other_intervals = merge_intervals(other_intervals, eps_ms=eps_ms)

    clean_vp = subtract_intervals(vp_intervals, other_intervals, eps_ms=eps_ms)

    result = AudioSegment.empty()
    for s, e in clean_vp:
        result += audio[s:e]

    result.export(out_path, format="wav")
    return out_path, clean_vp


def cut_only_vp_parts_and_remove_overlaps_nemo(audio_path, segments, vp_speaker, out_path, eps_ms=None):

    audio = AudioSegment.from_file(audio_path)
    total_ms = len(audio)

    vp_intervals = []
    other_intervals = []

    for seg in segments:
        
        start_s, end_s, speaker = seg.split()
        start_s, end_s = float(start_s), float(end_s)

        start_ms = max(0, int(start_s * 1000))
        end_ms = min(total_ms, int(end_s * 1000))
        if end_ms <= start_ms:
            continue

        if speaker == vp_speaker:
            vp_intervals.append((start_ms, end_ms))
        else:
            other_intervals.append((start_ms, end_ms))

    vp_intervals = merge_intervals(vp_intervals, eps_ms=eps_ms)
    other_intervals = merge_intervals(other_intervals, eps_ms=eps_ms)

    clean_vp = subtract_intervals(vp_intervals, other_intervals, eps_ms=eps_ms)

    result = AudioSegment.empty()
    for s, e in clean_vp:
        result += audio[s:e]

    result.export(out_path, format="wav")
    return out_path, clean_vp


