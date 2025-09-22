import os
import numpy as np
import requests
import base64
from scipy.io.wavfile import write, read
from pydub import AudioSegment
from scipy.signal import butter, lfilter, resample_poly
from IPython.display import Audio, HTML, display
import math
import subprocess
import sys
import platform
import shutil
from fractions import Fraction

try:
    import py7zr
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "py7zr"], check=True)
    import py7zr

try:
    import ffmpeg
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "ffmpeg-python"], check=True)
    import ffmpeg

# -----------------------
# FFmpeg installer
# -----------------------
def ensure_ffmpeg():
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    system = platform.system().lower()
    if system == "windows":
        url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-full.7z"
        ffmpeg_dir = os.path.abspath("FFMPEG")
        os.makedirs(ffmpeg_dir, exist_ok=True)
        archive_path = os.path.join(ffmpeg_dir, "ffmpeg.7z")
        if not os.path.exists(archive_path):
            print("⬇️ Downloading ffmpeg...")
            r = requests.get(url, stream=True)
            with open(archive_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("✅ Download complete.")
        print("📦 Extracting ffmpeg...")
        with py7zr.SevenZipFile(archive_path, mode="r") as z:
            z.extractall(path=ffmpeg_dir)
        subfolders = [d for d in os.listdir(ffmpeg_dir) if os.path.isdir(os.path.join(ffmpeg_dir, d))]
        ffmpeg_bin = os.path.join(ffmpeg_dir, subfolders[0], "bin", "ffmpeg.exe")
        return ffmpeg_bin
    elif system == "darwin":
        subprocess.run(["brew","install","ffmpeg"], check=True)
        return "ffmpeg"
    elif system == "linux":
        if shutil.which("apt"):
            subprocess.run(["sudo","apt","update"], check=True)
            subprocess.run(["sudo","apt","install","-y","ffmpeg"], check=True)
        return "ffmpeg"
    else:
        sys.exit(f"❌ Unsupported OS: {system}")

# -----------------------
# Parameters
# -----------------------
fs = 96000
mc = 12
REFLECTION_GAIN_START = 0.9
c = 343.0

# -----------------------
# Filters & reflection helpers
# -----------------------
def butter_lowpass(cutoff, fs, order=2):
    nyq = 0.5*fs
    normal_cutoff = max(1e-6, min(0.99, cutoff/nyq))
    b,a = butter(order, normal_cutoff, btype='low')
    return b,a

def apply_lowpass(sig, cutoff):
    b,a = butter_lowpass(cutoff, fs)
    return lfilter(b,a,sig)

def reflect_point_across_plane(s, p0, n):
    return s - 2*np.dot(s-p0,n)*n

def dist(a,b):
    return np.linalg.norm(a-b)

# -----------------------
# Adaptive reflection processing
# -----------------------
def process_reflections(frame, s_pos, listener, planes, out_mc, fs, idx, room_diag, current_refl_gain):
    for plane in planes:
        s_ref = reflect_point_across_plane(s_pos, plane['p0'], plane['n'])
        d_ref = dist(listener, s_ref)+1e-6
        d_direct = dist(listener, s_pos)+1e-6
        refl_att = current_refl_gain / (1 + 0.3*d_ref**1.5)
        norm_dist = min(1.0, d_ref/room_diag)
        cutoff_freq = fs*(0.45 - 0.35*norm_dist)
        cutoff_freq = np.clip(cutoff_freq, fs*0.1, fs*0.45)
        filtered_frame = apply_lowpass(frame, cutoff_freq)
        delay_samples = max(0,int(round((d_ref-d_direct)/c*fs)))
        vec_ref = listener - s_ref
        d_ref_norm = np.linalg.norm(vec_ref)
        if d_ref_norm > 1e-6:
            vec_ref /= d_ref_norm
            az_r = math.atan2(vec_ref[1], vec_ref[0])
            el_r = math.asin(np.clip(vec_ref[2],-1,1))
            gains_r = speaker_gains_7_1_4(az_r, el_r)
        else:
            gains_r = np.zeros(mc)
        start_idx = idx + delay_samples
        add_length = min(len(filtered_frame), out_mc.shape[0]-start_idx)
        if add_length>0 and start_idx<out_mc.shape[0]:
            out_mc[start_idx:start_idx+add_length,:] += (filtered_frame[:add_length,None]*gains_r*refl_att)
    return out_mc

# -----------------------
# Speaker gains
# -----------------------
def speaker_gains_7_1_4(az,el):
    FL = max(0,np.cos(az-np.pi/4))*max(0,np.cos(el))
    FR = max(0,np.cos(az+np.pi/4))*max(0,np.cos(el))
    C  = max(0,np.cos(az))*max(0,np.cos(el))*0.7
    RL = max(0,np.cos(az-3*np.pi/4))*max(0,np.cos(el))*0.8
    RR = max(0,np.cos(az+3*np.pi/4))*max(0,np.cos(el))*0.8
    SL = max(0,np.cos(az-np.pi/2))*max(0,np.cos(el))*0.6
    SR = max(0,np.cos(az+np.pi/2))*max(0,np.cos(el))*0.6
    LFE = 0.15
    height_factor = max(0.1,np.sin(el)+0.3)
    FHL = max(0,np.cos(az-np.pi/4))*height_factor
    FHR = max(0,np.cos(az+np.pi/4))*height_factor
    RHL = max(0,np.cos(az-3*np.pi/4))*height_factor
    RHR = max(0,np.cos(az+3*np.pi/4))*height_factor
    gains = np.array([FL,FR,C,LFE,SL,SR,RL,RR,FHL,FHR,RHL,RHR],dtype=np.float32)
    total = np.sum(gains)+1e-12
    gains = gains/max(total,1.0)
    return gains

# -----------------------
# Input music
# -----------------------
music_url = input("Enter the URL of the music file: ").strip()
video_file = input("Enter the path or URL of the video file (leave blank if none): ").strip()
music_file = "music.wav"
processed_music_file = "processed_music.wav"
wav_file = "music_96k.wav"
multichannel_wav = "3d_music_7_1_4_reflections.wav"
binaural_wav = "3d_music_binaural_reflections.wav"

if not os.path.exists(music_file) or os.path.getsize(music_file)==0:
    with requests.get(music_url,stream=True,timeout=30) as r:
        r.raise_for_status()
        r.raw.decode_content=True
        with open(music_file,"wb") as f:
            shutil.copyfileobj(r.raw,f)

ffmpeg_path = ensure_ffmpeg()
(
    ffmpeg.input(music_file)
    .output(processed_music_file, acodec='pcm_s16le', ar=str(fs), ac=1)
    .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
)
music = AudioSegment.from_file(processed_music_file).set_frame_rate(fs).set_channels(1)
music.export(wav_file,'wav')
sr,samples = read(wav_file)
samples = samples.astype(np.float32)
if samples.dtype.kind=='i':
    samples/=float(2**(samples.dtype.itemsize*8-1))
samples *= 0.2

total_samples = len(samples)
out_mc = np.zeros((total_samples, mc), dtype=np.float32)

# -----------------------
# Room & listener
# -----------------------
room = {'x':8.0,'y':6.0,'z':3.2}
rx,ry,rz = room['x']/2,room['y']/2,room['z']/2
planes = [
    {'p0': np.array([rx,0,0]), 'n': np.array([-1,0,0])},
    {'p0': np.array([-rx,0,0]), 'n': np.array([1,0,0])},
    {'p0': np.array([0,ry,0]), 'n': np.array([0,-1,0])},
    {'p0': np.array([0,-ry,0]), 'n': np.array([0,1,0])},
    {'p0': np.array([0,0,-rz]), 'n': np.array([0,0,1])},
    {'p0': np.array([0,0,rz]), 'n': np.array([0,0,-1])},
]
listener = np.array([0.0,0.0,0.15])
room_diag = np.linalg.norm([room['x'],room['y'],room['z']])

# -----------------------
# Linear scaling helpers
# -----------------------
def generate_trajectory(samples, room_size=(8,6,3.2)):
    t = np.linspace(0,1,samples)
    # Linearly increasing offset from listener
    offset_scale = 0.5 + 0.5*t  # 0.5 m start, grows to 1 m
    x = np.sin(2*np.pi*0.05*t)*room_size[0]*0.3*offset_scale
    y = np.cos(2*np.pi*0.03*t)*room_size[1]*0.3*offset_scale
    z = np.sin(2*np.pi*0.04*t)*room_size[2]*0.2*offset_scale + room_size[2]*0.1
    return np.stack([x,y,z],axis=1)

traj = generate_trajectory(total_samples)

# Linearly scaling frame window and hop
window_size_min, window_size_max = 2048,8192
hop_min, hop_max = 512,2048
def get_window_hop(i,total_samples):
    factor = i/total_samples
    window = int(window_size_min + factor*(window_size_max-window_size_min))
    hop = int(hop_min + factor*(hop_max-hop_min))
    return window, hop

# -----------------------
# Process frames
# -----------------------
current_refl_gain = REFLECTION_GAIN_START
print("Processing audio frames...")
i = 0
while i<total_samples:
    window, hop = get_window_hop(i,total_samples)
    frame = samples[i:i+window]
    if len(frame)==0: break

    s_pos = traj[i]
    d = dist(listener, s_pos)+1e-6
    vec = listener - s_pos
    vec_norm = vec/d
    az = math.atan2(vec_norm[1],vec_norm[0])
    el = math.asin(np.clip(vec_norm[2],-1,1))
    gains = speaker_gains_7_1_4(az,el)
    att = max(0.1,min(1.0,1.0/(1.0+0.1*d)))
    end_idx = min(i+len(frame),total_samples)
    out_mc[i:end_idx,:] += frame[:end_idx-i,None]*gains*att

    out_mc = process_reflections(frame, s_pos, listener, planes, out_mc, fs, i, room_diag, current_refl_gain)

    # Adaptive gain
    current_peak = np.max(np.abs(out_mc[i:end_idx,:]))
    if current_peak>0.8:
        current_refl_gain *= 0.95
        current_refl_gain = max(current_refl_gain,0.1)
    i+=hop

# -----------------------
# Normalize early and full audio
# -----------------------
max_peak = np.max(np.abs(out_mc))
if max_peak>0:
    out_mc = out_mc/max_peak*0.9

# -----------------------
# Resample and output
# -----------------------
TARGET_FS = 96000
BIT_DEPTH = np.float32
mc_filename = f"3d_music_7_1_4_reflections_{TARGET_FS//1000}k_32bit.wav"
write(mc_filename,TARGET_FS,out_mc.astype(BIT_DEPTH))

# Stereo downmix
stereo = np.zeros((out_mc.shape[0],2),dtype=np.float32)
stereo[:,0] = (out_mc[:,0]*0.7 + out_mc[:,4]*0.5 + out_mc[:,6]*0.4 +
               out_mc[:,2]*0.3 + out_mc[:,3]*0.2 + out_mc[:,8]*0.3 + out_mc[:,10]*0.3)
stereo[:,1] = (out_mc[:,1]*0.7 + out_mc[:,5]*0.5 + out_mc[:,7]*0.4 +
               out_mc[:,2]*0.3 + out_mc[:,3]*0.2 + out_mc[:,9]*0.3 + out_mc[:,11]*0.3)
peak_stereo = np.max(np.abs(stereo))
if peak_stereo>0:
    stereo = stereo/peak_stereo*0.9
stereo_filename = f"3d_music_binaural_reflections_{TARGET_FS//1000}k_32bit.wav"
write(stereo_filename,TARGET_FS,stereo.astype(BIT_DEPTH))

print(f"✅ Multichannel saved: {mc_filename}")
print(f"✅ Stereo saved: {stereo_filename}")
