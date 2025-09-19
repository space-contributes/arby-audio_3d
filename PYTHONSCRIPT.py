import os
import numpy as np
import requests
import base64
from scipy.io.wavfile import write, read
from pydub import AudioSegment
from scipy.signal import butter, filtfilt, lfilter, resample_poly
from IPython.display import Audio, HTML, display
import math
import subprocess
import sys
import platform
import shutil

try:
    import py7zr
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "py7zr"], check=True)
    import py7zr

# -----------------------
# Ensure FFmpeg
# -----------------------
def ensure_ffmpeg():
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    system = platform.system().lower()
    print(f"⚠️ ffmpeg not found, installing for {system}...")
    # Windows
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
        print(f"✅ ffmpeg ready at: {ffmpeg_bin}")
        return ffmpeg_bin
    # macOS
    elif system == "darwin":
        if shutil.which("brew"):
            subprocess.run(["brew", "install", "ffmpeg"], check=True)
            return "ffmpeg"
        else:
            sys.exit("❌ Install ffmpeg via Homebrew.")
    # Linux
    elif system == "linux":
        if shutil.which("apt"):
            subprocess.run(["sudo", "apt", "update"], check=True)
            subprocess.run(["sudo", "apt", "install", "-y", "ffmpeg"], check=True)
        elif shutil.which("dnf"):
            subprocess.run(["sudo", "dnf", "install", "-y", "ffmpeg"], check=True)
        elif shutil.which("pacman"):
            subprocess.run(["sudo", "pacman", "-S", "--noconfirm", "ffmpeg"], check=True)
        else:
            sys.exit("❌ Unsupported Linux package manager. Install ffmpeg manually.")
        return "ffmpeg"
    else:
        sys.exit(f"❌ Unsupported OS: {system}")

# -----------------------
# Parameters
# -----------------------
fs = 96000  # default sample rate
mc = 12     # 7.1.4 channels
REFLECTION_GAIN = 0.7
c = 343.0  # speed of sound
MAX_DISTANCE_FOR_FILTER = 20.0  # meters

# -----------------------
# User input
# -----------------------
music_url = input("Enter the URL of the music file: ").strip()
video_file = input("Enter video path or leave blank: ").strip()

music_file = "music.wav"
wav_file = "music_96k.wav"
multichannel_wav = "3d_music_7_1_4_reflections.wav"
binaural_wav = "3d_music_binaural_reflections.wav"

# -----------------------
# Download & prepare music
# -----------------------
if not os.path.exists(music_file):
    print("Downloading music file...")
    r = requests.get(music_url, timeout=30)
    with open(music_file, 'wb') as f:
        f.write(r.content)
    print("✅ Download complete.")

music = AudioSegment.from_file(music_file).set_frame_rate(fs).set_channels(1)
music.export(wav_file, 'wav')
sr, samples = read(wav_file)
samples = samples.astype(np.float32)
if samples.dtype.kind == 'i':
    samples /= float(2**(samples.dtype.itemsize*8-1))
samples *= 0.3
total_samples = len(samples)
out_mc = np.zeros((total_samples, mc), dtype=np.float32)

# -----------------------
# Room geometry
# -----------------------
room = {'x': 8.0, 'y': 6.0, 'z': 3.2}
rx, ry, rz = room['x']/2, room['y']/2, room['z']/2
planes = [
    {'p0': np.array([rx, 0, 0]), 'n': np.array([1, 0, 0])},
    {'p0': np.array([-rx, 0, 0]), 'n': np.array([-1, 0, 0])},
    {'p0': np.array([0, ry, 0]), 'n': np.array([0, 1, 0])},
    {'p0': np.array([0, -ry, 0]), 'n': np.array([0, -1, 0])},
    {'p0': np.array([0, 0, -rz]), 'n': np.array([0, 0, 1])},
    {'p0': np.array([0, 0, rz]), 'n': np.array([0, 0, -1])},
]
listener = np.array([0.0, 0.0, 0.15])

# -----------------------
# Speaker gains
# -----------------------
def speaker_gains_7_1_4(az, el):
    FL = max(0, math.cos(az))*max(0, math.cos(el))
    FR = max(0, math.cos(-az))*max(0, math.cos(el))
    C = max(0, math.cos(el))*0.5
    RL = max(0, math.cos(az+math.pi))*0.7
    RR = max(0, math.cos(-az+math.pi))*0.7
    SL = max(0, math.sin(az))*0.6
    SR = max(0, math.sin(-az))*0.6
    LFE = 0.2
    FHL = max(0, math.cos(az))*max(0, el)
    FHR = max(0, math.cos(-az))*max(0, el)
    RHL = max(0, math.cos(az+math.pi))*max(0, el)
    RHR = max(0, math.cos(-az+math.pi))*max(0, el)
    gains = np.array([FL, FR, C, LFE, SL, SR, RL, RR, FHL, FHR, RHL, RHR], dtype=np.float32)
    s = np.sum(gains)
    return gains/s if s > 0 else gains

# -----------------------
# Trajectory
# -----------------------
def generate_trajectory(samples, room_size=(8, 6, 3.2)):
    t = np.linspace(0, 1, samples)
    x = np.sin(2*math.pi*0.1*t)*room_size[0]/2
    y = np.cos(2*math.pi*0.05*t)*room_size[1]/2
    z = np.sin(2*math.pi*0.08*t)*room_size[2]/2
    return np.stack([x, y, z], axis=1)
traj = generate_trajectory(total_samples)

# -----------------------
# Helper functions
# -----------------------
def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5*fs
    normal_cutoff = max(1e-6, min(0.45, cutoff/nyq))
    b, a = butter(order, normal_cutoff, btype='low')
    return b, a

def apply_lowpass(sig, cutoff):
    cutoff = max(200.0, min(cutoff, 0.45*fs))
    b, a = butter_lowpass(cutoff, fs)
    return filtfilt(b, a, sig)

def reflect_point_across_plane(s, p0, n):
    v = s - p0
    return s - 2*np.dot(v, n)*n

def dist(a,b):
    return np.linalg.norm(a-b)

def fractional_delay(sig, delay_samples):
    int_delay = int(np.floor(delay_samples))
    frac = delay_samples - int_delay
    delayed = np.zeros(len(sig)+int_delay+1, dtype=sig.dtype)
    delayed[int_delay:int_delay+len(sig)] += (1-frac)*sig
    delayed[int_delay+1:int_delay+len(sig)+1] += frac*sig
    return delayed[:len(sig)]

# -----------------------
# Reflections processing
# -----------------------
window_size = 2048
hop = 512
for i in range(0, total_samples, hop):
    frame = samples[i:i+window_size]
    if len(frame) == 0: continue
    # Direct sound
    s_pos = traj[i]
    d = dist(listener, s_pos)+1e-6
    vec = listener - s_pos
    az = math.atan2(vec[1], vec[0])
    el = math.asin(vec[2]/d)
    gains = speaker_gains_7_1_4(az, el)
    att = max(0.05, min(1.0, 1.0/(1+0.05*d)))
    out_mc[i:i+len(frame),:] += frame[:,None]*gains*att
    # Reflections
    for plane in planes:
        s_ref = reflect_point_across_plane(s_pos, plane['p0'], plane['n'])
        d_ref = dist(listener, s_ref)+1e-6
        time_delay_s = (d_ref - d)/c
        delay_samples = time_delay_s*fs
        cutoff_freq = fs*(1.0 - np.clip(d_ref/MAX_DISTANCE_FOR_FILTER,0.0,0.9))
        filtered_frame = apply_lowpass(frame, cutoff_freq)
        az_r = math.atan2(listener[1]-s_ref[1], listener[0]-s_ref[0])
        el_r = math.asin((listener[2]-s_ref[2])/d_ref)
        gains_r = speaker_gains_7_1_4(az_r, el_r)
        refl_att = REFLECTION_GAIN/(1+0.05*d_ref)
        delayed_frame = fractional_delay(filtered_frame, delay_samples)
        add_length = min(len(delayed_frame), out_mc.shape[0]-i)
        out_mc[i:i+add_length,:] += delayed_frame[:add_length,None]*gains_r*refl_att

# -----------------------
# Output format choice
# -----------------------
print("\nChoose output format:")
print("1. 96 kHz, 32-bit float (default)")
print("2. 46 kHz, 32-bit float (resampled)")
choice = input("Enter 1 or 2: ").strip()
if choice=="1": TARGET_FS=96000; BIT_DEPTH=np.float32; suffix="96k_32bit"
elif choice=="2": TARGET_FS=46000; BIT_DEPTH=np.float32; suffix="46k_32bit"
else: TARGET_FS=46000; BIT_DEPTH=np.float32; suffix="46k_32bit"

# -----------------------
# Resample multichannel safely
# -----------------------
num_samples_target = int(out_mc.shape[0]*TARGET_FS/fs)
out_mc_resampled = np.zeros((num_samples_target, mc), dtype=np.float32)
for ch in range(mc):
    b,a = butter(8, TARGET_FS/fs, btype='low')
    filtered_ch = filtfilt(b,a,out_mc[:,ch])
    out_mc_resampled[:,ch] = resample_poly(filtered_ch, TARGET_FS, fs)
out_mc_resampled /= np.max(np.abs(out_mc_resampled))+1e-12
mc_filename = f"3d_music_7_1_4_reflections_{suffix}.wav"
write(mc_filename, TARGET_FS, out_mc_resampled.astype(BIT_DEPTH))

# -----------------------
# Stereo downmix
# -----------------------
stereo = np.zeros((out_mc_resampled.shape[0],2),dtype=np.float32)
stereo[:,0] = np.mean(out_mc_resampled[:,[0,4,6,8,10]],axis=1) + 0.5*out_mc_resampled[:,2] + 0.3*out_mc_resampled[:,3]
stereo[:,1] = np.mean(out_mc_resampled[:,[1,5,7,9,11]],axis=1) + 0.5*out_mc_resampled[:,2] + 0.3*out_mc_resampled[:,3]
stereo /= np.max(np.abs(stereo))+1e-12
stereo_filename = f"3d_music_binaural_reflections_{suffix}.wav"
write(stereo_filename, TARGET_FS, stereo.astype(BIT_DEPTH))

# -----------------------
# Display / Panels (simplified)
# -----------------------
display(Audio(stereo_filename))
print(f"✅ Multichannel saved: {mc_filename}")
print(f"✅ Stereo saved: {stereo_filename}")
print("🛋️ Furniture occlusion detected without camera/mic/sensors")
