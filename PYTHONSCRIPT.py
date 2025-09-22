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

# -----------------------
# FFmpeg installer
# -----------------------
def ensure_ffmpeg():
    """Ensure ffmpeg is installed and return the path to ffmpeg executable."""
    if shutil.which("ffmpeg"):
        return "ffmpeg"

    system = platform.system().lower()
    print(f"⚠️ ffmpeg not found, installing for {system}...")

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
        if not subfolders:
            sys.exit("❌ Could not find ffmpeg folder after extraction.")

        ffmpeg_bin = os.path.join(ffmpeg_dir, subfolders[0], "bin", "ffmpeg.exe")
        if not os.path.exists(ffmpeg_bin):
            sys.exit("❌ ffmpeg.exe not found in extracted archive.")

        print(f"✅ ffmpeg ready at: {ffmpeg_bin}")
        return ffmpeg_bin

    elif system == "darwin":
        if shutil.which("brew"):
            subprocess.run(["brew", "install", "ffmpeg"], check=True)
            return "ffmpeg"
        else:
            sys.exit("❌ Homebrew not found. Install ffmpeg manually.")

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
fs = 96000          # sample rate
mc = 12             # 7.1.4 channels
REFLECTION_GAIN = 0.7
c = 343.0           # speed of sound m/s
MAX_DISTANCE_FOR_FILTER = 20.0

# -----------------------
# Input
# -----------------------
music_url = input("Enter the URL of the music file: ").strip()
video_file = input("Enter the path or URL of the video file (leave blank if none): ").strip()

music_file = "music.wav"
wav_file = "music_96k.wav"
multichannel_wav = "3d_music_7_1_4_reflections.wav"
binaural_wav = "3d_music_binaural_reflections.wav"

print(f"Music URL: {music_url}")
print(f"Video file: {video_file if video_file else 'None'}")

# -----------------------
# Download and prepare music
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
# Room and listener
# -----------------------
room = {'x': 8.0, 'y': 6.0, 'z': 3.2}
rx, ry, rz = room['x']/2, room['y']/2, room['z']/2
planes = [
    {'p0': np.array([rx, 0, 0]), 'n': np.array([1, 0, 0])},
    {'p0': np.array([-rx, 0, 0]), 'n': np.array([-1, 0, 0])},
    {'p0': np.array([0, ry, 0]), 'n': np.array([0, 1, 0])},
    {'p0': np.array([0, -ry, 0]), 'n': np.array([0, -1, 0])},
    {'p0': np.array([0, 0, -rz]), 'n': np.array([0, 0, 1])},
    {'p0': np.array([0, 0, +rz]), 'n': np.array([0, 0, -1])}, # Fixed duplicate z plane
]
listener = np.array([0.0, 0.0, 0.15])

# -----------------------
# Speaker gains
# -----------------------
def speaker_gains_7_1_4(az, el):
    FL = max(0, math.cos(az))*max(0, math.cos(el))
    FR = max(0, math.cos(-az))*max(0, math.cos(el))
    C  = max(0, math.cos(el))*0.5
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
def generate_trajectory(samples, room_size=(8,6,3.2)):
    t = np.linspace(0,1,samples)
    x = np.sin(2*math.pi*0.1*t)*room_size[0]/2
    y = np.cos(2*math.pi*0.05*t)*room_size[1]/2
    z = np.sin(2*math.pi*0.08*t)*room_size[2]/2
    return np.stack([x, y, z], axis=1)
traj = generate_trajectory(total_samples)

# -----------------------
# Filters & reflection helpers
# -----------------------
def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5*fs
    normal_cutoff = max(1e-6, min(0.9999, cutoff/nyq)) # Clip to valid range (0, 1)
    b, a = butter(order, normal_cutoff, btype='low')
    return b, a

def apply_lowpass(sig, cutoff):
    cutoff = max(cutoff, 200.0)
    b,a = butter_lowpass(cutoff, fs)
    return lfilter(b,a,sig)

def reflect_point_across_plane(s, p0, n):
    return s - 2*np.dot(s-p0, n)*n

def dist(a,b):
    return np.linalg.norm(a-b)

# -----------------------
# Process frames
# -----------------------
window_size = 2048
hop = 512

for i in range(0, total_samples, hop):
    frame = samples[i:i+window_size]
    if len(frame)==0:
        continue

    # Direct sound
    s_pos = traj[i]
    d = dist(listener, s_pos)+1e-6
    vec = listener - s_pos
    az = math.atan2(vec[1], vec[0])
    el = math.asin(vec[2]/d)
    gains = speaker_gains_7_1_4(az, el)
    att = max(0.05, min(1.0, 1.0/(1.0+0.05*d)))
    out_mc[i:i+len(frame),:] += frame[:,None]*gains*att

    # Reflections
    for plane in planes:
        s_ref = reflect_point_across_plane(s_pos, plane['p0'], plane['n'])
        d_ref = dist(listener, s_ref)+1e-6
        time_delay_s = (d_ref-d)/c
        delay_samples = max(0,int(round(time_delay_s*fs)))

        # Lowpass and gains
        cutoff_freq = fs*(1.0 - np.clip(d_ref/MAX_DISTANCE_FOR_FILTER,0.0,0.9))
        filtered_frame = apply_lowpass(frame, cutoff_freq)
        az_r = math.atan2(listener[1]-s_ref[1], listener[0]-s_ref[0])
        el_r = math.asin((listener[2]-s_ref[2])/d_ref)
        gains_r = speaker_gains_7_1_4(az_r, el_r)
        refl_att = REFLECTION_GAIN/(1+0.05*d_ref)

        # Add delayed and filtered reflection to the output
        add_length = min(len(filtered_frame), total_samples - (i + delay_samples))
        if add_length > 0:
            end_idx = i+delay_samples+add_length
            if end_idx>out_mc.shape[0]:
                pad_amount = end_idx - out_mc.shape[0]
                out_mc = np.pad(out_mc, ((0,pad_amount),(0,0)),'constant')
            out_mc[i+delay_samples:i+delay_samples+add_length,:] += filtered_frame[:add_length,None]*gains_r*refl_att


# -----------------------
# Output format
# -----------------------
print("\nChoose output format:")
print("1. 96 kHz, 32-bit float (original)")
print("2. 42 kHz, 32-bit float (resampled)")
choice = input("Enter 1 or 2: ").strip()
if choice=="1":
    TARGET_FS=96000
    BIT_DEPTH=np.float32
    suffix="96k_32bit"
elif choice=="2":
    TARGET_FS=42000
    BIT_DEPTH=np.float32
    suffix="42k_32bit"
else:
    TARGET_FS=42000
    BIT_DEPTH=np.float32
    suffix="42k_32bit"

# -----------------------
# Resample multichannel
# -----------------------
num_samples_target = int(out_mc.shape[0]*TARGET_FS/fs)
ratio = Fraction(TARGET_FS, fs).limit_denominator()
out_mc_resampled = np.zeros((num_samples_target, mc), dtype=np.float32)

for ch in range(mc):
    out_mc_resampled[:, ch] = resample_poly(out_mc[:, ch], ratio.numerator, ratio.denominator)

out_mc_resampled /= np.max(np.abs(out_mc_resampled)) + 1e-12
mc_filename = f"3d_music_7_1_4_reflections_{suffix}.wav"
write(mc_filename, TARGET_FS, out_mc_resampled.astype(BIT_DEPTH))

# -----------------------
# Stereo downmix & resample
# -----------------------
stereo = np.zeros((out_mc.shape[0],2), dtype=np.float32)
stereo[:,0] = np.mean(out_mc[:,[0,4,6,8,10]],axis=1)+out_mc[:,2]*0.5+out_mc[:,3]*0.3
stereo[:,1] = np.mean(out_mc[:,[1,5,7,9,11]],axis=1)+out_mc[:,2]*0.5+out_mc[:,3]*0.3
stereo /= np.max(np.abs(stereo))+1e-12

stereo_resampled = np.zeros((num_samples_target,2), dtype=np.float32)
for ch in range(2):
    stereo_resampled[:,ch] = resample_poly(stereo[:,ch], TARGET_FS, fs)
stereo_resampled /= np.max(np.abs(stereo_resampled))+1e-12
stereo_filename = f"3d_music_binaural_reflections_{suffix}.wav"
write(stereo_filename, TARGET_FS, stereo_resampled.astype(BIT_DEPTH))

print(f"✅ Multichannel saved: {mc_filename}")
print(f"✅ Stereo saved: {stereo_filename}")

# -----------------------
# Merge audio with video
# -----------------------
def merge_audio(video_file, audio_file, ffmpeg_path, output_video=None):
    if output_video is None:
        output_video = os.path.splitext(video_file)[0]+"_7_1_4.mp4"
    print("🎬 Merging audio with video...")
    cmd = [ffmpeg_path, "-y", "-i", video_file, "-i", audio_file, "-map","0:v","-map","1:a",
           "-c:v","copy","-c:a","pcm_s24le", output_video]
    subprocess.run(cmd, check=True)
    print(f"✅ Output video: {output_video}")
    return output_video

def display_audio_video_links(video_file=None):
    from IPython import get_ipython
    def is_notebook():
        try:
            shell = get_ipython().__class__.__name__
            return shell=='ZMQInteractiveShell'
        except:
            return False
    if is_notebook():
        if video_file and os.path.exists(video_file):
            answer=input(f"Display video {os.path.basename(video_file)}? (y/n): ").strip().lower()
            if answer=='y':
                from IPython.display import Video
                display(Video(video_file, embed=True))
        if os.path.exists(stereo_filename):
            display(Audio(stereo_filename, rate=TARGET_FS))
            with open(stereo_filename,"rb") as f:
                b64=base64.b64encode(f.read()).decode()
            display(HTML(f'<a download="{os.path.basename(stereo_filename)}" href="data:audio/wav;base64,{b64}">⬇️ Download stereo WAV</a>'))
        if os.path.exists(mc_filename):
            with open(mc_filename,"rb") as f:
                b64_mc=base64.b64encode(f.read()).decode()
            display(HTML(f'<a download="{os.path.basename(mc_filename)}" href="data:audio/wav;base64,{b64_mc}">⬇️ Download 7.1.4 multichannel WAV</a>'))
    else:
        print(f"Stereo WAV: {stereo_filename}, Multichannel WAV: {mc_filename}")
        if video_file:
            print(f"Video path: {video_file}")

# -----------------------
# Main execution
# -----------------------
ffmpeg_path = ensure_ffmpeg()
output_video = None
if video_file and os.path.exists(video_file):
    if os.path.exists(mc_filename):
        output_video = merge_audio(video_file, mc_filename, ffmpeg_path)
    else:
        print(f"⚠️ Multichannel audio {mc_filename} not found, skipping video merge.")
display_audio_video_links(output_video)

# -----------------------
# Cleanup
# -----------------------
def cleanup_old_music_and_video(music_files, old_video_file=None):
    files_to_check = [f for f in music_files if os.path.exists(f)]
    if old_video_file and os.path.exists(old_video_file):
        files_to_check.append(old_video_file)
    if not files_to_check:
        return
    print("\n⚠️ Original files can be removed:")
    for f in files_to_check:
        print(f" - {f}")
    answer=input("Delete these original files? (y/n): ").strip().lower()
    if answer=='y':
        for f in files_to_check:
            try:
                os.remove(f)
                print(f"✅ Deleted {f}")
            except Exception as e:
                print(f"❌ Failed to delete {f}: {e}")
    else:
        print("Skipped deletion.")

cleanup_old_music_and_video(["music.wav","music_96k.wav"], video_file)



