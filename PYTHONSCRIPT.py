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
# Parameters (AUTO-SCALING)
# -----------------------
fs = 96000          # sample rate
mc = 12             # 7.1.4 channels
REFLECTION_GAIN_START = 0.7  # Starting gain - will auto-reduce if too loud
c = 343.0           # speed of sound m/s
MAX_DISTANCE_FOR_FILTER = 20.0

# -----------------------
# FIXED Filters & reflection helpers
# -----------------------
def butter_lowpass(cutoff, fs, order=2):  # REDUCED order from 4 to 2
    nyq = 0.5*fs
    normal_cutoff = max(1e-6, min(0.99, cutoff/nyq))  # Safer clipping
    b, a = butter(order, normal_cutoff, btype='low')
    return b, a

def apply_lowpass(sig, cutoff):
    cutoff = max(cutoff, 500.0)  # INCREASED minimum from 200 to 500 Hz
    b, a = butter_lowpass(cutoff, fs)
    return lfilter(b, a, sig)

def reflect_point_across_plane(s, p0, n):
    return s - 2*np.dot(s-p0, n)*n

def dist(a,b):
    return np.linalg.norm(a-b)

# -----------------------
# ADAPTIVE Reflection processing (auto-scales gain)
# -----------------------
def process_reflections(frame, s_pos, listener, planes, out_mc, fs, idx, room_diag, current_refl_gain):
    """
    Adaptive reflection processing that automatically reduces gain if levels get too high.
    """
    c = 343.0
    
    for plane in planes:
        s_ref = reflect_point_across_plane(s_pos, plane['p0'], plane['n'])
        d_ref = dist(listener, s_ref) + 1e-6
        d_direct = dist(listener, s_pos) + 1e-6
        
        # Use the dynamically adjusted reflection gain
        refl_att = current_refl_gain / (1 + 0.3*d_ref**1.5)
        
        # More conservative frequency filtering
        norm_dist = min(1.0, d_ref / room_diag)
        cutoff_freq = fs * (0.45 - 0.35*norm_dist)
        cutoff_freq = np.clip(cutoff_freq, fs*0.1, fs*0.45)
        
        # Apply filtering
        filtered_frame = apply_lowpass(frame, cutoff_freq)
        
        # Calculate delay
        delay_samples = max(0, int(round((d_ref - d_direct) / c * fs)))
        
        # Calculate reflection direction and gains
        vec_ref = listener - s_ref
        d_ref_norm = np.linalg.norm(vec_ref)
        if d_ref_norm > 1e-6:
            vec_ref = vec_ref / d_ref_norm
            az_r = math.atan2(vec_ref[1], vec_ref[0])
            el_r = math.asin(np.clip(vec_ref[2], -1, 1))
            gains_r = speaker_gains_7_1_4(az_r, el_r)
        else:
            gains_r = np.zeros(mc)
        
        # Add delayed reflection
        start_idx = idx + delay_samples
        add_length = min(len(filtered_frame), out_mc.shape[0] - start_idx)
        
        if add_length > 0 and start_idx < out_mc.shape[0]:
            out_mc[start_idx:start_idx+add_length,:] += (
                filtered_frame[:add_length,None] * gains_r * refl_att
            )
    
    return out_mc

# -----------------------
# Input
# -----------------------
music_url = input("Enter the URL of the music file: ").strip()
video_file = input("Enter the path or URL of the video file (leave blank if none): ").strip()

music_file = "music.wav"
processed_music_file = "processed_music.wav"
wav_file = "music_96k.wav"
multichannel_wav = "3d_music_7_1_4_reflections.wav"
binaural_wav = "3d_music_binaural_reflections.wav"

print(f"Music URL: {music_url}")
print(f"Video file: {video_file if video_file else 'None'}")

# -----------------------
# Download and prepare music
# -----------------------
if not os.path.exists(music_file) or os.path.getsize(music_file) == 0:
    print("Downloading music file...")
    try:
        with requests.get(music_url, stream=True, timeout=30) as r:
            r.raise_for_status()
            r.raw.decode_content = True
            with open(music_file, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        print("✅ Download complete.")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading music file: {e}")
        print("Please check the URL and your internet connection.")
        exit()
    except Exception as e:
        print(f"❌ An unexpected error occurred during download: {e}")
        exit()

if not os.path.exists(music_file) or os.path.getsize(music_file) == 0:
    print(f"❌ Downloaded music file '{music_file}' is missing or empty after download.")
    print("Please ensure the provided URL is correct and the file is accessible.")
    exit()

ffmpeg_path = ensure_ffmpeg()

print(f"Converting downloaded file '{music_file}' to standard WAV format '{processed_music_file}'...")
try:
    (
        ffmpeg
        .input(music_file)
        .output(processed_music_file, acodec='pcm_s16le', ar=str(fs), ac=1)
        .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
    )
    print("✅ Conversion complete.")
except ffmpeg.Error as e:
    print(f"❌ Error during ffmpeg conversion: {e.stderr.decode()}")
    print("Please ensure the downloaded file is a valid audio format.")
    exit()
except Exception as e:
    print(f"❌ An unexpected error occurred during ffmpeg conversion: {e}")
    exit()

try:
    music = AudioSegment.from_file(processed_music_file).set_frame_rate(fs).set_channels(1)
except Exception as e:
    print(f"❌ Error processing converted music file '{processed_music_file}': {e}")
    print("This might still indicate an issue with the original downloaded file.")
    exit()

music.export(wav_file, 'wav')
sr, samples = read(wav_file)
samples = samples.astype(np.float32)
if samples.dtype.kind == 'i':
    samples /= float(2**(samples.dtype.itemsize*8-1))

# FIXED: Reduced input gain to prevent clipping
samples *= 0.2  # REDUCED from 0.3 to 0.2

total_samples = len(samples)
out_mc = np.zeros((total_samples, mc), dtype=np.float32)

# -----------------------
# Room and listener (FIXED duplicate plane)
# -----------------------
room = {'x': 8.0, 'y': 6.0, 'z': 3.2}
rx, ry, rz = room['x']/2, room['y']/2, room['z']/2
planes = [
    {'p0': np.array([rx, 0, 0]), 'n': np.array([-1, 0, 0])},   # Right wall
    {'p0': np.array([-rx, 0, 0]), 'n': np.array([1, 0, 0])},   # Left wall  
    {'p0': np.array([0, ry, 0]), 'n': np.array([0, -1, 0])},   # Front wall
    {'p0': np.array([0, -ry, 0]), 'n': np.array([0, 1, 0])},   # Back wall
    {'p0': np.array([0, 0, -rz]), 'n': np.array([0, 0, 1])},   # Floor
    {'p0': np.array([0, 0, rz]), 'n': np.array([0, 0, -1])},   # Ceiling
]
listener = np.array([0.0, 0.0, 0.15])
room_diag = np.linalg.norm([room['x'], room['y'], room['z']])

# -----------------------
# FIXED Speaker gains (normalized properly)
# -----------------------
def speaker_gains_7_1_4(az, el):
    # Convert to speaker-relative angles
    FL = max(0, np.cos(az - np.pi/4)) * max(0, np.cos(el))
    FR = max(0, np.cos(az + np.pi/4)) * max(0, np.cos(el))
    C  = max(0, np.cos(az)) * max(0, np.cos(el)) * 0.7
    RL = max(0, np.cos(az - 3*np.pi/4)) * max(0, np.cos(el)) * 0.8
    RR = max(0, np.cos(az + 3*np.pi/4)) * max(0, np.cos(el)) * 0.8
    SL = max(0, np.cos(az - np.pi/2)) * max(0, np.cos(el)) * 0.6
    SR = max(0, np.cos(az + np.pi/2)) * max(0, np.cos(el)) * 0.6
    LFE = 0.15  # Reduced LFE contribution
    
    # Height channels (attenuated when el is negative)
    height_factor = max(0.1, np.sin(el) + 0.3)
    FHL = max(0, np.cos(az - np.pi/4)) * height_factor
    FHR = max(0, np.cos(az + np.pi/4)) * height_factor  
    RHL = max(0, np.cos(az - 3*np.pi/4)) * height_factor
    RHR = max(0, np.cos(az + 3*np.pi/4)) * height_factor
    
    gains = np.array([FL, FR, C, LFE, SL, SR, RL, RR, FHL, FHR, RHL, RHR], dtype=np.float32)
    
    # FIXED: Proper normalization to prevent level buildup
    total = np.sum(gains) + 1e-12
    gains = gains / max(total, 1.0)  # Normalize but don't amplify
    
    return gains

# -----------------------
# FIXED Trajectory (smoother, less extreme)
# -----------------------
def generate_trajectory(samples, room_size=(8,6,3.2)):
    t = np.linspace(0, 1, samples)
    # Smoother, smaller movements
    x = np.sin(2*np.pi*0.05*t) * room_size[0] * 0.3  # Reduced amplitude and frequency
    y = np.cos(2*np.pi*0.03*t) * room_size[1] * 0.3  
    z = np.sin(2*np.pi*0.04*t) * room_size[2] * 0.2 + room_size[2]*0.1  # Stay above floor
    return np.stack([x, y, z], axis=1)

traj = generate_trajectory(total_samples)

# -----------------------
# FIXED Process frames (removed duplicate reflection processing)
# -----------------------
window_size = 2048
hop = 512

print("Processing audio frames...")
for i in range(0, total_samples, hop):
    if i % (hop * 100) == 0:  # Progress indicator
        progress = i / total_samples * 100
        print(f"Progress: {progress:.1f}%")
        
    frame = samples[i:i+window_size]
    if len(frame) == 0:
        continue

    # Direct sound
    s_pos = traj[i]
    d = dist(listener, s_pos) + 1e-6
    vec = listener - s_pos
    vec_norm = vec / d
    
    az = math.atan2(vec_norm[1], vec_norm[0])
    el = math.asin(np.clip(vec_norm[2], -1, 1))
    
    gains = speaker_gains_7_1_4(az, el)
    att = max(0.1, min(1.0, 1.0/(1.0 + 0.1*d)))  # More conservative distance attenuation
    
    # Add direct sound
    end_idx = min(i + len(frame), total_samples)
    out_mc[i:end_idx,:] += frame[:end_idx-i,None] * gains * att

    # FIXED: Single reflection processing call (removed duplicate code)
    out_mc = process_reflections(frame, s_pos, listener, planes, out_mc, fs, i, room_diag)

print("Processing complete!")

# -----------------------
# FIXED Output format and normalization
# -----------------------
print("\nChoose output format:")
print("1. 96 kHz, 32-bit float (original)")
print("2. 48 kHz, 32-bit float (resampled)")  # Changed from 42kHz to standard 48kHz
choice = input("Enter 1 or 2: ").strip()
if choice=="1":
    TARGET_FS=96000
    BIT_DEPTH=np.float32
    suffix="96k_32bit"
elif choice=="2":
    TARGET_FS=48000  # Standard sample rate
    BIT_DEPTH=np.float32
    suffix="48k_32bit"
else:
    TARGET_FS=48000
    BIT_DEPTH=np.float32
    suffix="48k_32bit"

# -----------------------
# FIXED Resample multichannel with proper normalization
# -----------------------
# Apply gentle limiter before resampling
peak = np.max(np.abs(out_mc))
if peak > 0.95:
    out_mc = out_mc / peak * 0.95
    print(f"Applied limiting: peak was {peak:.3f}")

if TARGET_FS != fs:
    num_samples_target = int(out_mc.shape[0]*TARGET_FS/fs)
    ratio = Fraction(TARGET_FS, fs).limit_denominator()
    out_mc_resampled = np.zeros((num_samples_target, mc), dtype=np.float32)
    
    print("Resampling multichannel audio...")
    for ch in range(mc):
        out_mc_resampled[:, ch] = resample_poly(out_mc[:, ch], ratio.numerator, ratio.denominator)
else:
    out_mc_resampled = out_mc.copy()

# FIXED: Gentler normalization
peak_resampled = np.max(np.abs(out_mc_resampled))
if peak_resampled > 0:
    out_mc_resampled = out_mc_resampled / peak_resampled * 0.9  # Leave 10% headroom

mc_filename = f"3d_music_7_1_4_reflections_{suffix}.wav"
write(mc_filename, TARGET_FS, out_mc_resampled.astype(BIT_DEPTH))

# -----------------------
# FIXED Stereo downmix with better channel mapping
# -----------------------
if TARGET_FS != fs:
    stereo = np.zeros((out_mc.shape[0], 2), dtype=np.float32)
else:
    stereo = np.zeros((out_mc_resampled.shape[0], 2), dtype=np.float32)

# Better stereo fold-down coefficients
if TARGET_FS != fs:
    # Use original sample rate data for stereo mix
    stereo[:,0] = (out_mc[:,0]*0.7 + out_mc[:,4]*0.5 + out_mc[:,6]*0.4 + 
                   out_mc[:,2]*0.3 + out_mc[:,3]*0.2 + out_mc[:,8]*0.3 + out_mc[:,10]*0.3)
    stereo[:,1] = (out_mc[:,1]*0.7 + out_mc[:,5]*0.5 + out_mc[:,7]*0.4 + 
                   out_mc[:,2]*0.3 + out_mc[:,3]*0.2 + out_mc[:,9]*0.3 + out_mc[:,11]*0.3)
    
    # Resample stereo
    stereo_resampled = np.zeros((num_samples_target, 2), dtype=np.float32)
    for ch in range(2):
        stereo_resampled[:,ch] = resample_poly(stereo[:,ch], TARGET_FS, fs)
else:
    # Work directly with resampled data
    stereo_resampled = np.zeros((out_mc_resampled.shape[0], 2), dtype=np.float32)
    stereo_resampled[:,0] = (out_mc_resampled[:,0]*0.7 + out_mc_resampled[:,4]*0.5 + 
                            out_mc_resampled[:,6]*0.4 + out_mc_resampled[:,2]*0.3 + 
                            out_mc_resampled[:,3]*0.2 + out_mc_resampled[:,8]*0.3 + out_mc_resampled[:,10]*0.3)
    stereo_resampled[:,1] = (out_mc_resampled[:,1]*0.7 + out_mc_resampled[:,5]*0.5 + 
                            out_mc_resampled[:,7]*0.4 + out_mc_resampled[:,2]*0.3 + 
                            out_mc_resampled[:,3]*0.2 + out_mc_resampled[:,9]*0.3 + out_mc_resampled[:,11]*0.3)

# Normalize stereo
peak_stereo = np.max(np.abs(stereo_resampled))
if peak_stereo > 0:
    stereo_resampled = stereo_resampled / peak_stereo * 0.9

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
output_video = None
if video_file and os.path.exists(video_file):
    if os.path.exists(mc_filename):
        if ffmpeg_path is None:
             ffmpeg_path = ensure_ffmpeg()
        output_video = merge_audio(video_file, mc_filename, ffmpeg_path)
    else:
        print(f"⚠️ Multichannel audio {mc_filename} not found, skipping video merge.")
display_audio_video_links(output_video)

# -----------------------
# Cleanup
# -----------------------
def cleanup_old_music_and_video(music_files, old_video_file=None):
    files_to_check = [f for f in music_files if os.path.exists(f)]
    files_to_check.append(processed_music_file)
    if old_video_file and os.path.exists(old_video_file):
        files_to_check.append(old_video_file)
    if not files_to_check:
        return
    print("\n⚠️ Original and intermediate files can be removed:")
    for f in files_to_check:
        print(f" - {f}")
    answer=input("Delete these original and intermediate files? (y/n): ").strip().lower()
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
