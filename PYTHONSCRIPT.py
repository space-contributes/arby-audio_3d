import os
import numpy as np
import requests
import base64
from scipy.io.wavfile import write, read
from pydub import AudioSegment
from scipy.signal import butter, lfilter
from IPython.display import Audio, HTML, display
import math
# -----------------------
# Parameters
# -----------------------
fs = 96000  # sample rate
mc = 12     # 7.1.4 channels
REFLECTION_GAIN = 0.7
music_url = "https://TESTURL.COM"
music_file = "music.wav"
wav_file = "music_96k.wav"
multichannel_wav = "3d_music_7_1_4_reflections.wav"
binaural_wav = "3d_music_binaural_reflections.wav"
c = 343.0  # speed of sound m/s
MAX_DISTANCE_FOR_FILTER = 20.0 # Distance in meters for max filtering effect

parser = argparse.ArgumentParser(description="3D audio simulation with reflections")
parser.add_argument("--music_url", type=str, default=MUSIC_URL,
                    help="URL of the music file to use (default: %(default)s)")
args = parser.parse_args()
music_url = args.music_url

# -----------------------
# Download and prepare the music file
# -----------------------
if not os.path.exists(music_file):
    print("Downloading music file...")
    r = requests.get(music_url, timeout=30)
    with open(music_file, 'wb') as f:
        f.write(r.content)
    print("Download complete.")
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
# Room geometry (simple cube)
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
# 7.1.4 speaker gains
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
# NEW HELPER FUNCTIONS
# -----------------------
def butter_lowpass(cutoff, fs, order=4):
    """
    Creates the coefficients for a low-pass Butterworth filter.
    """
    nyq = 0.5 * fs
    normal_cutoff = max(1e-6, min(1.0 - 1e-6, cutoff/nyq)) # Ensure normal_cutoff is strictly less than 1
    b, a = butter(order, normal_cutoff, btype='low')
    return b, a
def apply_lowpass(sig, cutoff):
    """
     Applies the low-pass filter to a signal.
    """
    if cutoff <= 20: return np.zeros_like(sig)
    b, a = butter_lowpass(cutoff, fs)
    return lfilter(b, a, sig)
def reflect_point_across_plane(s, p0, n):
    """
    Reflects a point 's' across a plane defined by a point 'p0' and a normal 'n'.
    """
    v = s - p0
    return s - 2*np.dot(v, n)*n
def dist(a, b):
    """
    Calculates the Euclidean distance between two points.
    """
    return np.linalg.norm(a - b)
# -----------------------
# Frequency-based reflections with time delay and low-pass filter.
# -----------------------
window_size = 2048
hop = 512
for i in range(0, total_samples, hop):
    frame = samples[i:i+window_size]
    if len(frame) == 0:
        continue
    # Direct sound
    s_pos = traj[i]
    d = dist(listener, s_pos) + 1e-6
    vec = listener - s_pos
    az = math.atan2(vec[1], vec[0])
    el = math.asin(vec[2]/d)
    gains = speaker_gains_7_1_4(az, el)
    att = max(0.05, min(1.0, 1.0/(1.0+0.05*d)))
    out_mc[i:i+len(frame), :] += frame[:, None]*gains*att
    # Reflections
    for plane in planes:
        s_ref = reflect_point_across_plane(s_pos, plane['p0'], plane['n'])
        d_ref = dist(listener, s_ref) + 1e-6
        # Calculate and apply time delay based on distance difference
        time_delay_s = (d_ref - d) / c
        delay_samples = int(math.ceil(time_delay_s * fs))
        if i + delay_samples >= total_samples:
            continue
        # Calculate the cutoff frequency for the low-pass filter.
        cutoff_freq = fs * (1.0 - np.clip(d_ref / MAX_DISTANCE_FOR_FILTER, 0.0, 0.9))
        # Apply the low-pass filter to the reflected frame.
        filtered_frame = apply_lowpass(frame, cutoff_freq)
        az_r = math.atan2(listener[1]-s_ref[1], listener[0]-s_ref[0])
        el_r = math.asin((listener[2]-s_ref[2])/d_ref)
        gains_r = speaker_gains_7_1_4(az_r, el_r)
        # Reflection gain scales with frequency, energy, and distance.
        refl_att = REFLECTION_GAIN / (1+0.05*d_ref)
        # Determine the actual length to add to out_mc based on remaining space.
        add_length = min(len(filtered_frame), total_samples - (i + delay_samples))
        # Ensure the slice of out_mc has the same length as the portion of filtered_frame being added
        out_mc[i+delay_samples : i+delay_samples+add_length, :] += filtered_frame[:add_length, None] * gains_r * refl_att
# -----------------------
# Normalize and write multichannel,
# -----------------------
out_mc /= np.max(np.abs(out_mc)) + 1e-12
write(multichannel_wav, fs, out_mc.astype(np.float32))
# -----------------------
# Stereo downmix.
# -----------------------
stereo = np.zeros((total_samples, 2), dtype=np.float32)
stereo[:, 0] = out_mc[:, 0] + 0.5*out_mc[:, 4] + 0.5*out_mc[:, 6] + 0.3*out_mc[:, 8] + 0.3*out_mc[:, 10]
stereo[:, 1] = out_mc[:, 1] + 0.5*out_mc[:, 5] + 0.5*out_mc[:, 7] + 0.3*out_mc[:, 9] + 0.3*out_mc[:, 11]
stereo /= np.max(np.abs(stereo)) + 1e-12
write(binaural_wav, fs, stereo.astype(np.float32))
# -----------------------
# Display output
# -----------------------
display(Audio(binaural_wav, rate=fs))
with open(binaural_wav, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
display(HTML(f'<a download="{binaural_wav}" href="data:audio/wav;base64,{b64}">⬇️ Download stereo WAV</a>'))

print("Done — full music track, 7.1.4 with adaptive frequency-based reflections, time delays, and low-pass filtering.")




