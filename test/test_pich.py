import numpy as np

from utilities.pyrapt import pitch
#import raptparams
#import nccfparams
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
from scipy import signal
from scipy.signal import butter, lfilter, freqz

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y




file_name = "/home/vidar/projects/annotation_data/lukas/raw/2022-01-20-10-07-38-036639.wav"
#file_name = "/home/vidar/Desktop/annotation_tor_olav_raw_2022-03-17-14-19-25-938042.wav"
#file_name = "/home/vidar/Desktop/annotation_test_client_raw_2022-08-08-14-55-35-492600.wav"
audio_s, sample_rate = sf.read(file_name)
if len(audio_s.shape) > 1:
    audio_s = audio_s[:,0]
print(f"Audio shape {audio_s.shape}")
#audio_sample = butter_lowpass_filter(audio_s, 500.0, sample_rate)
audio_sample = audio_s
"""
audio_sample *= 16777216.0  # 16777216.0 32768.0
audio_sample = audio_sample.astype(int)
freq_estimates = np.array(())
for start_sample in range(0, len(audio_sample), 150000):
    end_sample = min(start_sample+150000, len(audio_sample)-1)
    samples = audio_sample[start_sample:end_sample, 0]
    freq_estimates = np.append(freq_estimates, rapt((sample_rate, samples)))
for i in range(len(freq_estimates)):
    if freq_estimates[i] > 480:
        freq_estimates[i] = 0.0
"""


# Test pitch for short audio sequences
seq_length = int(sample_rate*0.512)
seq_step = int(sample_rate*0.128)
freq_estimates = np.array(())
for start_sample in range(0, len(audio_sample)-seq_length, seq_step):
    estimates =pitch(audio_sample[start_sample:start_sample+seq_length], sample_rate)
    estimates = np.append(estimates, 0.0)
    freq_estimates = np.append(freq_estimates, estimates)
print(f"Finished, freq estimate size is {len(freq_estimates)}")
plt.plot(freq_estimates, linewidth=2)
default_x_ticks = range(0, len(freq_estimates), 100)
x = []
start_sample =  0
for i in default_x_ticks:
    x.append(i*0.016+(start_sample/sample_rate))
plt.plot()
plt.xticks(default_x_ticks, x)
plt.ylabel('Frequency')
plt.xlabel('Feature no')
plt.show()

exit(1)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
#ax1.plot(t, sig)
ax1.plot(freq_estimates, linewidth=2)
ax1.set_title('10 Hz and 20 Hz sinusoids')
#ax1.axis([0, 1, -2, 2]
#sos = signal.butter(10, 100, 'lp', fs=1000, output='sos')
#filtered = signal.sosfilt(sos, freq_estimates)
filtered = np.zeros((len(freq_estimates)))
for i in range(0, len(freq_estimates)-16, 8):
    var = np.var(freq_estimates[i:i+16])
    for j in range(8):
        filtered[i+j] = var

#ax2.plot(t, filtered)
ax2.plot(filtered)
ax2.set_title('After 15 Hz high-pass filter')
#ax2.axis([0, 1, -2, 2])
ax2.set_xlabel('Time [seconds]')
plt.tight_layout()
plt.show()
