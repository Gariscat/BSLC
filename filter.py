import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import tempfile
import os
import matplotlib.pyplot as plt


def ensure_mono(audio):
    """
    Ensures the audio signal is mono. If the audio is multi-channel, it is converted to mono by averaging the channels.

    Parameters:
    - audio: The input audio signal as a NumPy array (can be stereo or multi-channel).

    Returns:
    - mono_audio: The audio signal converted to mono.
    """
    if len(audio.shape) > 1:  # If audio has more than one channel
        mono_audio = np.mean(audio, axis=1)  # Average the channels to create mono audio
    else:
        mono_audio = audio  # Audio is already mono
    return mono_audio


def resample_audio(audio, original_sample_rate, target_sample_rate=44100):
    """
    Resamples an audio signal to the target sample rate (default: 44100 Hz).

    Parameters:
    - audio: The input audio signal as a NumPy array.
    - original_sample_rate: The original sample rate of the audio signal.
    - target_sample_rate: The desired sample rate (default is 44100 Hz).

    Returns:
    - resampled_audio: The audio signal resampled to the target sample rate.
    """
    # Ensure the audio is mono
    mono_audio = ensure_mono(audio)
    
    # Calculate the number of samples for the target sample rate
    num_samples = int(len(mono_audio) * target_sample_rate / original_sample_rate)

    # Resample the audio signal
    resampled_audio = signal.resample(mono_audio, num_samples)

    return resampled_audio, target_sample_rate


def split_audio_frequencies(audio, sample_rate, debug=False):
    """
    Splits an audio signal into low, mid, and high frequency bands, and optionally saves the parts as files.

    Parameters:
    - audio: The input audio signal as a NumPy array.
    - sample_rate: The sample rate of the audio signal.
    - debug: If True, save the low, mid, and high frequency parts as temporary files.

    Returns:
    - low_freq: The low-frequency band (35 - 150 Hz).
    - mid_freq: The mid-frequency band (150 - 2000 Hz).
    - high_freq: The high-frequency band (2000 - 20000 Hz).
    """

    # Ensure the audio is mono
    mono_audio = ensure_mono(audio)

    # Define frequency boundaries
    low_cutoff = (35, 150)
    mid_cutoff = (150, 2000)
    high_cutoff = (2000, 20000)

    # Design Butterworth bandpass filters for each frequency band
    sos_low = signal.butter(10, low_cutoff, btype='bandpass', fs=sample_rate, output='sos')
    sos_mid = signal.butter(10, mid_cutoff, btype='bandpass', fs=sample_rate, output='sos')
    sos_high = signal.butter(10, high_cutoff, btype='bandpass', fs=sample_rate, output='sos')

    # Apply the filters to the audio signal
    low_freq = signal.sosfilt(sos_low, mono_audio)
    mid_freq = signal.sosfilt(sos_mid, mono_audio)
    high_freq = signal.sosfilt(sos_high, mono_audio)

    # Debugging: Save the split audio parts to temporary files
    if debug:
        import tempfile
        import os
        
        # Create temporary directories
        temp_dir = '.tmp'
        os.makedirs(temp_dir, exist_ok=True)

        # Save low frequencies
        low_file = os.path.join(temp_dir, f"(LOW) {os.path.basename(audio)}.wav")
        wavfile.write(low_file, sample_rate, np.int16(low_freq / np.max(np.abs(low_freq)) * 32767))
        print(f"Low frequencies saved to: {low_file}")

        # Save mid frequencies
        mid_file = os.path.join(temp_dir, f"(MID) {os.path.basename(audio)}.wav")
        wavfile.write(mid_file, sample_rate, np.int16(mid_freq / np.max(np.abs(mid_freq)) * 32767))
        print(f"Mid frequencies saved to: {mid_file}")

        # Save high frequencies
        high_file = os.path.join(temp_dir, f"(HIGH) {os.path.basename(audio)}.wav")
        wavfile.write(high_file, sample_rate, np.int16(high_freq / np.max(np.abs(high_freq)) * 32767))
        print(f"High frequencies saved to: {high_file}")

    return low_freq, mid_freq, high_freq


def calculate_loudness(audio, window_size=1024, smoothing_window=1024, n=1):
    """
    Calculates the loudness of an audio signal using RMS (Root Mean Square) over time and applies average smoothing n times.

    Parameters:
    - audio: The input audio signal as a NumPy array.
    - window_size: The size of the window to compute RMS (default: 1024 samples).
    - smoothing_window: The size of the window for smoothing the loudness curve (default: 512 samples).
    - n: The number of times to apply the smoothing (default: 1).

    Returns:
    - smoothed_loudness: The loudness curve after applying recurrent average smoothing.
    """
    # Calculate the RMS of the signal over the given window size
    loudness = np.sqrt(np.convolve(audio ** 2, np.ones(window_size) / window_size, mode='valid'))
    
    # Recurrent smoothing n times
    for _ in range(n):
        loudness = np.convolve(loudness, np.ones(smoothing_window) / smoothing_window, mode='valid')
    
    return loudness


def detect_spikes(loudness, lambda_val=0.2):
    """
    Detects spikes in a loudness curve based on a defined threshold.

    Parameters:
    - loudness: The loudness curve as a NumPy array.
    - lambda_val: The multiplier for defining the spike threshold, default is 0.25.

    Returns:
    - spike: A binary array where 1 indicates a spike and 0 indicates no spike.
    """
    # Calculate the range of the loudness curve
    loudness_range = np.max(loudness) - np.min(loudness)
    
    # Initialize the spike array with zeros
    spike = np.zeros(len(loudness), dtype=int)
    
    # Iterate over the loudness curve, starting from the second element
    alpha = 2048
    
    for i in range(alpha, len(loudness)-1):
        # If the difference exceeds the threshold, mark a spike
        local_maximum = loudness[i] > loudness[i-1] and loudness[i] > loudness[i+1]
        if local_maximum and loudness[i] - loudness[i - alpha] > lambda_val * loudness_range:
            spike[i] = int((loudness[i] - loudness[i - alpha]) / (lambda_val * loudness_range))
    
    return spike
    
'''def detect_spikes(loudness, l=8, r=8):
    """
    Detects spikes in a loudness curve based on increasing and decreasing patterns.

    Parameters:
    - loudness: The loudness curve as a NumPy array.
    - l: Number of frames before the current frame to check for increasing loudness (default is 3).
    - r: Number of frames after the current frame to check for decreasing loudness (default is 3).

    Returns:
    - spike: A binary array where 1 indicates a spike and 0 indicates no spike.
    """
    # Initialize the spike array with zeros
    spike = np.zeros(len(loudness), dtype=int)
    
    # Loop through each frame, skipping the boundaries where we can't form a full window
    for i in range(l, len(loudness) - r):
        # Check if loudness is increasing from (i-l) to i
        increasing = all(loudness[i-j] < loudness[i-j+1] for j in range(l, 0, -1))
        # Check if loudness is decreasing from i to (i+r)
        decreasing = all(loudness[i+j] > loudness[i+j+1] for j in range(r))
        
        # If both increasing and decreasing conditions are met, it's a spike
        if increasing and decreasing:
            spike[i] = 1
    
    return spike'''



def plot_loudness_and_spikes(original_audio, low_freq, mid_freq, high_freq, sample_rate):
    """
    Plots the original waveform, loudness curves, and corresponding spikes of the original audio and frequency bands.

    Parameters:
    - original_audio: The original audio signal as a NumPy array.
    - low_freq, mid_freq, high_freq: The low, mid, and high frequency parts of the audio.
    - sample_rate: The sample rate of the audio signal.
    """

    # Calculate time axis for the original audio
    # time_axis = np.linspace(0, len(original_audio) / sample_rate, num=len(original_audio))
    ## time_axis = np.linspace(0, len(original_audio))

    # Calculate loudness (RMS) of the original audio and frequency bands
    original_loudness = calculate_loudness(original_audio)
    low_loudness = calculate_loudness(low_freq)
    mid_loudness = calculate_loudness(mid_freq)
    high_loudness = calculate_loudness(high_freq)

    # Detect spikes in the loudness curves
    original_spikes = detect_spikes(original_loudness)
    low_spikes = detect_spikes(low_loudness)
    mid_spikes = detect_spikes(mid_loudness)
    high_spikes = detect_spikes(high_loudness)

    # Create time axes for loudness and spikes (due to different window sizes, they are shorter than the original signal)
    # time_axis_loudness = np.linspace(0, len(original_loudness) / sample_rate, num=len(original_loudness))
    ## time_axis_loudness = np.linspace(0, len(original_loudness))
    # Create a figure with 8 subplots (4 rows, 2 columns)
    fig, axes = plt.subplots(4, 2, figsize=(12, 8), sharex=False)

    # Plot the original waveform
    axes[0, 0].plot(original_audio, color='blue')
    axes[0, 0].set_title('Original Audio Waveform')
    axes[0, 0].set_ylabel('Amplitude')

    # Plot the loudness of the original audio
    axes[0, 1].plot(original_loudness, color='green')
    axes[0, 1].set_title('Original Audio Loudness')
    axes[0, 1].set_ylabel('Loudness (RMS)')
    # axes[0, 1].set_xticks(time_axis_loudness[::int(len(time_axis_loudness)/10)])
    axes[0, 1].tick_params(labelbottom=True)

    # Plot the loudness of the low frequencies
    axes[1, 0].plot(low_loudness, color='red')
    axes[1, 0].set_title('Low Frequencies Loudness (35 - 150 Hz)')
    axes[1, 0].set_ylabel('Loudness (RMS)')
    # axes[1, 0].set_xticks(time_axis_loudness[::int(len(time_axis_loudness)/10)])
    axes[1, 0].tick_params(labelbottom=True)

    # Plot the spikes of the low frequencies (as a 0/1 curve)
    axes[1, 1].plot(low_spikes, color='red')
    axes[1, 1].set_title('Low Frequencies Spikes')
    axes[1, 1].set_ylim([-0.1, 1.1])

    # Plot the loudness of the mid frequencies
    axes[2, 0].plot(mid_loudness, color='orange')
    axes[2, 0].set_title('Mid Frequencies Loudness (150 - 2000 Hz)')
    axes[2, 0].set_ylabel('Loudness (RMS)')
    # axes[2, 0].set_xticks(time_axis_loudness[::int(len(time_axis_loudness)/10)])
    axes[2, 0].tick_params(labelbottom=True)

    # Plot the spikes of the mid frequencies (as a 0/1 curve)
    axes[2, 1].plot(mid_spikes, color='orange')
    axes[2, 1].set_title('Mid Frequencies Spikes')
    axes[2, 1].set_ylim([-0.1, 1.1])

    # Plot the loudness of the high frequencies
    axes[3, 0].plot(high_loudness, color='purple')
    axes[3, 0].set_title('High Frequencies Loudness (2000 - 20000 Hz)')
    axes[3, 0].set_ylabel('Loudness (RMS)')
    # axes[3, 0].set_xticks(time_axis_loudness[::int(len(time_axis_loudness)/10)])
    axes[3, 0].tick_params(labelbottom=True)

    # Plot the spikes of the high frequencies (as a 0/1 curve)
    axes[3, 1].plot(high_spikes, color='purple')
    axes[3, 1].set_title('High Frequencies Spikes')
    axes[3, 1].set_ylim([-0.1, 1.1])


    # Show the plot
    plt.tight_layout()
    plt.show()
    plt.close()

# Example of how to use the function with debugging enabled
# Assuming 'audio' is a NumPy array of the audio signal and 'sample_rate' is the sample rate
# low, mid, high = split_audio_frequencies(audio, sample_rate, debug=True)


if __name__ == '__main__':
    # Load an example audio file
    sample_rate, audio = wavfile.read("demo-live-set-inst.mp3")
    audio, sample_rate = resample_audio(audio, sample_rate, 44100)

    # Split the audio signal into low, mid, and high frequency bands
    low_freq, mid_freq, high_freq = split_audio_frequencies(audio, sample_rate, debug=False)
    plot_loudness_and_spikes(audio, low_freq, mid_freq, high_freq, sample_rate)