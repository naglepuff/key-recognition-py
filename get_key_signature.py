import numpy as np 

from scipy.io import wavfile 
import scipy.fftpack as fft 
import scipy.signal as signal 

import matplotlib.pyplot as plt 

import sys

from helper_methods import get_filename, find_nearest_index, make_key_signatures

TEST_FILENAME = "i_iv_v_keyA.wav"
RANGE_TOP_HZ = 4200 
RANGE_BOTTOM_HZ = 60
# source: https://pages.mtu.edu/~suits/notefreqs.html
NOTE_VALUES = [
    65.41, 69.30, 73.42, 77.78, 82.41, 87.31, 92.50, 98.00, 103.88, 110.0, 116.54, 123.47,
    130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185.00, 196.00, 207.65, 220.00, 233.08, 246.94,
    261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.15, 493.88,
    523.25, 554.37, 587.33, 622.25, 659.25, 698.46, 739.99, 783.99, 830.61, 880.00, 932.33, 987.77,
    1046.50, 1108.73, 1174.66, 1244.51, 1318.51, 1396.91, 1479.98, 1567.98, 1661.22, 1760.00, 1864.66, 1975.53,
    2093.00, 2217.46, 2349.32, 2489.02, 2637.02, 2793.83, 2959.96, 3135.96, 3322.44, 3520.00, 3729.31, 3951.07,
]

NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B",]

KEY_SIGNATURES = make_key_signatures() 


def main():
    filename = get_filename()

    try:
        # read the wav file, store its data (all channels) and sample rate
        sample_rate, data = wavfile.read(filename, "r") 
    except (FileNotFoundError, ValueError) as error:
        # if the input is not the name of a wav file, return and report to the user
        message = "Error: %s. Please try again with the name of a wav file" % error 
        print(message)
        return

    # ASSUMPTION: the wav is standard and has 2 channels. 
    # Average the 2 channels so as to not lose important frequency information
    # This is the first impovement to be made over the original attempt at the problem in MATLAB
    channel1 = data.T[0] 
    channel2 = data.T[1]

    channel_data = np.array([channel1[i] + channel2[i] / 2 for i in range(0, len(channel1))])

    # take the fft of the signal in the first channel, and create an array of the frequencies (by default the range is [-0.5, 0.5))
    freq_strengths = fft.rfft(channel_data) 
    freqs = fft.rfftfreq(len(freq_strengths))

    # take freqs array and turn it from normalized freqs into Hz range is 0...sample_rate / 2
    freqs_in_hz = freqs * sample_rate
    freq_step = freqs_in_hz[1] - freqs_in_hz[0]

    # I also want to take the absolute value of the magnitude vector 
    freq_strengths = np.absolute(freq_strengths) 

    # Now we want to get rid of any extraneous data. Arbitrarily I will say that songs live in the frequence range
    # of 65.41Hz (C2) to 4186.01 (C8), i.e. top and bottom C keys on a piano
    bottom_index = int(RANGE_BOTTOM_HZ / freq_step) * 2 
    top_index = int(RANGE_TOP_HZ / freq_step) * 2
    freq_strengths = freq_strengths[bottom_index:top_index] 
    freqs_in_hz = freqs_in_hz[bottom_index:top_index]

    # Here I apply a window function to the signal. This is in order to shift weight towards to lower end of the 
    # spectrum. This is an attempt to alleviate the potential issue of harmonics and overtones having a negative
    # impact on the overall result. 
    # This is the second improvement over the MATLAB attempt.
    window = np.logspace(1, 0, num=len(freq_strengths), base=2) 
    window = window - 1 # so it goes from 1..0
    freq_strengths = freq_strengths * window

    plt.plot(freqs_in_hz, freq_strengths)
    plt.show()

    # In my original attempt to solve this problem I found the strengths of notes based on their known frequencies. 
    # I will implement that below, but it should be noted that in the future a peak-finding approach would probably be better.
    # In this approach one would find the peaks in the Fourier Transform and find which notes are closest to those peaks. 
    #
    # Note that a third possible approach involves using a window around each pitch. This will correct for out-of-tune instruments
    # or perhaps other anomalies that would cause a piece of music to not conform to the pitch values above.
    note_strengths = np.zeros(12)

    for i in range(0, len(NOTE_NAMES)):
        indices_of_note = np.arange(i, len(NOTE_VALUES), len(NOTE_NAMES))
        note_values = np.array([NOTE_VALUES[j] for j in indices_of_note]) 
        current_note_idx = np.array([find_nearest_index(freqs_in_hz, j) for j in note_values]) 
        current_note_strengths = np.array([freq_strengths[j] for j in current_note_idx]) 
        note_strengths[i] += np.sum(current_note_strengths)

    """Now the note strengths have been computed. We can find statistics on these note values."""

    print(note_strengths)

if __name__ == "__main__":
    main() 