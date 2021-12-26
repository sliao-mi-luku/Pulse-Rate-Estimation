# Pulse Rate Estimation for Wearable Device
Estimates the pulse rate from signals recorded from a wearable device while the users are running

---
*Last updated: 12/25/2021*


[![pulse-rate-frontpage-image.png](https://i.postimg.cc/Kj9TQqd4/pulse-rate-frontpage-image.png)](https://postimg.cc/5QCj239M)
<p align="center">
    Algorithm estimates the pulse rate when the user is running
</p>

## Project Summary

1. We built an algorithm that estimates the pulse rate from the PPG signal data from a wristband
2. We used the acceleration data so that the movement of the hand is compensated
3. The algorithm achieves MAE = 13.6 BPM on the training dataset and 5.69 BPM on the test dataset
4. The algorithm is applied to the clinical data to study key healthcare trends

## Dataset

This project uses the **Troika** dataset. The PPG and acceleration (from a wristband device) signals in the x/y/z axis were recorded from subjects under age 18-35. The PPG is measured by the pulse oximeters with green (515 nm) LEDs. The acceleration was measured by the tree-axis accelerometer from the wrist. The signals are recorded at **125 Hz**. The data was recorded when each subject was running on a treadmill with changing speed, with 30 seconds of resting in the beginning and the end of each recording.

In **TYPE01** recording, the running speeds are:
> rest(30s) -> 8km/h(1min) -> 15km/h(1min) -> 8km/h(1min) -> 15km/h(1min) -> rest(30s)

In **TYPE02** recording, the running speeds are:
> rest(30s) -> 6km/h(1min) -> 12km/h(1min) -> 6km/h(1min) -> 12km/h(1min) -> rest(30s)

The training dataset contains 12 (2 TYPE01 and 10 TYPE02) recordings.

The ECG signal is recorded simultaneously, from which the pulse rate is calculated and used as the ground truth. The actual pulse rate is calculated from a sliding window of width = 8 seconds and step size = 2 seconds.

For more information of the dataset, please read the original paper:

> Zhilin Zhang, Zhouyue Pi, Benyuan Liu, ‘‘TROIKA: A General Framework for Heart Rate Monitoring Using Wrist-Type Photoplethysmographic Signals During Intensive Physical Exercise,’’IEEE Trans. on Biomedical Engineering, vol. 62, no. 2, pp. 522-531, February 2015.

Each data file contains a **data_fls** and **ref_fls**

- data_fls (a list of filenames) contains the paths to the physiological data
- ref_fls (a list of filenames) contains the paths to the actual pulse rate data.

Physioloical data can be extracted by loading **data_fls[i]** by the function `LoadTroikaDataFile` into a numpy array of shape (4, length_of_signal):
- data[0]: photoplethysmography (PPG) signal
- data[1]: acceleration signal along the x-axis
- data[2]: acceleration signal along the y-axis
- data[3]: acceleration signal along the z-axis

The actual pulse rate can be extracted by loading **ref_fls[i]** by using `sp.io.loadmat(ref_fls[i])['BPM0'][:, 0]`. This gives a 1-dimensional numpy array of the actual pulse rate. The actual pulse rate values were calculated by a sliding window (width = 8 sec, step size = 2 sec). In other words, the first value is the actual pulse rate from 0 sec to 8 sec, the second value is the actual pulse rate from 2 sec to 10 sec, and so on.

**short-comings**

The dataset has some short-comings.

1. The data was measured by a wristband. It's possible that the wristband could move relative to the wrist while the subject is running.

2. The data was recorded when subjects are **resting** or **running on a treadmill**. The posture of running and the movement of the wrist is different from person to person. We don't have data when the subject is doing some other activities like hiking, swimming, eating, or sleeping.

3. We only have 12 recordings in our training dataset. It's better to have data from more subjects so that we can capture a more universal features during running.

4. The actual pulse rates were calculated by an 8-second sliding window of step size 2 seconds. By changing these 2 parameters (window width and the step size), different actual pulse rates can be obtained. It is therefore important to quantified the confidence level in the "actual pulse rate" estimate.

5. We only have data recorded from subjects with age from 18 to 35. We don't have data from people younger than 18 or people order than 35.

## Code Description

To run the code, please run the **Code** cell to import the modules/libraries and the required functions.

*Dependencies*

The code needs the following modules/libraries to run:

- `glob` is the module that can be used to find all file paths that match the same pattern.
- `numpy` (version 1.12.1) is the library that can be used for scientific computing. We use this package for handling matrices/arrays and applying the fast Fourier transform (FFT) to the photoplethysmography (PPG) signal and accelerometer signal.
- `scipy` (version 1.2.1) is the library for scientific computing. We use the function in `scipy.signal` to detect peaks in the spectrum of the PPG signal and the accelerometer signal. We use the function in `scipy.io` to load the data file (in `.mat` format).
- `matplotlib` (version 2.1.0) contains functions that we'll use to make plots and visualize the data.

By running the **Code** cell, these modules/libraries should be imported:

```python
import glob
import numpy as np
import scipy as sp
import scipy.io
import scipy.signal
import matplotlib.pyplot as plt
```

For more details of the libraries and how to install their latest version, please visit their websites below:

[glob](https://docs.python.org/3/library/glob.html)

[numpy](https://numpy.org/)

[scipy](https://scipy.org/)

[matplotlib](https://matplotlib.org/)

*Functions*

Once the **Code** cell is run, required functions will also be defined and be ready to be called. The **Code** cell contains 6 functions, 4 were provided along with the project and 2 were written by me:

- `LoadTroikaDataset` (provided) loads the Troika dataset and returns 2 lists **data_fls** and **ref_fls**.
    - data_fls is a list containing the filenames of the physiological data.
    - ref_fls is a list containing the filenames to the actual pulse rate data.
    - This function does not take any input. You only need to run `data_fls, ref_fls = LoadTroikaDataset()` to get the two lists

- `LoadTroikaDataFile` (provided) loads a single data file.
    - It takes the filename as the argument
    - The function returns a numpy array **data** of shape (4, n_samples)
    - data[0]: photoplethysmography (PPG) signal
    - data[1]: acceleration signal along the x-axis
    - data[2]: acceleration signal along the y-axis
    - data[3]: acceleration signal along the z-axis
    - for example, you can run `ppg, accx, accy, accz = LoadTroikaDataFile('data_fls[0]')` to get the ppg and acc signals from the first training data file.

- `AggregateErrorMetric` (provided) chooses the 90% best estimate (whose confidence score is above the 10th percentile) and calculates the mean absolute error (MAE) of these estimates.
    - This function takes 2 inputs: *pr_errors* and *confidence_est*
    - pr_errors is a list (numpy array) of the absolute error of each prediction made by the algorithm
    - confidence_est is a list (numpy array) of the confidence scores of each prediction
    - this function will select all predictions that is above the 10th percentile and return the mean absolute error of these predictions

- `Evaluate` (provided) evaluates the algorithm (details given below) on the whole training dataset. It combines the estimates and the confidence scores from every data file in the training dataset and calls the function `AggregateErrorMetric` to calculate the MAE of the 90% best estimate.
    - This function does not take any input
    - You can evaluate the algorithm by simply running `Evaluate()`

- `ExtractSpectrumPeaks` (self-written) processes the signal in the frequency domain to extract features by the following steps. The function returns 2 numpy arrays **signal_pks** and **signal_fft**
    - This function takes 2 inputs: *signal* and *fft_freq*
        - signal is the signal you want to extract the locations (indices) of the peaks in the spectrum (ex. the PPG signal)
        - fft_freq is the frequency values where the FFT is calculated at. We need this information to link the location of the spectral peak to its actual frequency value
    - The function processes the signal by the following steps:
        - The signal is first passed through a fast Fourier transform (FFT) by `numpy.fft.rfft(signal)`
        - The frequency components below 40 BPM (2/3 Hz) or above 240 BPM (4 Hz) were removed (replaced with zeros)
        - Peaks in the spectrum are identified by `scipy.signal.find_peaks()`
        - Peaks are sorted based on it's height in the spectrum (descendingly)
        - The indices of the sorted peaks (signal_pks) and the spectrum data (signal_fft) are returned

- `RunPulseRateAlgorithm` (self-written) is the main algorithm that estimate the pulse rate from the PPG and acceleration data.
    - This function takes 2 inputs: *data_fl* and *ref_fl*, from a single data file
    - The algorithm estimates the pulse rate and calculates the absolute errors and thee corresponding confidence scores
    - More details of the algorithm can be found in the **Algorithm Description** below

**How to run the code**

1. Run the **Code** cell to import libraries and the functions

2. Run `RunPulseRateAlgorithm(data_fl, ref_fl)` to run the algorithm on a single data file

3. Run `Evaluate()` to calculate the mean absolute error from the 90% most confident estimates on the whole dataset

**About the parameters and how to modify them in the algorithm**

You can change the parameters in the function `RunPulseRateAlgorithm`

1. sampling rate

The sampling rate of the dataset used in this project is **125 Hz**. If you have a dataset with a different sampling rate, you can change the line:

```python
fs = 125.0
```

2. the temporal format of the actual pulse rate

In this dataset, the actual pulse rate is given in a sliding window of 8 seconds in width and a step size of 2 seconds. For example, the first value is the actual pulse rate from time 0 seconds to 8 seconds. The second value is from 2 seconds to 10 seconds. The width and the step size of the sliding window are given by the lines in `RunPulseRateAlgorithm`:

```python
window_width = 8
window_stepsize = 2
```

You can change these values if your actual pulse rate is obtained by a different sliding window.

3. Confidence score

This algorithm used a self-defined confidence to quantify how confident the algorithm is at making a prediction. By default, this score is calculated by dividing the **area near the prediction frequency in the PPG spectrum** by the **total area under the PPG spectrum**.

By saying "near the prediction frequency", we mean a range of 0.25 Hz in the frequency domain. For example, if the predicted pulse rate is 1.5 Hz, we will calculate the area under the PPG spectrum from 1.25 Hz to 1.75 Hz, and divided the area by the total area under the PPG spectrum.

This default value (0.25 Hz) gives a good result, but You can modify this value if needed. To modify, edit the line in the function `RunPulseRateAlgorithm`:

```python
conf_width = 0.25
```

## Algorithm

The algorithm works by the following:

- Individual ppg, accx, accy, accz signals are loaded by the function `LoadTroikaDataFile`
- Actual pulse rate data are loaded by `actual_bpm_whole = sp.io.loadmat(ref_fl)['BPM0'][:, 0]`
- The data signals (ppg, acc's) were segmented in the same way as the actual pulse rate data: using a sliding window with `width = 8` seconds and `step size = 2` seconds. Each window has its corresponding actual pulse rate value in `actual_bpm_whole`
- Iterate through each segmented window, and in each window:
    - Calculate the total acceleration (acc) by taking the square root of the sum of the squares of the accx, accy, and accz
    - Use the helper function `ExtractSpectrumPeaks` to extract the peaks from the spectrums of the ppg, accx, accy, accz, and acc. The details of the helper function can be found above in the "Code description" section
    - I removed the peaks from the PPG spectrum if a peak at the same frequency is also observed as a dominant peak in any of the acceleration signal (x, y, z, and total acc). This is because I suspect that the PPG is picking up this frequency due to the movement of the wristband and not by the pulse rate.
    - After removing those acceleration-related peaks. I choose the highest peak in the remaining peaks as my pulse rate estimate.
    - **(heuristic method)** I observed that the algorithm could give a high error if it gives a prediction that is too far away from its prediction made from the previous window. Theoretically the pulse rate can not change so abruptly. To take this physiological aspect into account, I introduce a heuristic method to correct any abrupt change in the model prediction: for each prediction, if it's more than 1 Hz away from its previous prediction (which is how I define the change as abrupt), I will correct my prediction by using the midpoint between the current and its previous prediction as my prediction instead. This method works like a median filter on my model output. By doing this, I can reduce the chance that the algorithm produces a "jumpy" pulse rate. This method works.
    - **(confidence score)** The confidence is calculated by dividing **area_near_prediction** by **area_under_whole_ppg_spectrum**
        - area_near_prediction is the area under the spectrum that is within 0.25 Hz from the predicted pulse rate
        - area_under_whole_ppg_spectrum is the area under the whole PPG spectrum (from 2/3 Hz to 4 Hz)
        - the logic is that when a predicted pulse rate has a dominant peak in the spectrum (larger area ratio), it's more likely that this is closer the actual pulse rate
- Finally, the algorithm calculates the absolute error of the prediction from the actual pulse rate, and returns 2 numpy arrays: a list of the absolute errors of each window, and a list of the confidence scores of each window.

Based on the algorithm I implemented. The model could fail if the actual pulse rate is the same as the movement of the wrist. If this is the scenario, the algorithm may not be able to find the actual pulse rate because it would remove the peak that is dominating in the acceleration signal.

## Algorithm Performance
The algorithm achieves **MAE = 13.606** on the training dataset and **MAE = 5.69** on the test dataset.

Although the algorithm gave a good result on the Troika dataset, we need to be very careful if we want to generalize this algorithm to some other dataset. This is because the dataset we used to design our model is based on the recordings during **resting and running on the treadmill**. Also the Troika dataset was only recorded from subjects with **age from 18 to 35**. If we have a different dataset recorded from **different activities or age groups**, the algorithm may not give a good prediction.

## Clinical Application

The algorithm is applied to the [Cardiac Arrythmia Suppression Trial (CAST) dataset](https://physionet.org/content/crisdb/1.0.0/) to study the relation between the resting heart rate and the age. The conclusions are:

1. For women, we see a higher resting heart rate in the age groups 40-60 and lower in the age group 70-74.

2. For men we see a relatively smaller variance between the age groups. The resting heart rate in the age group 45-49 is higher.

3. In comparison to men, women's heart rate is overall higher.

4. The data is biased and not balanced. According the description, the data is collected from people who have had a myocardial infarction (MI) within the past two years. And from the value counts below, we can see that we have more male data than female data. We also have relatively smaller data in younger and older groups in both male and female.

5. To improve, we need data from younger groups (<35) and older groups (>80). We also need to collect more female data to reduce the variance in each age group.

6. The trend is more obvious in female data. More statistical analysis is needed to validate the trend in male data.

## References

1. Troika dataset: https://arxiv.org/abs/1409.5181
2. CAST dataset: https://physionet.org/content/crisdb/1.0.0/
