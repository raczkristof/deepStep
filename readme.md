## Data collection

The data used in this project was collected by recording the sound of footsteps of participants. In total it contains 33 different person - shoe combinations among 6 persons, with 20 recordings for each combination. The basic data for the participants can be found in the `data.csv` file.

** The data for this project can be found [here](https://drive.google.com/open?id=1scHsJlCOq0luO0JiUdKg4VH2WY5fYLa1) for now.**

Raw recordings can be found in the `data_raw` folder. Each file contains the footsteps recorded by two microphones: a *Shure SM-58* dynamic microphone and an *AudioTechnica AT2020* condenser microphone, as the left and right channels of the recording respectively. All recordings are 32 bit float @ 44.1 kHz. Also included in the raw recordings a noise sample of the setup: a recording made with nothing making sound actively.

Processed recordings can be found in the `data` folder, organised by recording microphone first, and *train/vaildate/test* split on the next level.

The process for processing the raw recording is performed using the `wav_process.py` python script. The outline of the algorithm is as follows:
1. Reading the noise sample and splitting it by channel
2. Applying a 2nd order highpass butterworth filter to both noise samples (cutoff frequency = 1 kHz)
 I have found that there are some significant artifacts after the noise reduction step if using the unfiltered noise sample.
3. Iterating through each recording:
	a. Splitting the channels
	b. Applying the noisereduction using the noise samples
	c. Applying a higpass filter  (cutoff frequency = 1 kHz)
	d. Applying a notch filter between 43 and 47 Hz
	e. Write the processed file to disk: recordings 1-16 will be used for training, 17-18 for validation and 19-20 for testing