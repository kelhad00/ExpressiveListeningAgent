The Python scripts in this package exemplify a baseline for the AVEC 2017 Emotion Sub-Challenge.
An SVM model is trained based on the XBOW features from the audio, video, and text modalities (early fusion).
Evaluation is done in terms of the three metrics:
 CCC  (Concordance correlation coefficient)
 PCC  (Pearson's (linear) correlation coefficient)
 RMSE (Root mean squared error)
Only CCC will be used to evaluate the final system.

In oder to run the scripts, Python2 with numpy and scikit-learn is required.

NOTE: The scripts need to be copied to the main folder of the Sub-Challenge (AVEC_17_Emotion_Sub-Challenge/).

run_baseline.py: Run this script with optional parameters "delay (in seconds)" to shift the features in time in order to better match with the targets, and numbers "0" or "1" for each of the three modalities to be used (order: audio video text).
 E.g.: "python run_baseline.py 1.2 1 0 1" to evaluate the baseline with a delay of 1.2 seconds and only audio and text modality.
 By default, the predictions on the Test set are written into the folder "test_predictions".
 The results of the evaluation on the development set are printed on stdout and written to files "results_ccc.txt", "results_pcc.txt", and "results_rmse.txt".

run_optimised_baseline.py: Run the baseline script with optimum modalities and delay for each dimension.
 Complexity and delay have been optimised on the Development partition separately for each dimension (arousal, valence, liking).
 For each dimension, we selected the modality (audio, video, text, or multimodal) that provides the highest CCC on the Test partition.
 This requires 4 trials of the organisers, compared to the 5 trials each participant of the challenge has.

write_predictions.py: This function is used to store the predictions on the Test set.
 Submissions can be done in this format or in the format described on the website of the challenge website 
  (a separate folder for each dimension arousal, valence, and liking).
 It has 3 required arguments.
  path_output: Folder to output the predictions.
  filename:    Filename, e.g., "Test_03.csv"
  predictions: Numpy array of the predictions (number of dimensions X number of predictions)
  sr_labels (default=0.1): Must NOT be changed for the challenge.
 
 Example how to write random predictions for the file Test_01.csv:
  import numpy as np
  from write_predictions import write_predictions
  predA = np.array(np.random.rand(1755,1))
  predV = np.array(np.random.rand(1755,1))
  predL = np.array(np.random.rand(1755,1))
  predictions = np.array([predA,predV,predL])
  write_predictions("test_predictions/","Test_01.csv",predictions)

The scripts calc_scores.py and load_features.py include helper functions.

If you have any questions, please contact Maximilian Schmitt, University of Passau:
maximilian.schmitt@uni-passau.de
