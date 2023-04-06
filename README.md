# Deep-Learning-For-HJM-Prediction


The aim of the current project, that I carried out as part of my research at Arcadia University, is to estimate hip joint moments (HJM) starting from knee joint angles and ground reaction forces in healthy controls, patients affected by an hip condition called FAIS and post-operative patients.
Being all the input data time series, an LSTM model has been implemented to accomplish this task.
Although this approach has been employed also by other studies, the current investigation is the first one to analyse the behaviour of multiple groups of individuals (healthy controls, patients and post-operative patients), to deal with subjects performing single leg squat, and one of the few including statistical test (statistical parametric mapping) between real and predicted outputs.

## Dependencies
A list of the packages used for this project is included in the file Requirements.txt

## Dataset
The dataset used for the study includes 53 subjects (24 healthy controls and 29 patients with FAIS, including also 16 post-operative patients). Demographics of these subjects are included in the folder [Dataset](./Dataset/).
Since every subject performed three single leg squat trials with each leg, this leads to a total of 414 observations.
However, some observations have been discarded because of bad sampling during data acquisition (see main.py), so the final number of samples is 334.

Stratified splitting was performed, keeping a ratio of 4:1 between the number of controls, preoperative and postoperative patients between the training and the test set. Also, the observations corresponding to each subject were either all included in the training set or in the test set, and never in both of them, to avoid bias.


## Model Implementation
The image below describes the pipeline followed.

<br>

<p align="center">
  <img src="https://user-images.githubusercontent.com/98240588/230415706-bf5bee5d-1b2a-46e6-a98c-7b06b9526fb6.png" width="700" height="580">
</p>

<br>
<br>

The model architecture has an analogous structure to one of other [Similar_Studies](./References/Similar_Studies/) dealing with similar tasks.

10 fold cross-validation was performed.


## Results Analysis

The table below summarises the results obtained in terms of the evaluation metrics considered (nRMSE, r and MAE)

<table>
<thead>
  <tr>
    <th></th>
    <th>nRMSE</th>
    <th>r</th>
    <th>MAE (Nm(kg)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Training set</td>
    <td>13.45%</td>
    <td>0.95</td>
    <td>0.23</td>
  </tr>
  <tr>
    <td>Test set</td>
    <td>14.93%</td>
    <td>0.94</td>
    <td>0.21</td>
  </tr>
</tbody>
</table>



The graphs below compare the values of HJM predicted by the LSTM model (red) with the ground truth data (blue). 
    <p align="center">
      <img src="https://user-images.githubusercontent.com/98240588/230429667-658ef036-4a2f-4ae8-a5e9-862bf3a8fae1.png" width="600" height="400" alt="Image 1">
      

## Incidence of Input Data
An analysis aiming at quantifying the incidence of input data on the output (HJM) was carried out. Results were consistent with what found by (citazione Rane), since the best performance are achieved when giving in input both kinematic (knee joint angles) and kinetic (ground reaction forces) data.


    <p align="center">
      <img src="https://user-images.githubusercontent.com/98240588/230430115-1eef1ff7-49b7-426d-a5d8-b3257fb3fbf3.png" width="600" height="400" alt="Image 1">
    







