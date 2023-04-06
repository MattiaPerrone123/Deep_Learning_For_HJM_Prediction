from data_loader import data_loader
from data_splitter import data_splitter
from model_implementation import model_implementation
from cross_validator import cross_validator
from results_analysis import results_analysis
from incidence_data import incidence_data
from my_util import *


#Path that are used to import data from Visual3d and Mokka
path_all='/content/gdrive/MyDrive/Colab Notebooks/SLS/Data_all_V3D'
path_mokka='/content/gdrive/MyDrive/Colab Notebooks/SLS/Data_mokka'
file_path=path_mokka+"/Control_03_left_tr_1.xlsm"

#Values of GRF to fix (see function refining_GRF_values in class dataset_loader)
values_to_fix=['Control_06_right_tr_1.xlsm', 'Control_21_left_tr_2.xlsm',\
               'Control_21_left_tr_3.xlsm', 'FAIS_16_left_tr_2.xlsm',\
               'FAIS_20_left_tr_1.xlsm', 'FAIS_20_left_tr_2.xlsm',\
               'FAIS_20_left_tr_3.xlsm', 'FAIS_20_right_tr_1.xlsm',\
               'FAIS_20_right_tr_2.xlsm', 'FAIS_20_right_tr_3.xlsm',\
               'FAIS_29_right_tr_3.xlsm', "Postop_20_left_tr_1.xlsm",\
               "Postop_20_left_tr_2.xlsm", "Postop_20_left_tr_3.xlsm",\
               "Postop_20_right_tr_1.xlsm", "Postop_20_right_tr_2.xlsm",\
               "Postop_27_right_tr_1.xlsm", "Postop_27_right_tr_3.xlsm"]

#Observations that are dropped because of bad acquisition (see function import_HJM_V3D in 
#class dataset_loader)
value_to_drop=["Control_04_left_HJM_flexion_Trial 1",
                     "Control_04_left_HJM_flexion_Trial 2",
                     "Control_04_left_HJM_flexion_Trial 3",
                     "Control_05_right_HJM_flexion_Trial 3",
                     "Control_06_left_HJM_flexion_Trial 1",
                     "Control_13_right_HJM_flexion_Trial 1",
                     "Control_14_right_HJM_flexion_Trial 2",
                     "Control_19_right_HJM_flexion_Trial 3",
                     "Control_20_right_HJM_flexion_Trial 1",
                     "Control_20_right_HJM_flexion_Trial 2",
                     "Control_20_right_HJM_flexion_Trial 3",
                     "Control_27_right_HJM_flexion_Trial 2",
                     "FAIS_03_left_HJM_flexion_Trial 1",
                     "FAIS_03_left_HJM_flexion_Trial 2",
                     "FAIS_03_left_HJM_flexion_Trial 3",
                     "FAIS_03_right_HJM_flexion_Trial 1",
                     "FAIS_03_right_HJM_flexion_Trial 2",
                     "FAIS_03_right_HJM_flexion_Trial 3",
                     "FAIS_05_left_HJM_flexion_Trial 1",
                     "FAIS_05_left_HJM_flexion_Trial 2",
                     "FAIS_05_left_HJM_flexion_Trial 3",
                     "FAIS_05_right_HJM_flexion_Trial 1",
                     "FAIS_05_right_HJM_flexion_Trial 2",
                     "FAIS_05_right_HJM_flexion_Trial 3",
                     "FAIS_10_right_HJM_flexion_Trial 2",
                     "FAIS_10_right_HJM_flexion_Trial 3",
                     "FAIS_24_left_HJM_flexion_Trial 1",
                     "FAIS_25_right_HJM_flexion_Trial 2",
                     "FAIS_27_right_HJM_flexion_Trial 3",
                     "FAIS_32_right_HJM_flexion_Trial 1",
                     "FAIS_34_right_HJM_flexion_Trial 1",
                     "FAIS_36_left_HJM_flexion_Trial 2",
                     "Postop_04_right_HJM_flexion_Trial 3",
                     "Postop_10_6MPO_right_HJM_flexion_Trial 2",
                     "Postop_11_6MPO_left_HJM_flexion_Trial 1",
                     "Postop_13_6MPO_left_HJM_flexion_Trial 1",
                     "Postop_13_6MPO_left_HJM_flexion_Trial 2",
                     "Postop_13_6MPO_left_HJM_flexion_Trial 3",
                     "Postop_13_6MPO_right_HJM_flexion_Trial 1",
                     "Postop_13_6MPO_right_HJM_flexion_Trial 2",
                     "Postop_13_6MPO_right_HJM_flexion_Trial 3",
                     "Postop_14_6MPO_left_HJM_flexion_Trial 2",
                     "Postop_34_6MPO_left_HJM_flexion_Trial 3",
                     "Postop_36_6MPO_left_HJM_flexion_Trial 2"]               
                     
                     
#Saving values of HJM, KJA and GRF
my_dataset=data_loader()
HJM_flexion, GRF, KJA_flexion, files_sorted_mokka=my_dataset.execute(path_all, 
                                                                     path_mokka, 
                                                                     value_to_drop)
                                                                     
                                                                     

#Saving Training and testing samples
my_data_splitting=data_splitter(HJM_flexion, KJA_flexion, GRF, 2, files_sorted_mokka)
X_train_s, y_train, X_test_s, y_test = my_data_splitting.execute()




#Saving the training history of the model, predictions on training and testing dataset, and the model itself
my_lstm=model_implementation(X_train_s, X_test_s, y_train)
history, y_pred_test, y_pred_train, model = my_lstm.execute_model(lstm_units=512, 
                    dropout_rate_1=0.3, dense_units=100, dropout_rate_2=0.3, 
                    learning_rate=0.0001, validation_split=0.2, epochs=200, model_out=1)



#Saving values of rRMSE, r and MAE for each time series (real vs predicted values) of each fold and the mean for each fold
my_CrossValidator=cross_validator(model, X_train_s, y_train, n_splits=10, learning_rate=0.0001, epochs=200)
rrmse_scores_cv, r_scores_cv, mae_scores_cv, rrmse_cv, r_cv, mae_cv, ttest_scores =my_CrossValidator.execute()




#Saving values of rRMSE, r and MAE for each time series (real vs predicted values) of the test set
my_ResultAnalysis=results_analysis(y_pred_test, y_test)
rrmse_test, r_value_test, mae_test = my_ResultAnalysis.execute()




#Saving values of rRMSE, r, MAE and standard error for each time series (real vs predicted values) 
#of the test set when GRF and/or KJA are given as input data
my_incidence_data=incidence_data(X_train_s, X_test_s, y_train, y_test)
rrmse_one_in, r_one_in, mae_one_in, st_err_one_in=my_incidence_data.execute()
                     