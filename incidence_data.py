class incidence_data:
    """
    Class performing an analysis of incidence of input data (GRF and/or KJA)

    The "execute" function of this class returns:
        - rRMSE, r, MAE and standard error for each time series (real vs predicted values) 
          of the test set when GRF and/or KJA are given as input data

    The "execute" function of this class also displays:
        - a graph comparing real and predicted HJM (mean and standard error) in the three
          scenarios (just GRF, just KJA, GRF and KJA as input data)

    """
    def __init__(self, X_train_s, X_test_s, y_train, y_test):
        self.rrmse=[]
        self.r=[]
        self.mae=[]
        self.st_err=[]
        self.X_train=X_train_s
        self.X_test=X_test_s
        self.y_train=y_train
        self.y_test=y_test


    def initialize_input(self, matrix, index):  
        if index==self.X_train.shape[2]:
            return matrix

        matrix_one_in=np.empty((matrix.shape[0], matrix.shape[1], 1))
        matrix_one_in[:,:,0]=matrix[:,:,index]
        return matrix_one_in

    def execute(self):
        for index in range(self.X_train.shape[2]+1):
            X_train_one_in=my_incidence_data.initialize_input(self.X_train, index)
            X_test_one_in=my_incidence_data.initialize_input(self.X_test, index)

            my_lstm_one_in=model_implementation(X_train_one_in, X_test_one_in, self.y_train)
            _ , y_pred_one_in,_ = my_lstm_one_in.execute_model(lstm_units=512, 
                          dropout_rate_1=0.3, dense_units=100, dropout_rate_2=0.3, 
                          learning_rate=0.0001, validation_split=0.2, epochs=200,
                          model_out=0)



            my_ResultAnalysis_one_in=results_analysis(y_pred_one_in, self.y_test)
            rrmse, r, mae = my_ResultAnalysis_one_in.metrics_calculation()
      
            self.rrmse.append(rrmse)
            self.r.append(r)
            self.mae.append(mae)
            self.st_err.append(np.mean(np.std(y_pred_one_in,0)))

            my_ResultAnalysis_one_in.graph_mean_sd(y_pred_one_in, index, self.X_train.shape[2])
  
        return self.rrmse, self.r, self.mae, self.st_err