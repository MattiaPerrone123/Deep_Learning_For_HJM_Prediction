class results_analysis:
    """
    Class performing an analysis of the results obtained

    The "execute" function of this class returns:
        - rRMSE, r and MAE for each time series (real vs predicted values) of the test set

    The "execute" function of this class also displays:
        - a graph comparing real and predicted HJM (mean and standard error)
        - SPM graph showing no statistical significant difference between real and
          predicted values
    """
    def __init__(self, y_pred_test, y_test):
        self.y_pred_test=y_pred_test
        self.y_test=y_test


    def metrics_calculation(self):  
        self.y_pred_test=np.squeeze(self.y_pred_test)
        self.y_test=np.squeeze(self.y_test)


        r_values_test = []
        for i in range(len(y_pred_test)):
            r = pearsonr(self.y_pred_test[i], self.y_test[i])
            r_values_test.append(r)
        self.r_value_test=np.mean(r_values_test)

 
        mae_test=[]
        nmae_scores_test=[]
        for i in range(len(self.y_pred_test)):
            mae_test.append(mean_absolute_error(self.y_test[i], self.y_pred_test[i]))

            y_min = np.min(self.y_pred_test[i])
            y_max = np.max(self.y_pred_test[i])
            y_range = y_max - y_min
            nmae_test = (mean_absolute_error(self.y_test[i], self.y_pred_test[i]) / y_range) * 100
            nmae_scores_test.append(nmae_test)

        self.mae_test=np.mean(mae_test)
        nmae_test=np.mean(nmae_scores_test)


        mse_test=mean_squared_error(self.y_pred_test, self.y_test)
        y_min = np.min(self.y_pred_test)
        y_max = np.max(self.y_pred_test)
        y_range = y_max - y_min
        self.rrmse_test = (np.sqrt(mse_test) / y_range) * 100

        return self.rrmse_test, self.r_value_test, self.mae_test 



    def graph_mean_sd(self, vector, index, n_graphs):
        vector=np.squeeze(vector[:]).transpose()

        vector_mean = np.mean(vector, axis=1)
        cycle = np.linspace(0, 100, len(vector))
        std_err_vector = (np.std(vector, axis=1, ddof=1) / np.sqrt(np.size(vector, axis=1)))
        curve1 = vector_mean + std_err_vector
        curve2 = vector_mean - std_err_vector
        x2 = np.concatenate((cycle, np.flip(cycle)))
        inBetween = np.concatenate((curve1, np.flip(curve2)))
        plt.fill(x2, inBetween, 'b', alpha=0.7, label=str(vector))
        plt.show
        plt.plot(cycle, vector_mean, 'k', linewidth=2)
        if index==n_graphs:
            plt.legend()
        plt.xlabel("Cycle [%]")
        plt.ylabel("HJM/BM flex(+)/ext(-) [Nm/kg]")
        plt.title("Sagittal plane")


    def graph_spm(self, vec1, vec2):
        ttest = spm1d.stats.ttest_paired(vec1, vec2)
        ttest.plot()

        plt.xlabel('Cycle [%]')
        plt.ylabel('SPM{t}')
        plt.title('Test set')
        plt.ylim(-3.5,3.5)
        plt.axhline(y=-ttest.inference(alpha=0.05).zstar, color='r', linestyle='--')
        plt.axhline(y=ttest.inference(alpha=0.05).zstar, color='r', linestyle='--')

        plt.show()
        print(ttest.inference(alpha=0.05))


    def execute(self):
        index=1
        n_graphs=2

        my_ResultAnalysis.metrics_calculation()
        my_ResultAnalysis.graph_mean_sd(self.y_test,index, n_graphs)

        index=index+1

        my_ResultAnalysis.graph_mean_sd(self.y_pred_test, index, n_graphs)
        plt.figure()
        my_ResultAnalysis.graph_spm(self.y_pred_test, self.y_test)

        index=0
        return self.rrmse_test, self.r_value_test, self.mae_test