class cross_validator:
    """
    Class performing cross-validation

    The "execute" function of this class returns:
        - rRMSE, r and MAE for each time series (real vs predicted values) of each fold
        - Mean values of the metrcis above (for each fold)

    The "execute" function of this class also displays SPM graphs for each fold    

    """
    def __init__(self, model, X_train, y_train, n_splits, learning_rate, epochs):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.n_splits = n_splits
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.scores = []
        self.r_scores = []
        self.mae_scores = []
        self.nmae_scores = []
        self.rmse_scores = []
        self.rrmse_scores = []
        self.ttest_scores = []
        

    def cross_validation(self):
        kf = KFold(n_splits=self.n_splits)

        for train_idx, val_idx in kf.split(self.X_train):
            # Split the data into train and validation sets
            X_train_fold, X_val_fold = self.X_train[train_idx], self.X_train[val_idx]
            y_train_fold, y_val_fold = self.y_train[train_idx], self.y_train[val_idx]

            # Compile the model
            self.model.compile(loss='mean_squared_error',
                               optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                               metrics=[tf.keras.metrics.MeanSquaredError()])

            # Train the model on the train set and evaluate on the validation set
            history = self.model.fit(X_train_fold, y_train_fold, epochs=self.epochs, verbose=0)
            score = self.model.evaluate(X_val_fold, y_val_fold, verbose=0)
            self.scores.append(score[0])

            y_pred_val_fold = self.model.predict(X_val_fold)

            for i in range(len(y_pred_val_fold)):
                r, p_value = pearsonr(np.squeeze(y_pred_val_fold[i]), np.squeeze(y_val_fold[i]))
                self.r_scores.append(r)

                mae = mean_absolute_error(np.squeeze(y_pred_val_fold[i]), np.squeeze(y_val_fold[i]))
                self.mae_scores.append(mae)

                y_min = np.min(y_val_fold[i])
                y_max = np.max(y_val_fold[i])
                y_range = y_max - y_min
                nmae = (mae / y_range) * 100
                self.nmae_scores.append(nmae)

            # RMSE
            rmse = np.sqrt(score[0])
            self.rmse_scores.append(rmse)

            y_min = np.min(y_val_fold)
            y_max = np.max(y_val_fold)
            y_range = y_max - y_min
            rrmse = (rmse / y_range) * 100
            self.rrmse_scores.append(rrmse)

            # SPM
            ttest = spm1d.stats.ttest_paired(y_pred_val_fold, y_val_fold)
            self.ttest_scores.append(ttest)
        


    def metrics_calculation(self):

        # Calculate the mean of the evaluation metrics
        self.mean_r = np.mean(self.r_scores)
        self.mean_mae = np.mean(self.mae_scores)
        self.mean_rrmse = np.mean(self.rrmse_scores)

        mean_nmae = np.mean(self.nmae_scores)
        mean_rmse = np.mean(self.rmse_scores)

   
    def graph_spm(self):
        #Results cross validation
        fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))
        for i, axi in enumerate(ax.flatten()):
                self.ttest_scores[i].plot(ax=axi)
                axi.axhline(y=self.ttest_scores[i].inference(alpha=0.05).zstar, color='r', linestyle='--')
                axi.axhline(y=-self.ttest_scores[i].inference(alpha=0.05).zstar, color='r', linestyle='--')
                axi.set_xlabel('Cycle [%]', fontsize=16)
                axi.set_title('Fold '+str(i+1), fontsize=16)
                if max(self.ttest_scores[i].inference(alpha=0.05).z)>self.ttest_scores[i].inference(alpha=0.05).zstar:
                  axi.set_ylim([-self.ttest_scores[0].inference(alpha=0.05).zstar-0.5,max(self.ttest_scores[i].inference(alpha=0.05).z)+0.5])
                elif min(self.ttest_scores[i].inference(alpha=0.05).z)<-self.ttest_scores[i].inference(alpha=0.05).zstar:
                  axi.set_ylim([min(self.ttest_scores[i].inference(alpha=0.05).z)-0.5,self.ttest_scores[0].inference(alpha=0.05).zstar+0.5])
                else:
                  axi.set_ylim([-self.ttest_scores[0].inference(alpha=0.05).zstar-0.5, self.ttest_scores[0].inference(alpha=0.05).zstar+0.5]) 
        plt.tight_layout()


    def execute(self):
      my_CrossValidator.cross_validation()
      my_CrossValidator.metrics_calculation()
      my_CrossValidator.graph_spm()
      return self.rrmse_scores, self.r_scores, self.mae_scores, self.mean_rrmse, self.mean_r, self.mean_mae, self.ttest_scores