class model_implementation:
    """
    Class implementing the LSTM model used for predictions 

    The "execute" function of this class returns:
        - Training history of the model
        - Predictions on trainign data and on test data
        - The trained model itself
    """
    
    def __init__(self, X_train, X_test, y_train):
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        


    def build_model(self, lstm_units, dropout_rate_1, dense_units, 
                    dropout_rate_2, learning_rate):
        model = Sequential([
            LSTM(units=lstm_units, return_sequences=True, input_shape=(self.X_train.shape[1], self.X_train.shape[2])),
            Dropout(dropout_rate_1),
            Dense(units=dense_units),
            Dropout(dropout_rate_2),
            Dense(units=1)
        ])
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate),
                      metrics=[tf.keras.metrics.MeanSquaredError()])
        self.model=model
    def train(self, validation_split, epochs):
        return self.model.fit(self.X_train, self.y_train, validation_split=validation_split, epochs=epochs)
    
    def predict(self):
      y_pred_test=self.model.predict(self.X_test)
      y_pred_train=self.model.predict(self.X_train)
      return y_pred_test, y_pred_train


    def execute_model(self, lstm_units, dropout_rate_1, dense_units, 
                    dropout_rate_2, learning_rate, validation_split, 
                    epochs, model_out):  
                    
      my_lstm.build_model(lstm_units, dropout_rate_1, dense_units, 
                    dropout_rate_2, learning_rate)
      history=my_lstm.train(validation_split, epochs)
      y_pred_test, y_pred_train=my_lstm.predict()
      if model_out==1:
        return history, y_pred_test, y_pred_train, self.model
      else:
        return history, y_pred_test, y_pred_train