class data_splitter:
    """
    Class performing dataset splitting

    The "execute" function of this class returns:
        - Training and test samples (both X and y), accordingly scaled
    
    """
    def __init__(self, HJM_flexion, KJA_flexion, GRF, n_inputs, files_sorted_mokka):
        self.HJM_flexion=HJM_flexion
        self.KJA_flexion=KJA_flexion
        self.GRF=GRF
        self.n_inputs=n_inputs
        self.files_sorted_mokka=files_sorted_mokka

    
    def split_data(self):
    
        control_cols = [i for i in range(self.KJA_flexion.shape[1]) if "Control" in self.files_sorted_mokka[i]]
        fais_cols = [i for i in range(self.KJA_flexion.shape[1]) if "FAIS" in self.files_sorted_mokka[i]]
        postop_cols = [i for i in range(self.KJA_flexion.shape[1]) if "Postop" in self.files_sorted_mokka[i]]

        num_control = len(control_cols)
        num_fais =len(fais_cols)
        num_postop = len(postop_cols)

        threshold_control=int(num_control*0.8)-2
        threshold_fais=int(num_fais*0.8)+2
        threshold_postop=int(num_postop*0.8)+1

        # KJA
        KJA_train_control = self.KJA_flexion.iloc[:, control_cols].iloc[:, :threshold_control]
        KJA_train_fais = self.KJA_flexion.iloc[:, fais_cols].iloc[:, :threshold_fais]
        KJA_train_postop = self.KJA_flexion.iloc[:, postop_cols].iloc[:, :threshold_postop]

        KJA_test_control = self.KJA_flexion.iloc[:, control_cols].iloc[:, threshold_control:]
        KJA_test_fais = self.KJA_flexion.iloc[:, fais_cols].iloc[:, threshold_fais:]
        KJA_test_postop = self.KJA_flexion.iloc[:, postop_cols].iloc[:, threshold_postop:]

        # GRF
        GRF_train_control = self.GRF.iloc[:, control_cols].iloc[:, :threshold_control]
        GRF_train_fais = self.GRF.iloc[:, fais_cols].iloc[:, :threshold_fais]
        GRF_train_postop = self.GRF.iloc[:, postop_cols].iloc[:, :threshold_postop]

        GRF_test_control = self.GRF.iloc[:, control_cols].iloc[:, threshold_control:]
        GRF_test_fais = self.GRF.iloc[:, fais_cols].iloc[:, threshold_fais:]
        GRF_test_postop = self.GRF.iloc[:, postop_cols].iloc[:, threshold_postop:]

        # HJM
        HJM_train_control = self.HJM_flexion.iloc[:, control_cols].iloc[:, :threshold_control]
        HJM_train_fais = self.HJM_flexion.iloc[:, fais_cols].iloc[:, :threshold_fais]
        HJM_train_postop = self.HJM_flexion.iloc[:, postop_cols].iloc[:, :threshold_postop]

        HJM_test_control = self.HJM_flexion.iloc[:, control_cols].iloc[:, threshold_control:]
        HJM_test_fais = self.HJM_flexion.iloc[:, fais_cols].iloc[:, threshold_fais:]
        HJM_test_postop = self.HJM_flexion.iloc[:, postop_cols].iloc[:, threshold_postop:]


    
        self.KJA_train_data=np.array(pd.concat([KJA_train_control, KJA_train_fais, KJA_train_postop], axis=1))
        self.KJA_test_data=np.array(pd.concat([KJA_test_control, KJA_test_fais, KJA_test_postop], axis=1))

        self.GRF_train_data=np.array(pd.concat([GRF_train_control, GRF_train_fais, GRF_train_postop], axis=1))
        self.GRF_test_data=np.array(pd.concat([GRF_test_control, GRF_test_fais, GRF_test_postop], axis=1))

        self.HJM_train_data=np.array(pd.concat([HJM_train_control, HJM_train_fais, HJM_train_postop], axis=1))
        self.HJM_test_data=np.array(pd.concat([HJM_test_control, HJM_test_fais, HJM_test_postop], axis=1))


    def train_test_data(self):
        
        first_dim_train=np.array(self.GRF_train_data).shape[1]
        second_dim_train=np.array(self.GRF_train_data).shape[0]
        self.X_train=np.empty((first_dim_train, second_dim_train, self.n_inputs))
        self.y_train=np.empty((first_dim_train, second_dim_train,1))

    
        self.X_train[:,:,0]=np.transpose(self.GRF_train_data)
        self.X_train[:,:,1]=np.transpose(self.KJA_train_data)

    
        self.y_train=np.empty((np.array(self.HJM_train_data).shape[1], np.array(self.HJM_train_data).shape[0],1))
        self.y_train[:,:,0]=np.transpose(self.HJM_train_data)

    
        first_dim_test=np.array(self.GRF_test_data).shape[1]
        second_dim_test=np.array(self.GRF_test_data).shape[0]
        self.X_test=np.empty((first_dim_test, second_dim_test, self.n_inputs))
        self.y_test=np.empty((first_dim_test, second_dim_test,1))

 
        self.X_test[:,:,0]=np.transpose(self.GRF_test_data)
        self.X_test[:,:,1]=np.transpose(self.KJA_test_data)

        
        self.y_test=np.empty((np.array(self.HJM_test_data).shape[1], np.array(self.HJM_test_data).shape[0],1))
        self.y_test[:,:,0]=np.transpose(self.HJM_test_data)



    def scaling_shuffling(self):
        X_train_s = np.empty((self.X_train.shape[0], self.X_train.shape[1], self.X_train.shape[2]))
        X_test_s = np.empty((self.X_test.shape[0], self.X_test.shape[1], self.X_test.shape[2]))
        s = MinMaxScaler()
        for i in range(self.n_inputs):
            X_train_s[:,:,i] = s.fit_transform(self.X_train[:,:,i])
            X_test_s[:,:,i] = s.transform(self.X_test[:,:,i])


        X_train_s, y_train = shuffle(X_train_s, self.y_train, random_state=15) 
        X_test_s, y_test = shuffle(X_test_s, self.y_test, random_state=15) 

        return X_train_s, self.y_train, X_test_s, self.y_test


    def execute(self):
        my_data_splitting.split_data()
        my_data_splitting.train_test_data()
        return my_data_splitting.scaling_shuffling()