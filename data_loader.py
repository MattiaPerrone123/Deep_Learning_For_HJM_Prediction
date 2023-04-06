class data_loader:

  """
    Class processing .xls files exported from Visual3d and Mokka

    The "execute" function of this class returns:
        - Normalized time series of HJM, KJA and GRF
        - A sorted list of Mokka files.
  """
  
  def __init__(self):
    self.HJM_flexion=pd.DataFrame()
    self.GRF=pd.DataFrame()
    self.HJM_flexion_fin=pd.DataFrame()
    self.KJA_flexion=pd.DataFrame()
    self.markers=pd.DataFrame()
    self.count=0
    


  def import_HJM_V3D(self, path, value_to_drop):
    files_V3D = [f for f in listdir(path) if isfile(join(path, f))]
    files_sorted_V3D=sorted(files_V3D)

    for i in files_sorted_V3D:
      if "HJM_flexion" in i:
       temp=pd.read_excel(path+"/"+i, names=[str(i[:-5])+"_Trial 1",str(i[:-5])+"_Trial 2",str(i[:-5])+"_Trial 3"])
       self.HJM_flexion=pd.concat([self.HJM_flexion, temp], axis=1)    
    self.HJM_flexion.drop(value_to_drop, axis=1, inplace=True)   
    return self.HJM_flexion



  def import_preprocess_input_mokka(self, path):
    files_mokka = [f for f in listdir(path_mokka) if isfile(join(path_mokka, f))]
    self.files_sorted_mokka=sorted(files_mokka)
    return self.files_sorted_mokka



  def process_file(self, file_path):
    self.temp=pd.read_excel(file_path, names=["Frame number","Th_x","Th_y","Th_z","K_lat_x","K_lat_y","K_lat_z", "Sh_x","Sh_y","Sh_z"])
    dataframe=self.temp["Frame number"]=="Time"
    self.start_markers=7
    self.end_markers=np.where(list(dataframe))[0][1]-2
    self.length_markers=self.end_markers-self.start_markers

    self.start_GRF=np.where(list(dataframe))[0][1]+2
    self.end_GRF=dataframe.shape[0]
    self.length_GRF=self.end_GRF-self.start_GRF

    for i in range(self.temp.shape[1]):
        self.temp.iloc[self.start_markers:self.end_markers,i] = self.temp.iloc[self.start_markers:self.end_markers,i].astype(float)
    self.temp.iloc[self.start_GRF:self.end_GRF,1] = self.temp.iloc[self.start_GRF:self.end_GRF,1].astype(float)
  

  def fix_marker_trajectories(self):
    values_to_fix = ["Th_y", "K_lat_y", "Sh_y"]
    for value in values_to_fix:
         for obs in range(self.start_markers, self.end_markers):
            if -1000 < self.temp.loc[obs, value] < -10000:
              self.temp.loc[obs, value] /= 1000
            elif -10000 < self.temp.loc[obs, value] < -100000:
              self.temp.loc[obs, value] /= 10000  
            elif self.temp.loc[obs, value] < -100000:
              self.temp.loc[obs, value] /= 100000 
         

  def fix_GRF(self):
       for index in range(self.start_GRF, self.end_GRF):
           if -100 < self.temp.iloc[index, 1] < -1000:
              self.temp.iloc[index, 1] *= 1000
           elif -100 > self.temp.iloc[index, 1] > -1000:
              self.temp.iloc[index, 1] *= 1000  
          


  def reduce_vector(self, vec, length):
        return np.interp(np.linspace(0, 1, length), np.linspace(0, 1, len(vec)), vec)


  def reduce_marker_trajectories(self):
       for k in range(self.temp.shape[1]):
            temp_np = np.array(self.temp.iloc[self.start_markers:self.end_markers, k].astype(float))
            markers_temp = self.reduce_vector(temp_np, 100)
            self.markers = pd.concat([self.markers, pd.DataFrame(markers_temp)], axis=1, ignore_index=True)
       self.markers.columns = ["Frame number", "Th_x", "Th_y", "Th_z", "K_lat_x", "K_lat_y", "K_lat_z", "Sh_x", "Sh_y", "Sh_z"]
   

  def compute_KJA_flexion(self):
      #ang1
      v1_z = np.array(self.markers["Th_z"] - self.markers["K_lat_z"])
      v1_x = np.array(self.markers["Th_x"] - self.markers["K_lat_x"])
      ratio_1 = abs(v1_z) / abs(v1_x)
      ang1 = [(math.atan(i) * 180) /math.pi for i in ratio_1]

      # ang2
      v2_z = np.array(self.markers["Sh_z"] - self.markers["K_lat_z"])
      v2_x = np.array(self.markers["Sh_x"] - self.markers["K_lat_x"])
      ratio_2 = abs(v2_z) / abs(v2_x)
      ang2 = [(math.atan(i) * 180) /math.pi for i in ratio_2]

      angx = [180 - (ang1[i] + ang2[i]) for i in range(len(ang1))]

      #ang1
      v1_z = np.array(self.markers["Th_z"] - self.markers["K_lat_z"])
      v1_y = np.array(self.markers["Th_y"] - self.markers["K_lat_y"])
      ratio_1 = abs(v1_z) / abs(v1_y)
      ang1 = [(math.atan(i) * 180) /math.pi for i in ratio_1]

      # ang2
      v2_z = np.array(self.markers["Sh_z"] - self.markers["K_lat_z"])
      v2_y = np.array(self.markers["Sh_y"] - self.markers["K_lat_y"])
      ratio_2 = abs(v2_z) / abs(v2_y)
      ang2 = [(math.atan(i) * 180) /math.pi for i in ratio_2]

      angy = [180 - (ang1[i] + ang2[i]) for i in range(len(ang1))]

      if np.mean(angx) > np.mean(angy):
         ang = angx
      else:
         ang = angy

      self.KJA_flexion=pd.concat([self.KJA_flexion, pd.DataFrame(ang)], axis=1, ignore_index=True) 
      self.markers=pd.DataFrame()

  def process_HJM_flexion(self, count):
    first_frame_cut=self.temp.iloc[0,1]
    last_frame_cut=first_frame_cut+self.length_markers
    HJM_flexion_cut=self.HJM_flexion.iloc[first_frame_cut:last_frame_cut,count]
    HJM_flexion_np=np.array(HJM_flexion_cut.astype(float))

    HJM_flexion_red = self.reduce_vector(HJM_flexion_np, 100)
    self.HJM_flexion_fin=pd.concat([self.HJM_flexion_fin, pd.DataFrame(HJM_flexion_red)], axis=1, ignore_index=True)
    
   

  def process_grf(self):
    GRF_temp = self.temp.iloc[self.start_GRF:self.end_GRF, 1]
    GRF_np = np.array(GRF_temp.astype(float))
    GRF_red = self.reduce_vector(GRF_np, self.length_markers)
    GRF_fin = self.reduce_vector(GRF_red, 100)
    
    self.GRF=pd.concat([self.GRF, pd.DataFrame(GRF_fin)], axis=1, ignore_index=True) 



  def final_process_data(self):
      self.GRF=self.GRF.round(2)
      if self.GRF.min().min()<-1000: 
          self.GRF/=1000

      self.GRF.columns = self.files_sorted_mokka
      self.KJA_flexion.columns=self.files_sorted_mokka
      self.HJM_flexion_fin.columns=self.files_sorted_mokka

     


  def refining_GRF_values(self, values_to_fix):
      for column_name in values_to_fix:
          for index in range(100):
              if (self.GRF.loc[index, column_name] > -100) and (self.GRF.loc[index, column_name] < 0):
                  self.GRF.loc[index, column_name] *= 1000
                  if (self.GRF.loc[index+1, column_name] < -100):
                      self.GRF.loc[index+1, column_name] = self.GRF.loc[index, column_name]
                  if (self.GRF.loc[index-1, column_name] > -900):
                      self.GRF.loc[index-1, column_name] = self.GRF.loc[index, column_name]
      

  def refining_kja(self):
      self.KJA_flexion.iloc[78,53]=98
      self.KJA_flexion.iloc[79,53]=96
      self.KJA_flexion.iloc[80,53]=94
      self.KJA_flexion.iloc[36,140]=(self.KJA_flexion.iloc[35,140]+self.KJA_flexion.iloc[37,140])/2

      for i in range(self.KJA_flexion.shape[1]):
          self.KJA_flexion.iloc[:,i]=self.KJA_flexion.iloc[:,i]-self.KJA_flexion.iloc[0,i]


  def execute(self, path_all, path_mokka, value_to_drop):
    HJM_flexion=my_dataset.import_HJM_V3D(path_all, value_to_drop) 
    files_sorted_mokka=my_dataset.import_preprocess_input_mokka(path_mokka)

    for files in files_sorted_mokka:
       my_dataset.process_file(path_mokka+"/"+files)
       my_dataset.fix_marker_trajectories()
       my_dataset.fix_GRF()
       my_dataset.reduce_marker_trajectories()
       
       my_dataset.compute_KJA_flexion()
       my_dataset.process_HJM_flexion(self.count)
       my_dataset.process_grf()

       self.count=self.count+1    

    my_dataset.final_process_data()     
    my_dataset.refining_GRF_values(values_to_fix)
    my_dataset.refining_kja()

    return self.HJM_flexion_fin, self.GRF, self.KJA_flexion, self.files_sorted_mokka 
