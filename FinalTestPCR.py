# import all the packages
import pandas as pd
import numpy as np
from tensorflow import keras

test_set_dir = 'testDatasetExample.xls'
output_dir = 'predictedPCR.csv'
test_df = pd.read_excel(test_set_dir)

invalid = []
invalid_ID = []

# delete invalid data
for index, row in test_df.iterrows():
    for key in row.keys():
        if row[key] == 999:
            invalid.append(index)
            invalid_ID.append(row['ID'])
            
test_df = test_df.drop(index = invalid)

# construct the output dict
out_dict = {'ID':[], 'PCRPredicted':[]}

for id in invalid_ID:
    out_dict['ID'].append(id)
    out_dict['PCRPredicted'].append('Invalid data provided')

# delete the poorly coorelated features
poorly_coor = ['Age', 'original_shape_Elongation', 'original_shape_SurfaceArea', 
               'original_firstorder_90Percentile', 'original_firstorder_Energy', 
               'original_firstorder_Kurtosis', 'original_firstorder_Maximum', 
               'original_firstorder_Range', 'original_firstorder_TotalEnergy', 
               'original_glcm_Autocorrelation', 'original_glcm_ClusterShade', 
               'original_glcm_Correlation', 'original_glcm_Imc1', 'original_glcm_JointAverage', 
               'original_glcm_MCC', 'original_glcm_SumAverage', 'original_gldm_HighGrayLevelEmphasis', 
               'original_gldm_LargeDependenceLowGrayLevelEmphasis', 'original_gldm_LowGrayLevelEmphasis', 
               'original_glrlm_GrayLevelNonUniformity', 'original_glrlm_RunLengthNonUniformity', 
               'original_glszm_GrayLevelNonUniformity', 'original_glszm_HighGrayLevelZoneEmphasis', 
               'original_glszm_LargeAreaEmphasis', 'original_glszm_LargeAreaHighGrayLevelEmphasis', 
               'original_glszm_LargeAreaLowGrayLevelEmphasis', 'original_glszm_LowGrayLevelZoneEmphasis', 
               'original_glszm_SizeZoneNonUniformity', 'original_glszm_SizeZoneNonUniformityNormalized', 
               'original_glszm_SmallAreaEmphasis', 'original_glszm_SmallAreaHighGrayLevelEmphasis', 
               'original_glszm_SmallAreaLowGrayLevelEmphasis', 'original_glszm_ZoneEntropy', 'original_glszm_ZonePercentage', 
               'original_glszm_ZoneVariance', 'original_ngtdm_Coarseness']

test_df = test_df.drop(poorly_coor, axis=1)

from sklearn.preprocessing import StandardScaler

# perform zero means normalisation on the features
scaler = StandardScaler()
test_df.iloc[:,1:] = scaler.fit_transform(test_df.iloc[:,1:])

# set up the ANN model
model=keras.models.Sequential()
model.add(keras.layers.Dense(32, input_shape=(test_df.iloc[:,1:].shape[1],), activation="relu"))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.SGD(learning_rate=0.1), metrics=['accuracy'])

# load the weights from the best trained model
model.load_weights('./classification_models/fold_1_model_one_layer_1.hdf5')

pred = model.predict(np.array(test_df.iloc[:,1:]))

# store the prediction into the list
predicted = []
for elem in pred:
    predicted.append(elem[0])
    
for id in test_df['ID']:
    out_dict['ID'].append(id)

rounded = [round(x) for x in predicted]
out_dict['PCRPredicted'] = out_dict['PCRPredicted'] + rounded

out_df = pd.DataFrame(out_dict)
out_df.to_csv(output_dir, index=False)


