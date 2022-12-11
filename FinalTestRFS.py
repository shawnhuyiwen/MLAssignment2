import pandas as pd
import numpy as np
from tensorflow import keras

test_set_dir = 'testDatasetExample.xls'
output_dir = 'predictedRFS.csv'
test_df = pd.read_excel(test_set_dir)

out_dict = {'ID':[], 'RFSPredicted':[]}

poorly_coor = ['Age', 'ER', 'PgR', 'TrippleNegative', 'Proliferation', 'HistologyType', 'LNStatus', 
               'original_shape_Elongation', 'original_shape_SurfaceVolumeRatio', 'original_firstorder_10Percentile', 
               'original_firstorder_Minimum', 'original_firstorder_Skewness', 'original_glcm_Autocorrelation', 
               'original_glcm_Correlation', 'original_glcm_DifferenceEntropy', 'original_glcm_DifferenceVariance', 
               'original_glcm_JointAverage', 'original_glcm_MCC', 'original_glcm_SumAverage', 'original_gldm_DependenceVariance', 
               'original_gldm_HighGrayLevelEmphasis', 'original_gldm_LargeDependenceHighGrayLevelEmphasis', 
               'original_gldm_LargeDependenceLowGrayLevelEmphasis', 'original_gldm_LowGrayLevelEmphasis', 
               'original_gldm_SmallDependenceHighGrayLevelEmphasis', 'original_glrlm_GrayLevelNonUniformityNormalized', 
               'original_glrlm_GrayLevelVariance', 'original_glrlm_HighGrayLevelRunEmphasis', 'original_glrlm_LongRunEmphasis', 
               'original_glrlm_LongRunHighGrayLevelEmphasis', 'original_glrlm_LongRunLowGrayLevelEmphasis', 
               'original_glrlm_LowGrayLevelRunEmphasis', 'original_glrlm_RunEntropy', 'original_glrlm_RunLengthNonUniformityNormalized', 
               'original_glrlm_RunVariance', 'original_glrlm_ShortRunEmphasis', 'original_glrlm_ShortRunHighGrayLevelEmphasis', 
               'original_glrlm_ShortRunLowGrayLevelEmphasis', 'original_glszm_GrayLevelNonUniformity', 
               'original_glszm_GrayLevelNonUniformityNormalized', 'original_glszm_GrayLevelVariance', 'original_glszm_HighGrayLevelZoneEmphasis', 
               'original_glszm_LargeAreaHighGrayLevelEmphasis', 'original_glszm_LowGrayLevelZoneEmphasis', 
               'original_glszm_SizeZoneNonUniformity', 'original_glszm_ZoneVariance', 'original_ngtdm_Coarseness', 
               'original_ngtdm_Complexity', 'original_ngtdm_Strength']

test_df = test_df.drop(poorly_coor, axis=1)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
test_df.iloc[:,1:] = scaler.fit_transform(test_df.iloc[:,1:])

model=keras.models.Sequential()
model.add(keras.layers.Dense(32, input_shape=(test_df.iloc[:,1:].shape[1],), activation="relu"))
model.add(keras.layers.Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD(learning_rate=0.001), metrics=[keras.metrics.MeanSquaredError()])

model.load_weights('./regression_models/fold_5_model_one_layer_1.hdf5')

pred = model.predict(np.array(test_df.iloc[:,1:]))

predicted = []
for elem in pred:
    predicted.append(round(elem[0]))
    
for id in test_df['ID']:
    out_dict['ID'].append(id)

out_dict['RFSPredicted'] = predicted

out_df = pd.DataFrame(out_dict)
out_df.to_csv(output_dir, index=False)