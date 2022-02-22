from __future__ import absolute_import, division, print_function
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns
import missingno as msno

from sklearn import linear_model
import pandas as pd
from pandas import Series
from pandas import DataFrame
#from pandas import TimeGrouper
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
import math   # yep! going to a bit of maths later!!
from scipy import stats as st # and some stats

#Decision Tree
raw_dataset = pd.read_csv("database.csv.csv")

dataset = raw_dataset.copy()
dataset.tail()
print (dataset.tail())

#sns.pairplot(train[['SWVL1','TOT_MSL','TOT_SP','TOT_SSTK','TOT_VAR_2T','TOT_VAR_100U','TOT_VAR_100V','TOT_VAR_10U','TOT_VAR_10V']], diag_kind="kde")
dataset.isna().sum()
dataset = dataset.dropna()

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#sns.pairplot(train_dataset[['swvl1','d2m','t2m','evabs','sp','sshf','tp']], diag_kind="kde")

train2_dataset=train_dataset[['swvl2']]
test2_dataset=test_dataset[['swvl2']]

#train_dataset.skew(), train_dataset.kurt()
#msno.heatmap(train_dataset)
#plt.figure(figsize = (12,8))
#sns.distplot(train_dataset.skew(),color='blue',axlabel ='Skewness')
#plt.show()

#plt.figure(figsize = (12,8))
#sns.distplot(train_dataset.kurt(),color='r',axlabel ='Kurtosis',norm_hist= False, kde = True,rug = False)
#plt.hist(train.kurt(),orientation = 'vertical',histtype = 'bar',label ='Kurtosis', color ='blue')
#plt.show()

correlation = train_dataset.corr()
print(correlation['swvl1'].sort_values(ascending = False),'\n')

# f , ax = plt.subplots(figsize = (14,12))
# plt.title('Correlation of Numeric Features',y=1,size=16)
# sns.heatmap(correlation,square = True,  vmax=0.8)

k= 11
cols = correlation.nlargest(k,'swvl1')['swvl1'].index
print(cols)
cm = np.corrcoef(train_dataset[cols].values.T)
f , ax = plt.subplots(figsize = (14,12))
# sns_hist=sns.heatmap(cm, vmax=.8, linewidths=0.01,square=True,annot=True,cmap='viridis',
            # linecolor="white",xticklabels = cols.values ,annot_kws = {'size':12},yticklabels = cols.values)

# plt.title('Correlation Heatmap');
# sns_plot = sns_hist.get_figure()
# sns_plot.savefig("output1_heatmap.png")
def plot_dist_col(column):
    '''plot dist curves for train and test data for the given column name'''
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.distplot(train_dataset[column].dropna(), color='green', ax=ax).set_title(column, fontsize=16)
    sns.distplot(test_dataset[column].dropna(), color='purple', ax=ax).set_title(column, fontsize=16)
    plt.xlabel(column, fontsize=15)
    plt.legend(['train', 'test'])
    plt.show()

#plot_dist_col('swvl1')


## Function to reduce the DF size
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

# checking missing data
total = train_dataset.isnull().sum().sort_values(ascending = False)
percent = (train_dataset.isnull().sum()/train_dataset.isnull().count()*100).sort_values(ascending = False)
missing_weather_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

# Find correlations with the target and sort
correlations = train_dataset.corr()['swvl1'].sort_values()

# Display correlations
print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))

corrs = train_dataset.corr()
corrs



train_dataset= train_dataset.drop('swvl2', axis = 1)
#train_dataset= train_dataset.drop('swvl3', axis = 1)
#train_dataset= train_dataset.drop('swvl4', axis = 1)
train_dataset= train_dataset.drop('u10', axis = 1)
train_dataset= train_dataset.drop('sp', axis = 1)

test_dataset= test_dataset.drop('swvl2', axis = 1)
#test_dataset= test_dataset.drop('swvl3', axis = 1)
#test_dataset= test_dataset.drop('swvl4', axis = 1)
test_dataset= test_dataset.drop('u10', axis = 1)
test_dataset= test_dataset.drop('sp', axis = 1)

train_stats = train_dataset.describe()
train_stats.pop("swvl1")
train_stats = train_stats.transpose()
print(train_stats)


train_labels2 = train2_dataset.pop('swvl2')
test_labels2 = test2_dataset.pop('swvl2')

train_labels = train_dataset.pop('swvl1')
test_labels = test_dataset.pop('swvl1')



def norm(x):
  return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
#opt = Adam(lr=1e-3, decay=1e-3 / 200)
def build_model():
  model = keras.Sequential([
    layers.Dense(150, activation=tf.nn.relu,input_shape=[len(train_dataset.keys())]),
    layers.Dense(80, activation='sigmoid'),
    #layers.Dense(50, activation='sigmoid'),
    layers.Dense(1)
  ])

  #optimizer = tf.keras.optimizers.Adam(0.00146)
  #optimizer = tf.keras.optimizers.RMSprop(0.001)
  optimizer = tf.keras.optimizers.SGD(0.01)
  #optimizer = tf.keras.optimizers.SGD(momentum=0.01, nesterov=True)
  #optimizer = tf.keras.optimizers.Adagrad
  #optimizer = tf.keras.optimizers.nadam
  #optimizer = tf.keras.optimizers.Adamax
  #optimizer = tf.keras.optimizers.Adam(amsgrad=True)
  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model
  
model = build_model()
model.summary()

example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
example_result
print(example_result)

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.' , end='')

EPOCHS = 300

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])
  
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()
print(hist.tail())

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [swvl1]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,0.05])
  plt.legend()
  plt.show()
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$swvl1^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,0.05])
  plt.legend()
  plt.show()


plot_history(history) 

model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
#loss,mean_absolute_error,val_mean_squared_error,val_mean_absolute_error,mean_squared_error,val_loss
#early_stop = keras.callbacks.EarlyStopping(monitor='loss')

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)

loss, mae, mse = model.evaluate(normed_train_data, train_labels, verbose=0)


print("Training set Mean Abs Error: {:5.2f} SWVL1".format(mae))
print("Training set loss: {:5.2f} SWVL1".format(loss))

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

model.save('modelsoil.h5')

print("Testing set Mean Abs Error: {:5.2f} SWVL1".format(mae))
print("Testing set loss: {:5.2f} SWVL1".format(loss))

test_predictions = model.predict(normed_test_data).flatten()

train_predictions = model.predict(normed_train_data).flatten()

#my_submission = pd.DataFrame({'swvl1': train_predictions})
my_submission = pd.DataFrame(train_predictions.T,
	columns=['swvl1'])
	
my_submission2 = pd.DataFrame(test_predictions.T,
	columns=['swvl1'])	
#d = {train_dataset,my_submission}
print("Debut: {:5.2f} SWVL1".format(mae))

print(train_dataset.head())

print(my_submission.head())


print("Fin: {:5.2f} SWVL1".format(mae))
train_dataset2 = train_dataset.copy()
train_dataset2.reset_index(drop=True, inplace=True)
my_submission.reset_index(drop=True, inplace=True)

df_train = pd.concat([train_dataset2, my_submission], axis=1) 
df_train.to_csv('voir_submission.csv',index=False)

test_dataset2 = test_dataset.copy()
test_dataset2.reset_index(drop=True, inplace=True)
my_submission2.reset_index(drop=True, inplace=True)

df_test = pd.concat([test_dataset2, my_submission2], axis=1) 
df_test.to_csv('voir_submission2.csv',index=False) 

#df_train = pd.DataFrame(data=d)

print(my_submission.info())

print(train_dataset2.info())

#df_train = pd.DataFrame(np.array([train_dataset, my_submission]))
#                   columns=['d2m','t2m','evabs','sp','sshf','tp', 'swvl1'])
df_train.tail()
print (df_train.tail())

trainp_stats = df_train.describe()
trainp_stats = trainp_stats.transpose()
print(trainp_stats)

print(df_train.info())

def normp(x):
  return (x - trainp_stats['mean']) / trainp_stats['std']


normed_swl1_data = normp(df_train)
normed_testswl1_data = normp(df_test)


def build_model2():
  model2 = keras.Sequential([
    layers.Dense(140, activation=tf.nn.relu,input_shape=[len(df_train.keys())]),
    layers.Dense(90, activation='sigmoid'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.Adam(0.00146)
  #optimizer = tf.keras.optimizers.RMSprop(0.001)
  model2.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model2
  
model2 = build_model2()
model2.summary()

example_batch = normed_swl1_data[:10]
example_result = model2.predict(example_batch)
example_result
print(example_result)

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.' , end='')

EPOCHS = 5

history = model2.fit(
  normed_swl1_data, train_labels2,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])
  
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()
print(hist.tail())

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [swvl1]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,0.05])
  plt.legend()
  plt.show()
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$swvl1^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,0.05])
  plt.legend()
  plt.show()


plot_history(history) 

model2 = build_model2()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
#early_stop = keras.callbacks.EarlyStopping(monitor='loss')

history = model2.fit(normed_swl1_data, train_labels2, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)

loss, mae, mse = model2.evaluate(normed_swl1_data, train_labels2, verbose=0)


print("Training set Mean Abs Error: {:5.2f} SWVL1".format(mae))
print("Training set loss: {:5.2f} SWVL1".format(loss))

loss, mae, mse = model2.evaluate(normed_testswl1_data, test_labels2, verbose=0)

model.save('modelsoil2.h5')

print("Testing set Mean Abs Error: {:5.2f} SWVL1".format(mae))
print("Testing set loss: {:5.2f} SWVL1".format(loss))

test_predictions2 = model2.predict(normed_testswl1_data).flatten()

train_predictions2 = model2.predict(normed_swl1_data).flatten()


plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [SWVL1]')
plt.ylabel('Predictions [SWVL1]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
plt.plot([-100, 100], [-100, 100])
plt.show()

error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [SWVL1]")
plt.ylabel("Count")
plt.show()

# Superimposed: model prediction (blue) vs reality (red)

# Chart Test
lenPreds = len(test_labels)
fig= plt.figure()
ax=fig.gca() # define axis
UlimReal = lenPreds
#UlimReal = 311511
LlimReal = 0
SampleSizeReal = UlimReal - LlimReal
plt.plot(range(SampleSizeReal), test_labels[LlimReal:UlimReal], 'r', linewidth=0.4, label='real')
#plt.show()

# Chart Predicted 
UlimPred = lenPreds
#UlimPred = 10000
LlimPred = 0
SampleSizePred = UlimPred - LlimPred
plt.plot(range(SampleSizePred), test_predictions[LlimPred:UlimPred], 'b', linewidth=0.4, label='predicted')
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

plt.show()

#Compute R-Square value for validation set
#ValR2Value = r2_score(test_labels[LlimReal:UlimReal],test_predictions[LlimPred:UlimPred])
#ValR2Value = r2_score(test_labels[LlimReal:2000],test_predictions[LlimPred:2000])
ValR2Value = r2_score(test_labels,test_predictions)
print("Validation Set R-Square=",ValR2Value)

train_predictions = model.predict(normed_train_data).flatten()

ValR2Value = r2_score(train_labels,train_predictions)
print("Train set R-Square=",ValR2Value)



plt.scatter(test_labels2, test_predictions2)
plt.xlabel('True Values [SWVL2]')
plt.ylabel('Predictions [SWVL2]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
plt.plot([-100, 100], [-100, 100])
plt.show()

error = test_predictions2 - test_labels2
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [SWVL2]")
plt.ylabel("Count")
plt.show()

# Superimposed: model prediction (blue) vs reality (red)

# Chart Test
lenPreds = len(test_labels2)
fig= plt.figure()
ax=fig.gca() # define axis
UlimReal = lenPreds
#UlimReal = 311511
LlimReal = 0
SampleSizeReal = UlimReal - LlimReal
plt.plot(range(SampleSizeReal), test_labels2[LlimReal:UlimReal], 'r', linewidth=0.4, label='real')
#plt.show()

# Chart Predicted 
UlimPred = lenPreds
#UlimPred = 10000
LlimPred = 0
SampleSizePred = UlimPred - LlimPred
plt.plot(range(SampleSizePred), test_predictions2[LlimPred:UlimPred], 'b', linewidth=0.4, label='predicted')
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

plt.show()

ValR2Value = r2_score(test_labels2,test_predictions2)
print("Validation2 Set R-Square=",ValR2Value)

train_predictions2 = model2.predict(normed_swl1_data).flatten()

ValR2Value = r2_score(train_labels2,train_predictions2)
print("Train2 set R-Square=",ValR2Value)
