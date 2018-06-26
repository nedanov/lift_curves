import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import datetime

#takes in a sorted data frame holding the actuals, the predicted values (sorted descending) and the percentile of each obsevation
#and returns a new dataframe with all of the appropriate calculations
def lift_calculations(df):

	#adding a sample counter variable
    df['sample_num'] = range(len(df)) 
    
    #adding cumulative sum of actual target
    df['actual_sum'] = df['actual'].cumsum()
    
    #column tracking the percentage of samples of total covered
    df['per_sample_covered'] = ((df['sample_num']+1)*100)/len(df)
    
    #percentage of all positives captured
    df['per_pos_captured'] = (df['actual_sum']/(len(df[df['actual']==1])))*100
    
    #proportion of positives captured from total
    df['prop_pos_captured_from_all'] = df['per_pos_captured']/df['per_sample_covered']

    return df

#creates a plot of cumulative positive gain 
#takes in a dataframe with all of the relevant statistics already calculated
def gain_plot(df,figsize=None,x_range=None,y_range=None,legend='on'):

	 #Plot of the cumulative capture of positives as we go across the deciles
    if figsize:
    	fig = plt.figure(figsize=figsize)
    plt.plot(pd.Series([0]).append(df['per_sample_covered']),
             pd.Series([0]).append(df['per_pos_captured']))  #pre-pending zeos to the front of the series for polotting purposes
    plt.plot([0,100],[0,100])
    plt.title('Cumulative True Positives Captured vs Random (Gain Curve)',fontsize=20)    
    plt.xlabel('% of Sample Covered',fontsize=15)
    plt.ylabel('% of True Positives Captured',fontsize=15)
    if x_range:
    	plt.xlim(x_range[0],x_range[1])
    if y_range:
    	plt.ylim(y_range[0],y_range[1])
    if legend=='on':
    	plt.legend(['Predictive Targeting','Random Targeting'],fontsize=12,loc=2)

def lift_plot(df,figsize=None,x_range=None,y_range=None,legend='on'):

	#Lift Curve Plot(at whatever percent of customers cutoff, the model is targeting X times better than random)
    #i.e. at the XX percentile, the response rate is Y times as good as it would be if targeting at random at the XX percentile
    if figsize:
    	fig = plt.figure(figsize=figsize)
    plt.plot(df['per_sample_covered'],df['prop_pos_captured_from_all'])
    plt.plot([df['per_sample_covered'].min(),100],[1,1])
    plt.title('Lift Curve',fontsize=20)
    plt.xlabel('% of Customers',fontsize=15)
    plt.ylabel('Lift',fontsize=15)
    if x_range:
    	plt.xlim(x_range[0],x_range[1])
    if y_range:
    	plt.ylim(y_range[0],y_range[1])
    if legend=='on':
    	plt.legend(['Predictive Targeting','Random Targeting'],fontsize=12)

#a function which takes in an array of predicted values and returns the percentile associated with each one
def percentile_gen(arr_y_pred):
	return np.array(pd.qcut(pd.Series(arr_y_pred).rank(method='first'),100,labels=range(1,101)))  #method = first is used in the case when there are a lot of 0s and overlapping of labels

#a function which takes in an array of actual test values and the model predicted values and stacks them together
#then sorts them and puts them into a dataframe
def data_prep(arr_y_test,arr_y_pred):

	#assigning each observation into a percentile
	percentiles = percentile_gen(arr_y_pred)

	#print(percentiles.shape)

	#joining all the pieces together
	data = np.hstack((arr_y_test.reshape((len(arr_y_test),1)),
					  arr_y_pred.reshape((len(arr_y_pred),1)),
					  percentiles.reshape((len(percentiles),1))))
	
	#converting to a data frame
	data_df = pd.DataFrame(data)
	data_df.columns = ['actual','prob','percentile']
	data_df.actual = data_df.actual.astype(int)
	data_df.prob = data_df.prob.astype('float64')
	
	#sorting by the probability
	data_df = data_df.sort_values(by='prob',ascending=False)

	#calculating lift metrics
	data_df = lift_calculations(data_df)

	return data_df

#a function which plots the lift curve for the model
def lift_curve(arr_y_test,arr_y_pred,figsize=None,x_range=None,y_range=None,legend='on'):

	data_df = data_prep(arr_y_test,arr_y_pred)

	#print(data_df.groupby('percentile').size())

	#lift curve plot
	lift_plot(data_df,figsize=figsize,x_range=x_range,y_range=y_range,legend=legend)
	plt.show()

#a function which plots the gain curve for the model
def gain_curve(arr_y_test,arr_y_pred,figsize=None,x_range=None,y_range=None,legend='on'):

	data_df = data_prep(arr_y_test,arr_y_pred)

	#gain curve plot
	gain_plot(data_df,figsize=figsize,x_range=x_range,y_range=y_range,legend=legend)
	plt.show()

#a function which returns two numpy arrays:
#the first one is the percent of samples covered (X-value)
#the second being the lift values for the correponding the sample (Y-value)
def lift_values_generator(arr_y_test,arr_y_pred):

	data_df = data_prep(arr_y_test,arr_y_pred)

	return data_df.per_sample_covered, data_df.prop_pos_captured_from_all

#a function which plots multiple lift curves all on the same graph
#the first parameter is the x axis which represents %of the sample covered
#the second parameter is a list of lists, where each one presents the lift
#curve for a particular model, the last parameter holds the labels for the lift
#curves in the corresponding order

def plot_lift_curves(percent_sample_covered,list_of_lift_metrics,labels,figsize=None,x_range=None,y_range=None,legend='on'):

	if figsize:
		plt.figure(figsize=figsize)

	#plotting the various model lift curves
	for i,lift_scores in enumerate(list_of_lift_metrics):
		plt.plot(percent_sample_covered,lift_scores)
	#base line plot for random guessing
	plt.plot([percent_sample_covered.min(),100],[1,1])
	
	#formats and labels
	plt.title('Lift Curves Comparison',fontsize=20)
	plt.xlabel('% of Customers',fontsize=15)
	plt.ylabel('Lift',fontsize=15)
	if x_range:
		plt.xlim(x_range[0],x_range[1])
	if y_range:
		plt.ylim(y_range[0],y_range[1])
	model_labels = labels + ['Random Guessing']
	if legend == 'on':
		plt.legend(model_labels,fontsize=12,loc='best')
