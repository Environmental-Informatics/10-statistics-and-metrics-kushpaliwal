#!/bin/env python
# Created on March 25, 2020
#  by Keith Cherkauer
# Modified By: Kush Paliwal on 21st April 2020
#
# This script serves as the solution set for assignment-10 on descriptive
# statistics and environmental informatics.  See the assignment documention 
# and repository at:
# https://github.com/Environmental-Informatics/assignment-10.git for more
# details about the assignment.
#
#
import pandas as pd
import scipy.stats as stats
import numpy as np

def ReadData( fileName ):
    """This function takes a filename as input, and returns a dataframe with
    raw data read from that file in a Pandas DataFrame.  The DataFrame index
    should be the year, month and day of the observation.  DataFrame headers
    should be "agency_cd", "site_no", "Date", "Discharge", "Quality". The 
    "Date" column should be used as the DataFrame index. The pandas read_csv
    function will automatically replace missing values with np.NaN, but needs
    help identifying other flags used by the USGS to indicate no data is 
    availabiel.  Function returns the completed DataFrame, and a dictionary 
    designed to contain all missing value counts that is initialized with
    days missing between the first and last date of the file."""
    
    # define column names
    colNames = ['agency_cd', 'site_no', 'Date', 'Discharge', 'Quality']

    # open and read the file
    DataDF = pd.read_csv(fileName, header=1, names=colNames,  
                         delimiter=r"\s+",parse_dates=[2], comment='#',
                         na_values=['Eqp'])
    DataDF = DataDF.set_index('Date')
    
    # Replace negative values
    DataDF['Discharge'] = DataDF['Discharge'].mask(DataDF['Discharge']<0, np.nan)
    
    # quantify the number of missing values
    MissingValues = DataDF["Discharge"].isna().sum()
    
    return( DataDF, MissingValues )

def ClipData( DataDF, startDate, endDate ):
    """This function clips the given time series dataframe to a given range 
    of dates. Function returns the clipped dataframe and and the number of 
    missing values."""
    
    #Clip data within the time period
    DataDF.index = pd.to_datetime(DataDF.index)
    mask = (DataDF.index >= startDate) & (DataDF.index <= endDate)
    DataDF = DataDF.loc[mask]
    
    # quantify the number of missing values
    MissingValues = DataDF["Discharge"].isna().sum()
    
    return( DataDF, MissingValues )

def CalcTqmean(Qvalues):
    """This function computes the Tqmean of a series of data, typically
       a 1 year time series of streamflow, after filtering out NoData
       values.  Tqmean is the fraction of time that daily streamflow
       exceeds mean streamflow for each year. Tqmean is based on the
       duration rather than the volume of streamflow. The routine returns
       the Tqmean value for the given data array."""
    
    # Drop None values
    Qvalues=Qvalues.dropna()
    
    # calculate the number of values bigger than yearly mean value
    Tqmean = ((Qvalues > Qvalues.mean()).sum())/len(Qvalues)
    
    return ( Tqmean )

def CalcRBindex(Qvalues):
    """This function computes the Richards-Baker Flashiness Index
       (R-B Index) of an array of values, typically a 1 year time
       series of streamflow, after filtering out the NoData values.
       The index is calculated by dividing the sum of the absolute
       values of day-to-day changes in daily discharge volumes
       (pathlength) by total discharge volumes for each year. The
       routine returns the RBindex value for the given data array."""
       
    # Drop None values
    Qvalues=Qvalues.dropna()
    
    dif=Qvalues.diff()
    dif=dif.dropna()
    
    # Calculate the sum of absolute values of day-to-day discharge change
    Total_abs=abs(dif).sum()
    
    # Total yearly discharge
    Total_dis=Qvalues.sum()
    
    # R-B index
    RBindex=Total_abs/Total_dis
    
    return ( RBindex )

def Calc7Q(Qvalues):
    """This function computes the seven day low flow of an array of 
       values, typically a 1 year time series of streamflow, after 
       filtering out the NoData values. The index is calculated by 
       computing a 7-day moving average for the annual dataset, and 
       picking the lowest average flow in any 7-day period during
       that year.  The routine returns the 7Q (7-day low flow) value
       for the given data array."""
       
    # Drop None values
    Qvalues=Qvalues.dropna()
    
    # Calculate the rolling 7-day minimum valuefor a year
    val7Q=Qvalues.rolling(window=7).mean().min()
    
    return ( val7Q )

def CalcExceed3TimesMedian(Qvalues):
    """This function computes the number of days with flows greater 
       than 3 times the annual median flow. The index is calculated by 
       computing the median flow from the given dataset (or using the value
       provided) and then counting the number of days with flow greater than 
       3 times that value.   The routine returns the count of events greater 
       than 3 times the median annual flow value for the given data array."""
    
    # Drop None values
    Qvalues=Qvalues.dropna()
    
    # get the 3*median value
    med3=3*Qvalues.median()
    
    # output median3x
    median3x=(Qvalues>med3).sum()
    
    return ( median3x )

def GetAnnualStatistics(DataDF):
    """This function calculates annual descriptive statistcs and metrics for 
    the given streamflow time series.  Values are retuned as a dataframe of
    annual values for each water year.  Water year, as defined by the USGS,
    starts on October 1."""
    
    # Seperate data into yearly data
    DataDF['W_year']=DataDF.index.to_period('A-Sep')
    DataDF['W_year']=DataDF['W_year']-1    
    
    # Define the name of columns
    cols=['site_no','Mean Flow','Peak Flow','Median Flow','Coeff Var','Skew','Tqmean','R-B Index','7Q','3xMedian']
   
    # Create new dataframe
    W_year = DataDF.resample('AS-OCT')
    yearly_avg=W_year.mean()
    WYDataDF=pd.DataFrame(0,index=yearly_avg.index, columns=cols)
    GroupD=W_year
    
    #Calculate descriptive values 
    WYDataDF['site_no']=GroupD['site_no'].min()
    WYDataDF['Mean Flow']=GroupD['Discharge'].mean()
    WYDataDF['Peak Flow']=GroupD['Discharge'].max()
    WYDataDF['Median Flow']=GroupD['Discharge'].median()
    WYDataDF['Coeff Var']=(GroupD['Discharge'].std()/GroupD['Discharge'].mean())*100
    WYDataDF['Skew']=GroupD['Discharge'].apply(lambda x: stats.skew(x))
    WYDataDF['Tqmean']=GroupD['Discharge'].apply(lambda x:CalcTqmean(x))
    WYDataDF['R-B Index']=GroupD['Discharge'].apply(lambda x:CalcRBindex(x))
    WYDataDF['7Q']=GroupD['Discharge'].apply(lambda x:Calc7Q(x))
    WYDataDF['3xMedian']=GroupD['Discharge'].apply(lambda x:CalcExceed3TimesMedian(x))
    return ( WYDataDF )

def GetMonthlyStatistics(DataDF):
    """This function calculates monthly descriptive statistics and metrics 
    for the given streamflow time series.  Values are returned as a dataframe
    of monthly values for each year."""
    
    # Define the name of columns and create a new dataframe
    cols=['site_no','Mean Flow','Coeff Var','Tqmean','R-B Index']
    
    # Devide the dataset into monthly data
    Mon_Data=DataDF.resample('MS').mean()
    MoDataDF=pd.DataFrame(0,index=Mon_Data.index,columns=cols)
    GroupD=DataDF.resample('MS')
    
    #Calculate descriptive values
    MoDataDF['site_no']=GroupD['site_no'].min()
    MoDataDF['Mean Flow']=GroupD['Discharge'].mean()
    MoDataDF['Coeff Var']=(GroupD['Discharge'].std()/GroupD['Discharge'].mean())*100
    MoDataDF['Tqmean']=GroupD['Discharge'].apply(lambda x:CalcTqmean(x))
    MoDataDF['R-B Index']=GroupD['Discharge'].apply(lambda x:CalcRBindex(x))
    return ( MoDataDF )

def GetAnnualAverages(WYDataDF):
    """This function calculates annual average values for all statistics and
    metrics.  The routine returns an array of mean values for each metric
    in the original dataframe."""
    
    # Give the values to columns
    AnnualAverages=WYDataDF.mean(axis=0)
    
    return( AnnualAverages )

def GetMonthlyAverages(MoDataDF):
    """This function calculates annual average monthly values for all 
    statistics and metrics.  The routine returns an array of mean values 
    for each metric in the original dataframe."""
    
    # Define the name of columns and create a new dataframe
    cols=['site_no','Mean Flow','Coeff Var','Tqmean','R-B Index']
    m=[3,4,5,6,7,8,9,10,11,0,1,2]
    index=0
    
    # Create a new dataframe
    MonthlyAverages=pd.DataFrame(0,index=range(1,13),columns=cols)
    
    # Create the output table
    for i in range(12):
        MonthlyAverages.iloc[index,0]=MoDataDF['site_no'][::12].mean()
        MonthlyAverages.iloc[index,1]=MoDataDF['Mean Flow'][m[index]::12].mean()
        MonthlyAverages.iloc[index,2]=MoDataDF['Coeff Var'][m[index]::12].mean()
        MonthlyAverages.iloc[index,3]=MoDataDF['Tqmean'][m[index]::12].mean()
        MonthlyAverages.iloc[index,4]=MoDataDF['R-B Index'][m[index]::12].mean()
        index+=1
    return( MonthlyAverages )

# the following condition checks whether we are running as a script, in which 
# case run the test code, otherwise functions are being imported so do not.
# put the main routines from your code after this conditional check.

if __name__ == '__main__':

    # define filenames as a dictionary
    # NOTE - you could include more than jsut the filename in a dictionary, 
    #  such as full name of the river or gaging site, units, etc. that would
    #  be used later in the program, like when plotting the data.
    fileName = { "Wildcat": "WildcatCreek_Discharge_03335000_19540601-20200315.txt",
                 "Tippe": "TippecanoeRiver_Discharge_03331500_19431001-20200315.txt" }
    
    # define blank dictionaries (these will use the same keys as fileName)
    DataDF = {}
    MissingValues = {}
    WYDataDF = {}
    MoDataDF = {}
    AnnualAverages = {}
    MonthlyAverages = {}
    
    # process input datasets
    for file in fileName.keys():
        
        print( "\n", "="*50, "\n  Working on {} \n".format(file), "="*50, "\n" )
        
        DataDF[file], MissingValues[file] = ReadData(fileName[file])
        print( "-"*50, "\n\nRaw data for {}...\n\n".format(file), DataDF[file].describe(), "\n\nMissing values: {}\n\n".format(MissingValues[file]))
        
        # clip to consistent period
        DataDF[file], MissingValues[file] = ClipData( DataDF[file], '1969-10-01', '2019-09-30' )
        print( "-"*50, "\n\nSelected period data for {}...\n\n".format(file), DataDF[file].describe(), "\n\nMissing values: {}\n\n".format(MissingValues[file]))
        
        # calculate descriptive statistics for each water year
        WYDataDF[file] = GetAnnualStatistics(DataDF[file])
        
        # calcualte the annual average for each stistic or metric
        AnnualAverages[file] = GetAnnualAverages(WYDataDF[file])
        
        print("-"*50, "\n\nSummary of water year metrics...\n\n", WYDataDF[file].describe(), "\n\nAnnual water year averages...\n\n", AnnualAverages[file])

        # calculate descriptive statistics for each month
        MoDataDF[file] = GetMonthlyStatistics(DataDF[file])

        # calculate the annual averages for each statistics on a monthly basis
        MonthlyAverages[file] = GetMonthlyAverages(MoDataDF[file])
        
        print("-"*50, "\n\nSummary of monthly metrics...\n\n", MoDataDF[file].describe(), "\n\nAnnual Monthly Averages...\n\n", MonthlyAverages[file])
    
    # Write data into annual metrics csv file
    Wc = WYDataDF['Wildcat']
    Wc['Station'] = 'Wildcat'
    Tip = WYDataDF['Tippe']
    Tip['Station'] = 'Tippe'
    Wc = Wc.append(Tip)
    Wc.to_csv('Annual_Metrics.csv',sep=',', index=True)
        
    # Write data into monthly metrics csv file
    Wc_m = MoDataDF['Wildcat']
    Wc_m['Station'] = 'Wildcat'
    Tip_m = MoDataDF['Tippe']
    Tip_m['Station'] = 'Tippe'
    Wc_m = Wc_m.append(Tip_m)
    Wc_m.to_csv('Monthly_Metrics.csv',sep=',', index=True)
    
    # Write data into average annual metrics text file
    Wc_avg = AnnualAverages['Wildcat']
    Wc_avg['Station'] = 'Wildcat'
    Tip_avg = AnnualAverages['Tippe']
    Tip_avg['Station'] = 'Tippe'
    Wc_avg = Wc_avg.append(Tip_avg)
    Wc_avg.to_csv('Average_Annual_Metrics.txt',sep='\t', index=True)
        
    # Write data into average monthly metrics text file
    Wc_avg_m = MonthlyAverages['Wildcat']
    Wc_avg_m['Station'] = 'Wildcat'
    Tip_avg_m = MonthlyAverages['Tippe']
    Tip_avg_m['Station'] = 'Tippe'
    Wc_avg_m = Wc_avg_m.append(Tip_avg_m)
    Wc_avg_m.to_csv('Average_Monthly_Metrics.txt',sep='\t', index=True)