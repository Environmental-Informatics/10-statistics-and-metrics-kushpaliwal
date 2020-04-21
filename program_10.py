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

    # remove the negative streamflow values
    DataDF = DataDF[~(DataDF['Discharge']<0)]
    
    # quantify the number of missing values
    MissingValues = DataDF["Discharge"].isna().sum()
    
    return( DataDF, MissingValues )

def ClipData( DataDF, startDate, endDate ):
    """This function clips the given time series dataframe to a given range 
    of dates. Function returns the clipped dataframe and and the number of 
    missing values."""

    # clips time series dataframe to given range of dates
    DataDF = DataDF[startDate:endDate]
    
    # quantify number of missing values
    MissingValues = DataDF["Discharge"].isna().sum()
    
    return( DataDF, MissingValues )

def CalcTqmean(Qvalues):
    """This function computes the Tqmean of a series of data, typically
       a 1 year time series of streamflow, after filtering out NoData
       values.  Tqmean is the fraction of time that daily streamflow
       exceeds mean streamflow for each year. Tqmean is based on the
       duration rather than the volume of streamflow. The routine returns
       the Tqmean value for the given data array."""
    
    # filter out the NoData values
    #Qvalues = Qvalues.dropna()

    # calculate mean streamflow
    mean_Q = Qvalues.mean()

    # find number daily streamflows exceeding mean value
    index=( Qvalues > mean_Q )

    # calculate Tqmean
    Tqmean = (index.sum()/len(Qvalues))
    
    return ( Tqmean )

def CalcRBindex(Qvalues):
    """This function computes the Richards-Baker Flashiness Index
       (R-B Index) of an array of values, typically a 1 year time
       series of streamflow, after filtering out the NoData values.
       The index is calculated by dividing the sum of the absolute
       values of day-to-day changes in daily discharge volumes
       (pathlength) by total discharge volumes for each year. The
       routine returns the RBindex value for the given data array."""

    # filter out the NoData values
    #Qvalues = Qvalues.dropna()

    # calculate sum of absolute values of daily changes in discharge
    sum_abs = 0.0    
    for i in range(len(Qvalues)-1):
        sum_abs += abs(Qvalues[i+1]-Qvalues[i])

    # R-B Index

    RBindex = (sum_abs/Qvalues.sum())
    
    return ( RBindex )

def Calc7Q(Qvalues):
    """This function computes the seven day low flow of an array of 
       values, typically a 1 year time series of streamflow, after 
       filtering out the NoData values. The index is calculated by 
       computing a 7-day moving average for the annual dataset, and 
       picking the lowest average flow in any 7-day period during
       that year.  The routine returns the 7Q (7-day low flow) value
       for the given data array."""
    
    # filter out the NoData values
    #Qvalues = Qvalues.dropna()

    # obtain lowest average flow in any 7-day period during the year
    val7Q = Qvalues.rolling(window=7).mean().min()
    
    return ( val7Q )

def CalcExceed3TimesMedian(Qvalues):
    """This function computes the number of days with flows greater 
       than 3 times the annual median flow. The index is calculated by 
       computing the median flow from the given dataset (or using the value
       provided) and then counting the number of days with flow greater than 
       3 times that value.   The routine returns the count of events greater 
       than 3 times the median annual flow value for the given data array."""

    # filter out the NoData values
    #Qvalues = Qvalues.dropna()

    # find number of daily streamflows greater than 3 times the median flow
    index=( Qvalues > 3*Qvalues.median() )

    # calculate number of days with flows greater than median flow
    median3x = index.sum()
    
    return ( median3x )

def GetAnnualStatistics(DataDF):
    """This function calculates annual descriptive statistcs and metrics for 
    the given streamflow time series.  Values are retuned as a dataframe of
    annual values for each water year.  Water year, as defined by the USGS,
    starts on October 1."""

    # define column names
    colNames = ['site_no','Mean Flow','Peak Flow','Median','Coeff Var','Skew','TQmean','R-B Index','7Q','3xMedian']

    # obtain water year range of the data
    start_Year = 1970
    end_Year = 2020
    WYrange = range(start_Year,end_Year)
    
    # define new dataframe to store annual (water year) metric values, set the index as the "water year"
    WYDataDF = pd.DataFrame(0.0, index=WYrange, columns=colNames)
    WYDataDF.index.name = 'Water Year'

    # calculate metrics for each water year   
    WYDataDF['site_no'] = DataDF['site_no'][0]
    for WY in WYrange:
        WYDataDF.loc[WY,'Mean Flow'] = DataDF['Discharge'][str(WY-1)+"-10-01":str(WY)+"-09-30"].mean()
        WYDataDF.loc[WY,'Peak Flow'] = DataDF['Discharge'][str(WY-1)+"-10-01":str(WY)+"-09-30"].max()
        WYDataDF.loc[WY,'Median'] = DataDF['Discharge'][str(WY-1)+"-10-01":str(WY)+"-09-30"].median()
        WYDataDF.loc[WY,'Coeff Var'] = DataDF['Discharge'][str(WY-1)+"-10-01":str(WY)+"-09-30"].std()/WYDataDF.loc[WY,'Mean Flow']*100
        WYDataDF.loc[WY,'Skew'] = stats.skew(DataDF['Discharge'][str(WY-1)+"-10-01":str(WY)+"-09-30"])
        WYDataDF.loc[WY,'TQmean'] = CalcTqmean(DataDF['Discharge'][str(WY-1)+"-10-01":str(WY)+"-09-30"])
        WYDataDF.loc[WY,'R-B Index'] = CalcRBindex(DataDF['Discharge'][str(WY-1)+"-10-01":str(WY)+"-09-30"])
        WYDataDF.loc[WY,'7Q'] = Calc7Q(DataDF['Discharge'][str(WY-1)+"-10-01":str(WY)+"-09-30"])
        WYDataDF.loc[WY,'3xMedian'] = CalcExceed3TimesMedian(DataDF['Discharge'][str(WY-1)+"-10-01":str(WY)+"-09-30"])        

    WYDataDF.index = DataDF.resample('AS-OCT').mean().index
         
    return ( WYDataDF )

def GetMonthlyStatistics(DataDF):
    """This function calculates monthly descriptive statistics and metrics 
    for the given streamflow time series.  Values are returned as a dataframe
    of monthly values for each year."""

    # define column names
    colNames = ['site_no','Mean Flow','Coeff Var','TQmean','R-B Index']

    # define monthly index
    index = DataDF.resample('MS').mean().index
    
    # define new dataframe to store monthly metric values
    MoDataDF = pd.DataFrame(0.0, index=index, columns=colNames)
    MoDataDF.index.name = 'Month'
  
    # calculate metrics for each month in each water year
    MoDataDF['site_no'] = DataDF['site_no'][0]
    MoDataDF['Mean Flow'] = DataDF['Discharge'].resample("MS").mean().values
    MoDataDF['Coeff Var'] = DataDF['Discharge'].resample("MS").std().values/MoDataDF['Mean Flow']*100
    MoDataDF['TQmean'] = DataDF['Discharge'].resample("MS").apply(CalcTqmean).values
    MoDataDF['R-B Index'] = DataDF['Discharge'].resample("MS").apply(CalcRBindex).values

    return ( MoDataDF )

def GetAnnualAverages(WYDataDF):
    """This function calculates annual average values for all statistics and
    metrics.  The routine returns an array of mean values for each metric
    in the original dataframe."""
    
    # calculate mean value for each metric
    AnnualAverages = WYDataDF.mean()

    return( AnnualAverages )

def GetMonthlyAverages(MoDataDF):
    """This function calculates annual average monthly values for all 
    statistics and metrics.  The routine returns an array of mean values 
    for each metric in the original dataframe."""

    # group by month and calculate mean value for each month
    Mo_index=lambda x:x.month
    MonthlyAverages = MoDataDF.groupby(Mo_index).mean()
    MonthlyAverages.index.name = 'Date'    
    
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

        # insert a column "Station" with the river name
        WYDataDF[file].insert(0,'Station',file)
        # output annual metrics tables to CSV files in the current directory
        WYDataDF[file].to_csv('Annual_Metrics.csv', mode='a')

        # insert a column "Station" with the river name
        MoDataDF[file].insert(0,'Station',file)
        # output monthly metrics tables to CSV files in the current directory
        MoDataDF[file].to_csv('Monthly_Metrics.csv', mode='a')

        # insert a column "Station" with the river name
        AA_df = pd.DataFrame(AnnualAverages[file])
        Station = pd.DataFrame(file,index=['Station'],columns=[0])
        AA_Df = Station.append(AA_df)
        # output annual averages to TAB delimited files in the current directory
        AA_Df.to_csv('Average_Annual_Metrics.txt', sep='\t', mode='a', header=False)

        # insert a column "Station" with the river name
        MonthlyAverages[file].insert(0,'Station',file)
        # output monthly averages to TAB delimited files in the current directory
        MonthlyAverages[file].to_csv('Average_Monthly_Metrics.txt', sep='\t', mode='a', header=True)