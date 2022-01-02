from flask import Flask, request, render_template

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as st
import math as mt
from scipy import stats

def meanDf(df, ax = 0):
  return df.mean(axis=ax, skipna=True)

def stdDevDf(df):
  return st.stdev(df)

def stdErrDf(df):
  return stats.sem(df)

def varDf(df):
  return st.variance(df)
  
# def pr(*obj):
#   print(*obj)

def normCurve(df, step=0.01):
#   pr(min(df), max(df))
  x_axis = np.arange(min(df), max(df), step)
    
  # Calculating mean and standard deviation
  mean = meanDf(x_axis)
  sd = stdDevDf(x_axis)
    
  plt.plot(x_axis, stats.norm.pdf(x_axis, mean, sd))
  plt.show()

def getInt(stat, lst):
#   pr(stat)
#   for i, x in enumerate(lst):
    # pr("{}) {}".format(i+1, x))
  num = int(input("Enter: "))
  return int(num - 1)

def getDataSet():
  df_name = input("Enter Dataset Name: ")
  df = pd.read_csv(df_name)
  return df

def get_Results(p_value, alpha = 0.05):  
  if(p_value < alpha):
    return ("We reject the null hypothesis")
  else:
    return("We do not reject the null hypothesis")


def oneSampleTest(sample_mean, var):
    t_stat = (meanDf(dataset[var]) - sample_mean) / stdErrDf(dataset[var])
    dof = len(dataset[var]) - 1
    # pr(dof)
    alpha = 0.05

    critical_val = stats.t.ppf(1 - alpha, dof)
    p_val = (1 - stats.t.cdf(abs(t_stat), dof))
    # pr("T-Statistic: {}, P-Value: {}".format(t_stat, p_val))
    return [get_Results(p_val, alpha), p_val, t_stat]

def independentTTest(dep_var, indep_var):
    types_dep_var = list(set(dataset[dep_var]))
    dep_df_1 = dataset.query("{} == '{}'".format(dep_var, types_dep_var[0]))
    dep_int_arr_1 = (dep_df_1[indep_var])
    # print((dep_int_arr_1))

    dep_df_2 = dataset.query("{} == '{}'".format(dep_var, types_dep_var[1]))
    dep_int_arr_2 = (dep_df_2[indep_var])
    # print((dep_int_arr_2))

    mean1, mean2 = meanDf(dep_int_arr_1, 0), meanDf(dep_int_arr_2, 0)
    sdErr1, sdErr2 = stdErrDf(dep_int_arr_1), stdErrDf(dep_int_arr_2)

    sdErr = mt.sqrt(sdErr1**2.0 + sdErr2**2.0)
    t_stat = (mean1 - mean2) / sdErr

    dof = len(dep_int_arr_1) + len(dep_int_arr_2) - 2
    alpha = 0.05

    critical_val = stats.t.ppf(1 - alpha, dof)
    p_val = (1 - stats.t.cdf(abs(t_stat), dof)) * 2
    
    # pr("T-Statistic: {}, P-Value: {}".format(t_stat, p_val))
    return [get_Results(p_val, alpha), p_val, t_stat]

app = Flask(__name__)

# test_number = -1
# dataset = ""
database = { 
    "test_number" : -1, 
    "dataset": ""
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_variables", methods=['GET', 'POST'])
def getVariables():    
    test_no = request.form['test_type']    
    test_number = int(test_no)
    database.test_number = test_number
    dataset_name = request.form['dataset_file']
    dataset = pd.read_csv(dataset_name)

    data_cols = dataset.columns

    if(test_number == 0):
        return render_template("variable.html", content={"category1":"Enter Sample Mean: ", "category2":"Enter the test variable number: ", "data_cols":enumerate(data_cols)})
    elif(test_number == 1):
        return render_template("variable.html", content={"category1":"Enter The Dependent Variable: ", "category2":"Enter The Independent Variable: ", "data_cols":enumerate(data_cols)})
    elif(test_number == 2):
        return render_template("variable.html", content={"category1":"Enter The Before Test Variable: ", "category2":"Enter The After Test Variable:", "data_cols":enumerate(data_cols)})
    elif(test_number == 3):
        return render_template("variable.html", content={"category1":"Enter The Dependent Variable: ", "category2":"Enter The Independent Variable: ", "data_cols":enumerate(data_cols)})
    elif(test_number == 4):
        return render_template("variable.html", content={"category1":"Enter The Row Category: ", "category2":"Enter The Column Category: ", "data_cols":enumerate(data_cols)})
    else:
        return render_template("variable.html", content={"category1":"Enter The Categorical Variable: ", "category2":"Enter The Numeric Variable: ", "data_cols":enumerate(data_cols)})


@app.route("/get_results", methods=['GET', 'POST'])
def getResults():
    cat1 = request.form['category1']
    cat2 = request.form['category2']
    result = ""
    return database.test_number

    # if test_number == 0:            
    #     result = oneSampleTest(int(cat1), cat2)        
    #     return render_template("result.html", content={"test_name":"One Sample T Test", "p_val": result[1], "test_stat": result[2], "hypo": result[0], "test_stat_name": "T-Statistic"})

    # elif test_number == 1:
    #     result = oneSampleTest(cat1, cat2)        
    #     return render_template("result.html", content={"test_name":"Independent T Test", "p_val": result[1], "test_stat": result[2], "hypo": result[0], "test_stat_name": "T-Statistic"})

    # elif test_number == 2:
    #     return cat1+" "+cat2
    # elif test_number == 3:
    #     return cat1+" "+cat2
    # elif test_number == 4:
    #     return cat1+" "+cat2
    # else:
    #     return cat1+" "+cat2
    


if __name__ == ("__main__"):
    app.run(debug=True)