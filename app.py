from flask import Flask, request, render_template

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as st
import math as mt
from scipy import stats


def meanDf(df, ax=0):
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


def get_Results(p_value, alpha=0.05):
    if(p_value < alpha):
        return ("We reject the null hypothesis")
    else:
        return("We do not reject the null hypothesis")


def oneSampleTest(sample_mean, var):
    dataset = database["dataset"]
    t_stat = (meanDf(dataset[var]) - sample_mean) / stdErrDf(dataset[var])
    dof = len(dataset[var]) - 1
    # pr(dof)
    alpha = 0.05

    critical_val = stats.t.ppf(1 - alpha, dof)
    p_val = (1 - stats.t.cdf(abs(t_stat), dof))
    # pr("T-Statistic: {}, P-Value: {}".format(t_stat, p_val))
    return [get_Results(p_val, alpha), p_val, t_stat]


def independentTTest(dep_var, indep_var):
    dataset = database["dataset"]
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


def pairedTTest(dep_var, indep_var):
    dataset = database["dataset"]
    # dep_var = dataset.columns[getInt(
    #     "Enter the before test variable: ", dataset.columns)]
    # indep_var = dataset.columns[getInt(
    #     "Enter the after test variable: ", dataset.columns)]

    t_value, p_value = stats.ttest_rel(
        list(dataset[dep_var]), list(dataset[indep_var]))
    one_tailed_p_value = float("{:.6f}".format(p_value/2))
    # print('Test statistic: %f'%float("{:.6f}".format(t_value)))
    # print('p-value: %f'%one_tailed_p_value)
    alpha = 0.05
    # pr("T-Statistic: {}, P-Value: {}".format(t_value, one_tailed_p_value))
    return [get_Results(one_tailed_p_value, alpha), one_tailed_p_value, t_value]


def fTest(dep_var, indep_var):
    dataset = database["dataset"]
    # dep_var = dataset.columns[getInt(
    #     "Enter the dependent variable: ", dataset.columns)]
    # indep_var = dataset.columns[getInt(
    #     "Enter the independent variable: ", dataset.columns)]

    types_dep_var = list(set(dataset[dep_var]))

    dep_df_1 = dataset.query("{} == '{}'".format(dep_var, types_dep_var[0]))
    dep_int_arr_1 = (dep_df_1[indep_var])
    # print((dep_int_arr_1))

    dep_df_2 = dataset.query("{} == '{}'".format(dep_var, types_dep_var[1]))
    dep_int_arr_2 = (dep_df_2[indep_var])
    # print((dep_int_arr_2))

    f_calc = varDf(dep_int_arr_1) / varDf(dep_int_arr_2)
    dof1, dof2 = len(dep_int_arr_1) - 1, len(dep_int_arr_2) - 1
    p_value = 1 - stats.f.cdf(f_calc, dof1, dof2)
    # print(f_calc, p_value)
    alpha = 0.05
    # pr("F-Statistic: {}, P-Value: {}".format(f_calc, p_value))
    return [get_Results(p_value, alpha), p_value, f_calc]


def chiSquareTest(row_var, col_var):
    dataset = database["dataset"]
    # row_var = dataset.columns[getInt("Enter row category: ", dataset.columns)]
    # col_var = dataset.columns[getInt(
    #     "Enter column category: ", dataset.columns)]

    types_row = list(set(dataset[row_var]))
    types_col = list(set(dataset[col_var]))
    data_comb = []

    for i in types_row:
        for j in types_col:
            data_comb.append([i, j])

    # pr(types_row, types_col)
    # pr(data_comb)

    row_total = []
    col_total = []
    for i in types_row:
        length = len(dataset.query("{} == '{}'".format(row_var, i)))
        row_total.append(length)
    for i in types_col:
        length = len(dataset.query("{} == '{}'".format(col_var, i)))
        col_total.append(length)

    row_col_comb = []
    for i in row_total:
        for j in col_total:
            row_col_comb.append([i, j])

    data_total = sum(row_total)
    # pr(row_total, col_total, data_total)
    # pr(row_col_comb)

    observed_O = []
    for lst in data_comb:
        length = len(dataset.query("{} == '{}' and {} == '{}'".format(
            row_var, lst[0], col_var, lst[1])))
        observed_O.append(length)
    # observed_O

    expected_e = []
    for i in row_col_comb:
        expected_e.append((i[0]*i[1]) / data_total)
    # expected_e

    chi_sq_calc = 0
    for i in range(len(observed_O)):
        chi_sq_calc = chi_sq_calc + \
            ((observed_O[i] - expected_e[i]) ** 2) / expected_e[i]
    # chi_sq_calc

    # chi_sq_tab = stats.chi2.ppf(1-.05, df=dof)

    dof = (len(types_row) - 1) * (len(types_col) - 1)

    p_value = stats.distributions.chi2.sf(chi_sq_calc, dof)
    alpha = 0.05
    # pr("Chi_Sq-Statistic: {}, P-Value: {}".format(chi_sq_calc, p_value))
    return [get_Results(p_value, alpha), p_value, chi_sq_calc]


def ANOVATest(indep_var, dep_var):
    dataset = database["dataset"]
    # indep_var = dataset.columns[getInt(
    #     "Enter the categorical variable: ", dataset.columns)]
    # dep_var = dataset.columns[getInt(
    #     "Enter the numeric variable: ", dataset.columns)]

    types_list = list(set(dataset[indep_var]))
    # types_list

    df_list = []
    for i in types_list:
        df = dataset.query("{} == '{}'".format(indep_var, i))
        df_list.append(df[dep_var].reset_index())
    # pr(list(df_list[1][dep_var].values))
    # pr(len(df_list))
    total_sample = len(df_list)
    # for i in df_list:
    #   if(len(df_list[0]) != len(i)):
    #     print('Unequal array size')
    #     break

    N = len(dataset[dep_var])
    n = len(df_list[0])
    # pr(N, n)
    # Calculate total(T) ie total  of sum of each sample
    Total = sum(dataset[dep_var])
    # pr(Total)

    # correlation factor = T^2/number of data values
    corr_fac = Total ** 2 / len(dataset[dep_var])
    # pr(corr_fac)

    # total sum of squares - Correlation factor
    sum_of = 0
    for i in dataset[dep_var]:
        sum_of = sum_of + i ** 2
    total_sum_of_sq = sum_of - corr_fac
    # pr(total_sum_of_sq)

    # Sums of sq between sample
    ssq_between_sample = 0
    for df in df_list:
        ssq_between_sample = ssq_between_sample + (sum(df[dep_var]) ** 2)/n
    ssq_between_sample = ssq_between_sample - corr_fac
    # pr(ssq_between_sample)

    # Sums of sq within sample
    ssq_within_sample = total_sum_of_sq - ssq_between_sample
    # pr(ssq_within_sample)

    # F Value
    var_between_sample = ssq_between_sample / (len(df_list)-1)
    var_within_sample = ssq_within_sample / (N-len(df_list))
    f_value_calc = var_between_sample / var_within_sample
    # pr(N-len(df_list), n-1)
    p_value = 1 - stats.f.cdf(f_value_calc, len(df_list)-1, N-len(df_list))
    # pr(f_value_calc, p_value)
    alpha = 0.05
    # pr("P-Value: {}".format(p_value))
    return [get_Results(p_value, alpha), p_value, f_value_calc]


app = Flask(__name__)

# test_number = -1
# dataset = ""
database = {
    "test_number": -1,
    "dataset": ""
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get_variables", methods=['GET', 'POST'])
def getVariables():
    test_no = request.form['test_type']
    test_number = int(test_no)
    database["test_number"] = test_number
    # print(database["test_number"])
    dataset_name = request.form['dataset_file']
    dataset = pd.read_csv(dataset_name)
    database["dataset"] = dataset
    data_cols = dataset.columns

    if(test_number == 0):
        return render_template("variable.html", content={"category1": "Enter Sample Mean: ", "category2": "Enter the test variable number: ", "data_cols": enumerate(data_cols), "test_name": "One Sample T-Test"})
    elif(test_number == 1):
        return render_template("variable.html", content={"category1": "Enter The Dependent Variable: ", "category2": "Enter The Independent Variable: ", "data_cols": enumerate(data_cols), "test_name": "Independent T-Test"})
    elif(test_number == 2):
        return render_template("variable.html", content={"category1": "Enter The Before Test Variable: ", "category2": "Enter The After Test Variable:", "data_cols": enumerate(data_cols), "test_name": "Paired T-Test"})
    elif(test_number == 3):
        return render_template("variable.html", content={"category1": "Enter The Dependent Variable: ", "category2": "Enter The Independent Variable: ", "data_cols": enumerate(data_cols), "test_name": "F-Test"})
    elif(test_number == 4):
        return render_template("variable.html", content={"category1": "Enter The Row Category: ", "category2": "Enter The Column Category: ", "data_cols": enumerate(data_cols), "test_name": "Chi Square Test"})
    else:
        return render_template("variable.html", content={"category1": "Enter The Categorical Variable: ", "category2": "Enter The Numeric Variable: ", "data_cols": enumerate(data_cols), "test_name": "ANOVA"})


@app.route("/get_results", methods=['GET', 'POST'])
def getResults():
    cat1 = request.form['category1']
    cat2 = request.form['category2']
    result = ""
    # return str(database["test_number"])

    if database["test_number"] == 0:
        result = oneSampleTest(int(cat1), cat2)
        return render_template("result.html", content={"test_name": "One Sample T Test", "p_val": result[1], "test_stat": result[2], "hypo": result[0], "test_stat_name": "T-Statistic"})

    elif database["test_number"] == 1:
        result = independentTTest(cat1, cat2)
        return render_template("result.html", content={"test_name": "Independent T Test", "p_val": result[1], "test_stat": result[2], "hypo": result[0], "test_stat_name": "T-Statistic"})

    elif database["test_number"] == 2:
        result = pairedTTest(cat1, cat2)
        return render_template("result.html", content={"test_name": "Paired T Test", "p_val": result[1], "test_stat": result[2], "hypo": result[0], "test_stat_name": "T-Statistic"})

    elif database["test_number"] == 3:
        result = fTest(cat1, cat2)
        return render_template("result.html", content={"test_name": "F Test", "p_val": result[1], "test_stat": result[2], "hypo": result[0], "test_stat_name": "F-Statistic"})

    elif database["test_number"] == 4:
        result = chiSquareTest(cat1, cat2)
        return render_template("result.html", content={"test_name": "Chi Square Test", "p_val": result[1], "test_stat": result[2], "hypo": result[0], "test_stat_name": "Chi-Statistic"})

    else:
        result = ANOVATest(cat1, cat2)
        return render_template("result.html", content={"test_name": "ANOVA", "p_val": result[1], "test_stat": result[2], "hypo": result[0], "test_stat_name": "F-Statistic"})


if __name__ == ("__main__"):
    app.run(debug=True)
