#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Dr. W SUN on 22/07/2018

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

import matplotlib.style  # https://matplotlib.org/users/dflt_style_changes.html

mpl.style.use('classic')

import statsmodels.api as sm  # import statsmodels
from arch import arch_model

os.chdir(os.path.dirname(__file__))  # switch to the folder where you script is stored

##########################################################################
# read four factors of fama french data
##########################################################################

file_3_Factors = 'Europe_3_Factors_Daily.csv'
df_threefators = pd.read_csv(file_3_Factors, parse_dates=[0], index_col=0, skiprows=6).drop('RF', axis=1)['2013':'2017']

file_MOM_Factor = 'Europe_MOM_Factor_Daily.csv'
df_forthfactor = pd.read_csv(file_MOM_Factor, parse_dates=[0], index_col=0, skiprows=6)['2013':'2017']

factors = pd.concat([df_threefators, df_forthfactor], axis=1)
print(factors.head())
print(factors.describe())

##########################################################################
# read green bond daily price
##########################################################################

file = 'SRI Final Price.xlsx'
xl = pd.ExcelFile(file)
print(xl.sheet_names)

stats_list = []
ols_list = []
garch_list = []

for select_sheet in xl.sheet_names:
    df = xl.parse(select_sheet, parse_dates=[0], index_col=0, pase_dates=True, skiprows=[0, 1, 2, 4], header=0)
    print('Import sheet: {}'.format(select_sheet))

    ##########################################################################
    # calculate daily average returns and describe stats
    ##########################################################################
    returns = df.pct_change().mean(axis=1)['2013':'2017']
    print(returns.describe())

    returns.plot()
    plt.savefig('{}_daily_returns.png'.format(select_sheet))
    plt.close()

    stats_current = returns.describe()
    stats_current.name = select_sheet
    stats_list.append(stats_current)

    ##########################################################################
    # linear regression of fama french factors
    ##########################################################################
    X = factors
    y = returns
    X = sm.add_constant(X)
    model_static = sm.OLS(y, X).fit()
    print(model_static.params)
    ols_current = model_static.params
    ols_current.name = select_sheet
    ols_list.append(ols_current)

    ##########################################################################
    # arch analysis of volatility
    ##########################################################################
    am = arch_model(returns)
    res = am.fit()
    print(res.summary())

    garch_current = res.params
    garch_current.name = select_sheet
    garch_list.append(garch_current)

    with open('garch_summary_{}.csv'.format(select_sheet), 'w') as f:
        f.write(res.summary().as_csv())

    res.plot(annualize='D')
    plt.savefig('garch_{}.png'.format(select_sheet))
    plt.close()

##########################################################################
# write all results
##########################################################################
pd.concat(stats_list, axis=1).to_csv('greenbond_stats.csv')
pd.concat(ols_list, axis=1).to_csv('greenbond_ols.csv')
pd.concat(garch_list, axis=1).to_csv('greenbond_garch.csv')

if __name__ == '__main__':
    pass
