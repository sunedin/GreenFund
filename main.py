#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Dr. W SUN on 22/07/2018

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import numpy as np

import matplotlib.style  # https://matplotlib.org/users/dflt_style_changes.html

mpl.style.use('classic')

import statsmodels.api as sm  # import statsmodels
from arch import arch_model


def main(fund_price_file=None, fund_region='EU', returns_type='pct', tag=''):
    os.chdir(os.path.dirname(__file__))  # switch to the folder where you script is stored
    output_folder = '{}_{}_{}_return'.format(tag, fund_region, returns_type)
    output_dir = os.path.join(os.path.dirname(__file__), output_folder)

    ##########################################################################
    # read four factors of fama french data
    ##########################################################################

    if fund_region == 'EU':
        file_3_Factors = 'Europe_3_Factors_Daily.csv'
        file_MOM_Factor = 'Europe_MOM_Factor_Daily.csv'
        df_threefators = pd.read_csv(file_3_Factors, parse_dates=[0], index_col=0, skiprows=6).drop('RF', axis=1)['2013':'2018']
        df_forthfactor = pd.read_csv(file_MOM_Factor, parse_dates=[0], index_col=0, skiprows=6)['2013':'2018']
        ff_rf = pd.read_csv(file_3_Factors, parse_dates=[0], index_col=0, skiprows=6)['RF']['2013':'2018']

    if fund_region == 'US':
        file_3_Factors = 'F-F_Research_Data_Factors_daily.CSV'
        file_MOM_Factor = 'F-F_Momentum_Factor_daily.csv'
        df_threefators = pd.read_csv(file_3_Factors, parse_dates=[0], index_col=0, skiprows=4).drop('RF', axis=1)['2013':'2018']
        df_forthfactor = pd.read_csv(file_MOM_Factor, parse_dates=[0], index_col=0, skiprows=13)['2013':'2018']
        ff_rf = pd.read_csv(file_3_Factors, parse_dates=[0], index_col=0, skiprows=4)['RF']['2013':'2018']

    if fund_region == 'Global':
        file_3_Factors = 'Global_3_Factors_daily.CSV'
        file_MOM_Factor = 'Global_MOM_Factor_daily.csv'
        df_threefators = pd.read_csv(file_3_Factors, parse_dates=[0], index_col=0, skiprows=6).drop('RF', axis=1)['2013':'2018']
        df_forthfactor = pd.read_csv(file_MOM_Factor, parse_dates=[0], index_col=0, skiprows=6)['2013':'2018']
        ff_rf = pd.read_csv(file_3_Factors, parse_dates=[0], index_col=0, skiprows=6)['RF']['2013':'2018']

    factors = pd.concat([df_threefators, df_forthfactor], axis=1)
    factors.index = pd.to_datetime(factors.index)
    factors = factors/100
    ff_rf = ff_rf/100
    print(factors.head())
    print(factors.describe())

    ##########################################################################
    # read green fund daily price
    ##########################################################################

    file = fund_price_file
    xl = pd.ExcelFile(file)
    print(xl.sheet_names)

    stats_list = []
    ols_list = []
    pvalues_list = []
    garch_list = []
    arx_list = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    os.chdir(output_dir)

    for select_sheet in xl.sheet_names:
        df = xl.parse(select_sheet, parse_dates=[0], index_col=0, pase_dates=True, skiprows=[0, 1, 2, 4], header=0)
        df.index = pd.to_datetime(df.index)
        print('Import sheet: {}'.format(select_sheet))

        # skip/filter Nan colomns
        print('the following columns are not numeric ')
        print(df.select_dtypes(exclude=['float64']))
        df = df.select_dtypes(include=['float64'])

        ##########################################################################
        # calculate daily average returns and describe stats ; https://stackoverflow.com/questions/35365545/calculating-cumulative-returns-with-pandas-dataframe
        ##########################################################################
        if returns_type == 'pct':  # simple return
            returns = df.pct_change(limit=2).mean(axis=1)['2013':'2018']
        if returns_type == 'cum':  # cumulative_return
            returns = df.pct_change(limit=2)['2013':'2018']
            returns = ((1 + returns).cumprod() - 1).mean(axis=1)
        if returns_type == 'log':  # log return
            returns = np.log(1 + df.pct_change(limit=2)).mean(axis=1)['2013':'2018']
        print(returns.describe())

        # check data completeness
        print('The following date have NaN return value')
        print(returns[returns.isna().any()])
        returns.fillna(method='bfill', inplace=True)

        returns.plot()
        plt.savefig('{}_daily_returns.png'.format(select_sheet))
        plt.close()

        stats_current = returns.describe()
        stats_current.name = select_sheet
        stats_list.append(stats_current)

        ##########################################################################
        # linear regression of fama french factors
        ##########################################################################
        slice_index_ols = returns.index.intersection(factors.index)

        X = factors.loc[slice_index_ols]
        y = returns.loc[slice_index_ols] - ff_rf[slice_index_ols]
        X_with_constant = sm.add_constant(X)
        model_static = sm.OLS(y, X_with_constant, missing='drop').fit()

        print(model_static.params)
        ols_current = model_static.params
        ols_current.name = select_sheet
        ols_list.append(ols_current)

        pvalues_current = model_static.pvalues
        pvalues_current.name = select_sheet
        pvalues_list.append(pvalues_current)

        with open('ols_summary_{}.csv'.format(select_sheet), 'w') as f:
            f.write(model_static.summary().as_csv())

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
        # arx analysis of volatility
        ##########################################################################
        from arch.univariate import ARX
        arx = ARX(returns, lags=[1])
        res = arx.fit()

        print(res.summary())

        arx_current = res.params
        arx_current.name = select_sheet
        arx_list.append(arx_current)

        with open('arx_summary_{}.csv'.format(select_sheet), 'w') as f:
            f.write(res.summary().as_csv())

        res.plot(annualize='D')
        plt.savefig('arx_{}.png'.format(select_sheet))
        plt.close()

    ##########################################################################
    # write all results
    ##########################################################################
    pd.concat(stats_list, axis=1).to_csv('greenfund_stats.csv')
    pd.concat(ols_list, axis=1).to_csv('greenfund_ols.csv')
    pd.concat(pvalues_list, axis=1).to_csv('greenfund_pvalues.csv')
    pd.concat(garch_list, axis=1).to_csv('greenfund_garch.csv')
    pd.concat(arx_list, axis=1).to_csv('greenfund_arx.csv')


if __name__ == '__main__':
    # main(fund_price_file='SRI EU Final Price.xlsx', fund_region='EU', returns_type='cum', tag='SRI')
    # main(fund_price_file='SRI EU Final Price.xlsx', fund_region='EU', returns_type='log', tag='SRI')
    # main(fund_price_file='SRI US Final Price.xlsx', fund_region='US', returns_type='cum', tag='SRI')
    # main(fund_price_file='SRI US Final Price.xlsx', fund_region='US', returns_type='log', tag='SRI')
    # main(fund_price_file='Pair EU Final Price.xlsx', fund_region='EU', returns_type='cum', tag='PAIR')
    # main(fund_price_file='Pair EU Final Price.xlsx', fund_region='EU', returns_type='log', tag='PAIR')
    # main(fund_price_file='Pair US Final Price.xlsx', fund_region='US', returns_type='cum', tag='PAIR')
    # main(fund_price_file='Pair US Final Price.xlsx', fund_region='US', returns_type='log', tag='PAIR')
    # main(fund_price_file='US All Price.xlsx', fund_region='US', returns_type='log', tag='US')
    # main(fund_price_file='EU All Price.xlsx', fund_region='EU', returns_type='log', tag='EU')
    # main(fund_price_file='All Sample.xlsx', fund_region='Global', returns_type='log', tag='All')
    # main(fund_price_file='US All Price.xlsx', fund_region='US', returns_type='cum', tag='US')
    # main(fund_price_file='EU All Price.xlsx', fund_region='EU', returns_type='cum', tag='EU')
    # main(fund_price_file='All Sample.xlsx', fund_region='Global', returns_type='cum', tag='All')

    main(fund_price_file='New_SRI EU Final Price.xlsx', fund_region='EU', returns_type='pct', tag='New_SRI')
    # main(fund_price_file='New_SRI EU Final Price.xlsx', fund_region='EU', returns_type='cum', tag='New_SRI')
    # main(fund_price_file='New_SRI EU Final Price.xlsx', fund_region='EU', returns_type='log', tag='New_SRI')
    main(fund_price_file='New_SRI US Final Price.xlsx', fund_region='US', returns_type='pct', tag='New_SRI')
    # main(fund_price_file='New_SRI US Final Price.xlsx', fund_region='US', returns_type='cum', tag='New_SRI')
    # main(fund_price_file='New_SRI US Final Price.xlsx', fund_region='US', returns_type='log', tag='New_SRI')
    main(fund_price_file='New_Pair EU Final Price.xlsx', fund_region='EU', returns_type='pct', tag='New_PAIR')
    # main(fund_price_file='New_Pair EU Final Price.xlsx', fund_region='EU', returns_type='cum', tag='New_PAIR')
    # main(fund_price_file='New_Pair EU Final Price.xlsx', fund_region='EU', returns_type='log', tag='New_PAIR')
    main(fund_price_file='New_Pair US Final Price.xlsx', fund_region='US', returns_type='pct', tag='New_PAIR')
    # main(fund_price_file='New_Pair US Final Price.xlsx', fund_region='US', returns_type='cum', tag='New_PAIR')
    # main(fund_price_file='New_Pair US Final Price.xlsx', fund_region='US', returns_type='log', tag='New_PAIR')
    # main(fund_price_file='New_US All Price.xlsx', fund_region='US', returns_type='log', tag='New_US')
    # main(fund_price_file='New_EU All Price.xlsx', fund_region='EU', returns_type='log', tag='New_EU')
    # main(fund_price_file='New_All Sample.xlsx', fund_region='Global', returns_type='log', tag='New_All')
    main(fund_price_file='New_US All Price.xlsx', fund_region='US', returns_type='pct', tag='New_US')
    # main(fund_price_file='New_US All Price.xlsx', fund_region='US', returns_type='cum', tag='New_US')
    main(fund_price_file='New_EU All Price.xlsx', fund_region='EU', returns_type='pct', tag='New_EU')
    # main(fund_price_file='New_EU All Price.xlsx', fund_region='EU', returns_type='cum', tag='New_EU')
    main(fund_price_file='New_All Sample.xlsx', fund_region='Global', returns_type='pct', tag='New_All')
    # main(fund_price_file='New_All Sample.xlsx', fund_region='Global', returns_type='cum', tag='New_All')

