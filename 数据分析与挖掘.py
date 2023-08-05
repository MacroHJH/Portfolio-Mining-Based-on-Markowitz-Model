# -*- coding: utf-8 -*-
"""
Created on Wed May  3 21:36:07 2023

@author: 26931
"""

import pandas as pd
import numpy as np
import sqlalchemy as sql
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from datetime import datetime, date
import matplotlib.dates as mdates
import dataframe_image as dfi
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score



conn = sql.create_engine('mysql+pymysql://root:Tis180012..@localhost:3306/CSI_300',encoding = 'utf-8')

def get_table_image(df, address , cols = -1, rows = -1):
    df.index.name = None
    dfi.export(df, address, dpi = 600, max_cols = cols, max_rows = rows)
    return

csi300_stocks_yield = pd.read_sql_table('沪深300成分股收益率',con = conn)
csi300_stocks_yield.set_index(csi300_stocks_yield.columns[0], inplace = True, drop = True)

def reset_mulindex(index):
    level = [[],[]]
    for i in index:
        i = i.replace('(','').replace(')','').replace('\'','')
        j,k = i.split(',')
        level[0].append(j)
        level[1].append(k)
    mul_index = pd.MultiIndex.from_arrays(level, names = ['行业','股票代码'])
    return mul_index

mul_index = reset_mulindex(csi300_stocks_yield.columns)
csi300_stocks_yield = csi300_stocks_yield.T.set_index(mul_index).T

yield_corr = csi300_stocks_yield.corr()

im_fig = plt.figure(figsize = (5,4), dpi = 1000)
im_ax = im_fig.add_subplot(1,1,1)
im = im_ax.imshow(yield_corr, cmap = plt.cm.BuPu)
plt.colorbar(im)
plt.title('Heat Map of Stocks Correlation Coefficient', x = 0.55, y = 1.05)

def t_test(corr_df):
    same = []
    diff = []
    industry_set = set([i[0] for i in corr_df.columns])
    
    for i in industry_set:
        same_ls = []
        diff_ls = []
        same_df = corr_df[i].loc[i]
        same_columns = set(same_df.columns)
        diff_columns = set(corr_df.columns) - same_columns
        diff_df = corr_df[list[diff_columns]].loc[i]
        
        for j in range(len(corr_df.loc[i])):
            same_ls.append(same_df.iloc[j].mean())
            diff_ls.append(diff_df.iloc[j].mean())
        
        same.append(np.array(same_ls).mean())
        diff.append(np.array(diff_ls).mean())
    
    print()
    print(stats.levene(same, diff),"\n")
    print(stats.ttest_ind(same,diff),"\n")
    print('行业内股票相关系数均值：', np.array(same).mean())
    print('行业外股票相关系数均值：', np.array(diff).mean())

    return

t_test(yield_corr)

yield_train = pd.read_sql_table('训练集', con = conn)
yield_train.set_index(yield_train.columns[0], inplace = True)
yield_train = yield_train.T.set_index(mul_index).T

def divide_industry(df):
    industry_set = sorted({i[0] for i in df.columns})
    df_ls = [df[j] for j in industry_set]
    return df_ls

industry_yield = divide_industry(yield_train)

IT_mean_std = industry_yield[0].describe().loc[['mean','std']].T
med_mean_std = industry_yield[1].describe().loc[['mean','std']].T
mat_mean_std = industry_yield[2].describe().loc[['mean','std']].T
fin_mean_std = industry_yield[3].describe().loc[['mean','std']].T
bas_mean_std = industry_yield[4].describe().loc[['mean','std']].T
ind_mean_std = industry_yield[5].describe().loc[['mean','std']].T
con_mean_std= industry_yield[6].describe().loc[['mean','std']].T

scalar = StandardScaler()
for i in [con_mean_std, ind_mean_std, mat_mean_std, bas_mean_std, IT_mean_std, med_mean_std, fin_mean_std]:
    i[['scaled_mean', 'scaled_std']] = scalar.fit_transform(i)

get_table_image(IT_mean_std.head(), address = 'C:/Users/26931/Desktop/学习资料/BDA数据分析师/IT.jpg', cols = 6)

def get_cluster_tree(df):
    df1 = df.copy()
    dist_mat = linkage(df1[['scaled_mean', 'scaled_std']],
                       method='ward', metric='euclidean')
    return dist_mat

tree_fig,tree_ax = plt.subplots(figsize = (6,3), dpi = 1000)
dendrogram(get_cluster_tree(ind_mean_std),
           ax=tree_ax,
           color_threshold = 5)
plt.title('Cluster Tree', fontdict = {'fontsize':10}, pad = 10)

def get_cluster(df, linkage = 'ward'):
    df1 = df.copy()
    cluster = AgglomerativeClustering(n_clusters=3, linkage=linkage)
    labels = cluster.fit_predict(df1[['scaled_mean', 'scaled_std']])
    df1['labels'] = labels
    
    label_mean = df1.groupby(by = 'labels', as_index = False).mean()[['scaled_mean', 'scaled_std', 'labels']]
    label_mean.sort_values(by = 'scaled_std', inplace = True)
    df1['labels'] = df1['labels'].replace({label_mean['labels'].iloc[0]:'低风险低收益', label_mean['labels'].iloc[1]:'中风险中收益',          label_mean['labels'].iloc[2]:'高风险高收益'})
    label_mean['labels'] = label_mean['labels'].replace({label_mean['labels'].iloc[0]:'低风险低收益', label_mean['labels'].iloc[1]:'中风险中收益',          label_mean['labels'].iloc[2]:'高风险高收益'})

    label_mean = label_mean.rename(columns = {'scaled_mean':'mean_centroid', 'scaled_std':'std_centroid'})
    index = df1.index
    df1 = pd.merge(df1, label_mean, on = 'labels', how = 'left')
    df1 = df1.set_index(index)
    df1.insert(0,'labels',df1.pop('labels'))
     
    df1['distance'] = (df1['scaled_mean'] - df1['mean_centroid'])**2 +(df1['scaled_std'] - df1['std_centroid'])**2
    df1['rank'] = df1.groupby(by = 'labels')['distance'].rank(method = 'first')
    df1.sort_values(by = ['labels', 'rank'], inplace = True)
    return df1
        
IT_clustered = get_cluster(IT_mean_std)
med_clustered = get_cluster(med_mean_std)
mat_clustered = get_cluster(mat_mean_std)
fin_clustered = get_cluster(fin_mean_std)
bas_clustered = get_cluster(bas_mean_std)
ind_clustered = get_cluster(ind_mean_std)
con_clustered = get_cluster(con_mean_std)

get_table_image(IT_clustered.head(), address = 'C:/Users/26931/Desktop/学习资料/BDA数据分析师/IT_clustered.jpg')

outlier = fin_clustered[fin_clustered['labels'] == '高风险高收益'].copy()
fin_mean_std_new = fin_mean_std.drop(index = outlier.index)
fin_clustered_new = get_cluster(fin_mean_std_new)
outlier['mean_centroid'] = fin_clustered_new[fin_clustered_new['labels'] == '高风险高收益']['mean_centroid'].iloc[1]
outlier['std_centroid'] = fin_clustered_new[fin_clustered_new['labels'] == '高风险高收益']['std_centroid'].iloc[1]
outlier['distance'] = (outlier['scaled_mean'] - outlier['mean_centroid'])**2 +(outlier['scaled_std'] - outlier['std_centroid'])**2
fin_clustered_new = pd.concat([fin_clustered_new, outlier])
fin_clustered_new['rank'] = fin_clustered_new.groupby(by = 'labels')['distance'].rank()

dot_fig,axes = plt.subplots(2,4, sharex = True, sharey = True, figsize = (10,6), dpi = 1000)
plt.subplots_adjust(wspace = 0, hspace = 0, top = 0.95)
plt.suptitle('The Cluster Results of Different Industries')
colors = {'低风险低收益':'palegreen', '中风险中收益':'pink', '高风险高收益':'violet'}
c_colors = {'低风险低收益':'darkgreen', '中风险中收益':'deeppink', '高风险高收益':'darkviolet'}
axes[0][0].set_ylabel('Returns')
axes[1][3].set_xlabel('Risk')
axes[0][0].scatter(IT_clustered['scaled_std'],
                   IT_clustered['scaled_mean'],
                   c = [colors[label] for label in IT_clustered['labels']])
axes[0][0].scatter(IT_clustered['std_centroid'],
                   IT_clustered['mean_centroid'],
                   c = [c_colors[label] for label in IT_clustered['labels']])
axes[0][1].scatter(med_clustered['scaled_std'],
                   med_clustered['scaled_mean'],
                   c = [colors[label] for label in med_clustered['labels']])
axes[0][1].scatter(med_clustered['std_centroid'],
                   med_clustered['mean_centroid'],
                   c = [c_colors[label] for label in med_clustered['labels']])
axes[0][2].scatter(mat_clustered['scaled_std'],
                   mat_clustered['scaled_mean'],
                   c = [colors[label] for label in mat_clustered['labels']])
axes[0][2].scatter(mat_clustered['std_centroid'],
                   mat_clustered['mean_centroid'],
                   c = [c_colors[label] for label in mat_clustered['labels']])
axes[0][3].scatter(fin_clustered_new['scaled_std'],
                   fin_clustered_new['scaled_mean'],
                   c = [colors[label] for label in fin_clustered_new['labels']])
axes[0][3].scatter(fin_clustered_new['std_centroid'],
                   fin_clustered_new['mean_centroid'],
                   c = [c_colors[label] for label in fin_clustered_new['labels']])
axes[1][0].scatter(bas_clustered['scaled_std'],
                   bas_clustered['scaled_mean'],
                   c = [colors[label] for label in bas_clustered['labels']])
axes[1][0].scatter(bas_clustered['std_centroid'],
                   bas_clustered['mean_centroid'],
                   c = [c_colors[label] for label in bas_clustered['labels']])
axes[1][1].scatter(ind_clustered['scaled_std'],
                   ind_clustered['scaled_mean'],
                   c = [colors[label] for label in ind_clustered['labels']])
axes[1][1].scatter(ind_clustered['std_centroid'],
                   ind_clustered['mean_centroid'],
                   c = [c_colors[label] for label in ind_clustered['labels']])
axes[1][2].scatter(con_clustered['scaled_std'],
                   con_clustered['scaled_mean'],
                   c = [colors[label] for label in con_clustered['labels']])
axes[1][2].scatter(con_clustered['std_centroid'],
                   con_clustered['mean_centroid'],
                   c = [c_colors[label] for label in con_clustered['labels']])

def get_target_stocks(risk, n = 1):
    df = pd.DataFrame()
    IT_stock = IT_clustered[(IT_clustered['labels'] == risk) & (IT_clustered['rank'].isin([i+1 for i in list(range(n))]))].index[0]
    med_stock = med_clustered[(med_clustered['labels'] == risk) & (med_clustered['rank'].isin([i+1 for i in list(range(n))]))].index[0]
    mat_stock = mat_clustered[(mat_clustered['labels'] == risk) & (mat_clustered['rank'].isin([i+1 for i in list(range(n))]))].index[0]
    fin_stock = fin_clustered_new[(fin_clustered_new['labels'] == risk) & (fin_clustered_new['rank'].isin([i+1 for i in list(range(n))]))].index[0]
    bas_stock = bas_clustered[(bas_clustered['labels'] == risk) & (bas_clustered['rank'].isin([i+1 for i in list(range(n))]))].index[0]
    ind_stock = ind_clustered[(ind_clustered['labels'] == risk) & (ind_clustered['rank'].isin([i+1 for i in list(range(n))]))].index[0]
    con_stock = con_clustered[(con_clustered['labels'] == risk) & (con_clustered['rank'].isin([i+1 for i in list(range(n))]))].index[0]
    target_stocks_ls = [IT_stock,med_stock,mat_stock,fin_stock,bas_stock,ind_stock,con_stock]
    for i in yield_train.columns:
        if i[1] in target_stocks_ls:
            df[i[1]] = yield_train[i]
        else:
            continue
    return df
#不知道为什么运行索引前面有个空格，但是无伤大雅

low_target_stocks = get_target_stocks(risk = '低风险低收益')
med_target_stocks = get_target_stocks(risk = '中风险中收益')
hig_target_stocks = get_target_stocks(risk = '高风险高收益')

get_table_image(low_target_stocks.head(), address = 'C:/Users/26931/Desktop/学习资料/BDA数据分析师/low_target_stocks.jpg')


def MonteCarlo(target_stocks):

    annual_returns = target_stocks.mean()*242
    annual_volatilitys = target_stocks.cov()*242

    weights = []
    expected_returns = []
    expected_volatilitys = []
    n = 0
    while n < 10000:
        weight = np.random.random(7)
        weight /= np.sum(weight)
        expected_return = np.sum(annual_returns*weight)
        expected_volitity = np.sqrt(np.dot(weight.T, np.dot(annual_volatilitys, weight)))
        weights.append(weight)
        expected_returns.append(expected_return)
        expected_volatilitys.append(expected_volitity)
        n += 1

    expected_returns = np.array(expected_returns)
    expected_volatilitys = np.array(expected_volatilitys)
    weights = np.array(weights)
    
    risk_free = 0.03
    sharpe = (expected_returns - risk_free)/expected_volatilitys
    
    max_sharpe_weight = weights[sharpe.argmax()]
    min_volatility_weight = weights[expected_volatilitys.argmin()]
    
    return expected_returns, expected_volatilitys, sharpe, max_sharpe_weight, min_volatility_weight
    
lexp_returns, lexp_volatilitys, lsharpe, lmax_sharpe, lmin_volatility = MonteCarlo(low_target_stocks)
mexp_returns, mexp_volatilitys, msharpe, mmax_sharpe, mmin_volatility = MonteCarlo(med_target_stocks)
hexp_returns, hexp_volatilitys, hsharpe, hmax_sharpe, hmin_volatility = MonteCarlo(hig_target_stocks)

def Makowits(expected_returns, expected_volatilitys, sharpe, label):
    fig_bullet, ax_bullet = plt.subplots(figsize = (8,5), dpi = 1000)
    bullet = ax_bullet.scatter(expected_volatilitys, expected_returns, s = 0.8, c = sharpe, cmap = 'summer')
    ax_bullet.scatter(expected_volatilitys[sharpe.argmax()],
                      expected_returns[sharpe.argmax()],
                      c = 'crimson')
    ax_bullet.scatter(expected_volatilitys[expected_volatilitys.argmin()],
                  expected_returns[expected_volatilitys.argmin()],
                  c = 'forestgreen')
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(mappable = bullet, label = 'Sharp Ratio')
    plt.title(label)
    return

Makowits(lexp_returns, lexp_volatilitys, lsharpe, label = 'Low Risk Low Return')
Makowits(mexp_returns, mexp_volatilitys, msharpe, label = 'Medium Risk Medium Return')
Makowits(hexp_returns, hexp_volatilitys, hsharpe, label = 'High Risk High Return')




csi300_index = pd.read_sql_table('沪深300指数走势', con = conn, index_col = 'date')
csi300_index['date'] = [datetime.strftime(i, '%Y-%m-%d') for i in csi300_index.index]
csi300_index.set_index('date', inplace = True, drop = True)
csi300_index = csi300_index.loc['2021-12-31':'2022-12-31']
csi300_index['yield'] = csi300_index['close'].pct_change()
csi300_index = csi300_index.dropna(axis = 0)
csi300_index['cumprod'] = np.cumprod(1 + csi300_index['yield']) - 1

yield_test = pd.read_sql_table('测试集', con = conn)
yield_test.set_index(yield_test.columns[0], inplace = True, drop = True)
yield_test = yield_test.T.set_index(mul_index).T

def get_portfolio_test(stocks, weight):
    df = pd.DataFrame()
    for i in yield_test.columns:
        if i[1] in stocks:
            df[i[1]] = yield_test[i]
        else:
            continue
    df['portfolio'] = np.dot(df, weight)
    df['cumprod'] = np.cumprod(1 + df['portfolio']) - 1
    return df

l_minportfolio_test = get_portfolio_test(low_target_stocks, lmin_volatility)
l_maxportfolio_test = get_portfolio_test(low_target_stocks, lmax_sharpe)
m_minportfolio_test = get_portfolio_test(med_target_stocks, mmin_volatility)
m_maxportfolio_test = get_portfolio_test(med_target_stocks, mmax_sharpe)
h_minportfolio_test = get_portfolio_test(hig_target_stocks,hmin_volatility)
h_maxportfolio_test = get_portfolio_test(hig_target_stocks,hmax_sharpe)

get_table_image(l_minportfolio_test.head(), address = 'C:/Users/26931/Desktop/学习资料/BDA数据分析师/l_minportfolio_test.jpg')

csi300_index_σ = round(csi300_index['yield'].std(), 4)
l_minportfolio_σ = round(l_minportfolio_test['portfolio'].std(), 4)
l_maxportfolio_σ = round(l_maxportfolio_test['portfolio'].std(), 4)
m_minportfolio_σ = round(m_minportfolio_test['portfolio'].std(), 4)
m_maxportfolio_σ = round(m_maxportfolio_test['portfolio'].std(), 4)
h_minportfolio_σ = round(h_minportfolio_test['portfolio'].std(), 4)
h_maxportfolio_σ = round(h_maxportfolio_test['portfolio'].std(), 4)

def get_cumyield_plot(portfolio, market, label):
    portfolio.index = pd.to_datetime(portfolio.index)
    market.index = pd.to_datetime(market.index)

    fig_cum, ax_cum = plt.subplots(dpi=1000, figsize=(10, 6))
    ax_cum.plot(portfolio['cumprod'],
                c='dodgerblue',
                label='Portfolio accumulated yield')
    ax_cum.plot(market['cumprod'],
                c='crimson',
                label='CSI300 accumulated yield')
    plt.legend()
    plt.title(label, fontdict = {'fontsize': 16})
    plt.grid(visible=True)
    ax_cum.axhline(portfolio['cumprod'].iloc[-1],
                   c='grey',
                   linestyle='dashed')
    ax_cum.axhline(market['cumprod'].iloc[-1],
                   c='grey',
                   linestyle='dashed')

    plt.show()

    return

get_cumyield_plot(l_minportfolio_test, csi300_index, 'Low Risk Low Return Min Volatility Portfolio')
get_cumyield_plot(l_maxportfolio_test, csi300_index, 'Low Risk Low Return Max Sharpe Portfolio')
get_cumyield_plot(m_minportfolio_test, csi300_index, 'Med Risk Med Return Min Volatility Portfolio')
get_cumyield_plot(m_maxportfolio_test, csi300_index, 'Med Risk Med Return Max Sharpe Portfolio')
get_cumyield_plot(h_minportfolio_test, csi300_index, 'High Risk Hign Return Min Volatility Portfolio')
get_cumyield_plot(h_maxportfolio_test, csi300_index, 'Hign Risk Hign Return Max sharpe Portfolio')

fig_allcum,axes = plt.subplots(2,3, sharex = True, sharey = True, figsize = (10,6), dpi = 1000)
plt.subplots_adjust(wspace = 0, hspace = 0, top = 0.95)
plt.suptitle('The Accumulated Yield of Different Industries', fontsize = 14)

axes[0][0].plot(l_minportfolio_test['cumprod'],
                c='dodgerblue',
                label='Portfolio accumulated yield')
axes[0][0].plot(csi300_index['cumprod'],
            c='crimson',
            label='CSI300 accumulated yield')
axes[0][0].axhline(l_minportfolio_test['cumprod'].iloc[-1],
               c='grey',
               linestyle='dashed')
axes[0][0].axhline(csi300_index['cumprod'].iloc[-1],
               c='grey',
               linestyle='dashed')
axes[0][0].legend()
axes[0][0].grid(visible=True)

axes[1][0].plot(l_maxportfolio_test['cumprod'],
                c='dodgerblue',
                label='Portfolio accumulated yield')
axes[1][0].plot(csi300_index['cumprod'],
            c='crimson',
            label='CSI300 accumulated yield')
axes[1][0].axhline(l_maxportfolio_test['cumprod'].iloc[-1],
               c='grey',
               linestyle='dashed')
axes[1][0].axhline(csi300_index['cumprod'].iloc[-1],
               c='grey',
               linestyle='dashed')
axes[1][0].grid(visible=True)
axes[1][0].tick_params(axis = 'x', rotation = 45)

axes[0][1].plot(m_minportfolio_test['cumprod'],
                c='dodgerblue',
                label='Portfolio accumulated yield')
axes[0][1].plot(csi300_index['cumprod'],
            c='crimson',
            label='CSI300 accumulated yield')
axes[0][1].axhline(m_minportfolio_test['cumprod'].iloc[-1],
               c='grey',
               linestyle='dashed')
axes[0][1].axhline(csi300_index['cumprod'].iloc[-1],
               c='grey',
               linestyle='dashed')
axes[0][1].grid(visible=True)

axes[1][1].plot(m_maxportfolio_test['cumprod'],
                c='dodgerblue',
                label='Portfolio accumulated yield')
axes[1][1].plot(csi300_index['cumprod'],
            c='crimson',
            label='CSI300 accumulated yield')
axes[1][1].axhline(m_maxportfolio_test['cumprod'].iloc[-1],
               c='grey',
               linestyle='dashed')
axes[1][1].axhline(csi300_index['cumprod'].iloc[-1],
               c='grey',
               linestyle='dashed')
axes[1][1].grid(visible=True)
axes[1][1].tick_params(axis = 'x', rotation = 45)

axes[0][2].plot(h_minportfolio_test['cumprod'],
                c='dodgerblue',
                label='Portfolio accumulated yield')
axes[0][2].plot(csi300_index['cumprod'],
            c='crimson',
            label='CSI300 accumulated yield')
axes[0][2].axhline(h_minportfolio_test['cumprod'].iloc[-1],
               c='grey',
               linestyle='dashed')
axes[0][2].axhline(csi300_index['cumprod'].iloc[-1],
               c='grey',
               linestyle='dashed')
axes[0][2].grid(visible=True)

axes[1][2].plot(h_maxportfolio_test['cumprod'],
                c='dodgerblue',
                label='Portfolio accumulated yield')
axes[1][2].plot(csi300_index['cumprod'],
            c='crimson',
            label='CSI300 accumulated yield')
axes[1][2].axhline(h_maxportfolio_test['cumprod'].iloc[-1],
               c='grey',
               linestyle='dashed')
axes[1][2].axhline(csi300_index['cumprod'].iloc[-1],
               c='grey',
               linestyle='dashed')
axes[1][2].grid(visible=True)
axes[1][2].tick_params(axis = 'x', rotation = 45)



def get_beta(x,y):
    x = x.values.reshape(-1, 1)
    y = y.values.reshape(-1)
    
    model = LinearRegression()
    model.fit(x,y)
    y_pred = model.predict(x)
    coefficients = model.coef_[0]
    
    fig,ax = plt.subplots(dpi = 1000, figsize = (10,8))
    ax.scatter(x, y, c = 'royalblue')
    ax.plot(x, y_pred, linewidth = 3, c = 'purple')
    plt.title('Low Risk Low Return Low volatility',
              fontdict = {'fontsize':18})
    plt.xlabel('Index Yield', fontdict = {'fontsize':14})
    plt.ylabel('Portfolio Yield', fontdict = {'fontsize':14})
    
    return coefficients

l_minportfolio_beta = get_beta(csi300_index['yield'],l_minportfolio_test['portfolio'])
l_maxportfolio_beta = get_beta(csi300_index['yield'],l_maxportfolio_test['portfolio'])
m_minportfolio_beta = get_beta(csi300_index['yield'],m_minportfolio_test['portfolio'])
m_maxportfolio_beta = get_beta(csi300_index['yield'],m_maxportfolio_test['portfolio'])
h_minportfolio_beta = get_beta(csi300_index['yield'],h_minportfolio_test['portfolio'])
h_maxportfolio_beta = get_beta(csi300_index['yield'],h_maxportfolio_test['portfolio'])

portfolio_beta= pd.DataFrame({'l_minportfolio':[l_minportfolio_beta],
                                 'l_maxportfolio':[l_maxportfolio_beta],
                                 'm_minportfolio':[m_minportfolio_beta],
                                 'm_maxportfolio':[m_maxportfolio_beta],
                                 'h_minportfolio':[h_minportfolio_beta],
                                 'h_maxportfolio':[h_maxportfolio_beta]},
                             index = ['beta'])
get_table_image(portfolio_beta, address = 'C:/Users/26931/Desktop/学习资料/BDA数据分析师/beta.jpg')

portfolio_σ = pd.DataFrame({'l_minportfolio':[l_minportfolio_σ],
                                 'l_maxportfolio':[l_maxportfolio_σ],
                                 'm_minportfolio':[m_minportfolio_σ],
                                 'm_maxportfolio':[m_maxportfolio_σ],
                                 'h_minportfolio':[h_minportfolio_σ],
                                 'h_maxportfolio':[h_maxportfolio_σ]},
                             index = ['σ'])
get_table_image(portfolio_σ, address = 'C:/Users/26931/Desktop/学习资料/BDA数据分析师/σ.jpg')