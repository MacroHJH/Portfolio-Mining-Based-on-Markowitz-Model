# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 23:32:58 2023

@author: 26931
"""
from collections import Counter
import pandas as pd
import numpy as np
import sqlalchemy as sql
import akshare as ak
from datetime import datetime,date
import dataframe_image as dfi
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
#设置matplotlib的字体，使之能正常显示中文

conn = sql.create_engine('mysql+pymysql://root:Tis180012..@localhost:3306/CSI_300',encoding = 'utf-8')
#创建数据库连接

def get_table_image(df, address , cols = -1, rows = -1):
    df.index.name = None
    dfi.export(df, address, dpi = 600, max_cols = cols, max_rows = rows)
    return
#这个函数用于生成DataFrame的图片

index_price = ak.stock_zh_index_daily(symbol="sh000300")
index_price = index_price.set_index('date')

index_price.to_sql(name = '沪深300指数走势', con = conn, if_exists = 'replace', index = True)
#获取沪深300指数的历史行情并保存至数据库

get_table_image(index_price.head(), address = 'C:/Users/26931/Desktop/学习资料/BDA数据分析师/index_price.jpg')

index_stocks = ak.index_stock_cons_csindex(symbol="000300")

index_stocks.to_sql(name = '沪深300成分股', con = conn, if_exists = 'replace', index = True)
#获取沪深300指数的成分股并保存进数据库

get_table_image(index_stocks.head(), address = 'C:/Users/26931/Desktop/学习资料/BDA数据分析师/index_stocks.jpg')

def get_stock_hist(stocks):
    for i in stocks:
        i_hist_price = ak.stock_zh_a_hist(symbol = i, start_date = '20200101', end_date = '20221231', adjust = 'hfq')
        names = '{}_{}'.format('stock_',i)
        i_hist_price.to_sql(name = names, con = conn,if_exists = 'replace', index = False)
    return

get_stock_hist(stocks = index_stocks['成分券代码'])
#获取沪深300成分股的历史行情并保存至数据库

def get_stocks_price_mat(stocks, start_date, end_date):
    df = pd.DataFrame()
    for i in stocks:
        stock = pd.read_sql_table('{}_{}'.format('stock_',i), conn, index_col = '日期')
        df[i] = stock['收盘'].loc[start_date:end_date]
        df = df.dropna(axis = 1)
    return df

csi300_stocks_price = get_stocks_price_mat(stocks = index_stocks['成分券代码'], start_date = '2020-01-02', end_date = '2022-12-30')
#从指数的成分股中挑选出上市日期超过三年的股票并且把它们的收盘价放进一个DataFrame

get_table_image(csi300_stocks_price.head(), address = 'C:/Users/26931/Desktop/学习资料/BDA数据分析师/stocks_price.jpg', cols = 6)

csi300_stocks_yield = csi300_stocks_price.pct_change()
csi300_stocks_yield = csi300_stocks_yield.dropna()
#根据收盘价计算出成分股每天的收益率

get_table_image(round(csi300_stocks_yield.head(), 4), address = 'C:/Users/26931/Desktop/学习资料/BDA数据分析师/stocks_yield.jpg', cols = 6)

industries = pd.read_excel("C:\\Users\\26931\\Downloads\\行业分类.xlsx")
industries = industries[['证券代码','中证一级行业分类简称']]
industries.to_sql(name = '股票所属行业', con = conn, if_exists = 'replace',  index = False)
#获取股票所属行业信息

get_table_image(industries.head(), address = 'C:/Users/26931/Desktop/学习资料/BDA数据分析师/industries.jpg')

def stock_industry_dict(stocks):
    industry_dict = {}
    industry = pd.read_sql_table('股票所属行业', con = conn).set_index('证券代码')
    for i in stocks:
        industry_dict[i] = industry['中证一级行业分类简称'].loc[i]
    return industry_dict

industry_dict1 = stock_industry_dict(stocks = csi300_stocks_yield.columns)
#建立股票代码与对应的所属行业的字典

industry_num1 = dict(Counter(industry_dict1.values()))
#获得行业出现的频数

plt_num1, ax_num1 = plt.subplots(dpi = 1000)
ax_num1.bar(industry_num1.keys(),
            industry_num1.values(),
            color = 'dodgerblue')
plt.xticks(rotation = 90)
plt.title('调整前行业股票直方图')
plt.ylabel('股票数量')
#可视化结果

def industry_comb(si_dict):
    for i in si_dict.keys():
        if si_dict[i] in ['可选消费','主要消费']:
            si_dict[i] = '消费'
        elif si_dict[i] in ['房地产','金融']:
            si_dict[i] = '地产金融'
        elif si_dict[i] in ['公用事业','能源','通信服务']:
            si_dict[i] = '基础服务'
        else:
            continue
    return si_dict

industry_dict2 = industry_comb(industry_dict1)
industry_num2 = Counter(industry_dict2.values())
#对行业进行合并

plt_num2, ax_num2 = plt.subplots(dpi = 1000)
ax_num2.bar(industry_num2.keys(),
            industry_num2.values(),
            color = 'purple',
            width = 0.5)
plt.xticks(rotation = 90)
plt.title('调整后行业股票直方图')
plt.ylabel('股票数量')
#可视化合并的结果

industry_ls = [tuple(reversed(i)) for i in industry_dict2.items()]
columns = pd.MultiIndex.from_tuples(industry_ls,names = ['行业','股票代码'])
csi300_stocks_yield = csi300_stocks_yield.T.set_index(columns).T
csi300_stocks_yield = csi300_stocks_yield.sort_index(axis = 1)
csi300_stocks_price = csi300_stocks_price.T.set_index(columns).T
csi300_stocks_price = csi300_stocks_price.sort_index(axis = 1)

get_table_image(csi300_stocks_yield.head(), address = 'C:/Users/26931/Desktop/学习资料/BDA数据分析师/stocks_yield_new.jpg', cols = 6)
#获得股票对应的行业,并且把行业标签打到数据上

yield_train = csi300_stocks_yield.loc['2020-01-02':'2021-12-31']
yield_test = csi300_stocks_yield.loc['2022-01-04':'2022-12-30']

#直接保存的话报错，原因是索引的格式是字符串，所以重置一下索引
csi300_stocks_price.reset_index(drop = False, inplace = True)
csi300_stocks_price.to_sql(name = '沪深300成分股收盘价', con = conn, if_exists = 'replace', index = False)
#把股票的收盘价表导入数据库，方便以后使用

csi300_stocks_yield.reset_index(drop = False, inplace = True)
csi300_stocks_yield.to_sql(name = '沪深300成分股收益率', con = conn, if_exists = 'replace', index = False)
#保存到数据库方便后续使用

yield_train.reset_index(drop = False, inplace = True)
yield_train.to_sql(name = '训练集', con = conn, if_exists = 'replace', index = False)

yield_test.reset_index(drop = False, inplace = True)
yield_test.to_sql(name = '测试集', con = conn, if_exists = 'replace', index = False)
#把数据分为训练集和测试集