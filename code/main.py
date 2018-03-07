#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Feiniu

import numpy as np
import pandas as pd
from data_preprocessing import ML_data_preproccessing
from CB_recommendation import Content_based_recommendation
from CF_recommendation import Collaborative_Filtering_recommendation
from MF_recommendation import Matrix_Factorization
from measure import measure_method


if __name__ == '__main__':

    # ML_data_preproccessing.data_preprocess()

    # 导入数据
    movies_feature = pd.read_csv('../data/Moivelens/ml-latest-small/movies_feature.csv', index_col=0)
    user_rating = pd.read_csv('../data/Moivelens/ml-latest-small/user-rating.csv', index_col=0)

    train, test = ML_data_preproccessing.train_test_split(user_rating)


    # 使用基于内容的推荐算法来估计评分
    # count = 0
    # total = float(train.shape[0])
    # for idx, user in train.iterrows():
    #     unrated_index = user[user == 0].index.values
    #     rates_lst = []
    #
    #     for item in unrated_index:
    #         rate_h = Content_based_recommendation.CB_recommend_estimate(user, movies_feature, int(item))
    #         rates_lst.append(rate_h)
    #
    #     train.loc[idx, unrated_index] = rates_lst
    #
    #     count += 1
    #     if count % 100 == 0:
    #         presentage = round((count / total) * 100)
    #         print 'Completed %d' % presentage + '%'
    #
    # train.to_csv('../data/Moivelens/ml-latest-small/pred_ratings_CB.csv')


    # 使用user-user的协同过滤算法来估计评分
    # count = 0
    # total = float(train.shape[0])
    # for idx, user in train.iterrows():
    #     unrated_index = user[user == 0].index.values
    #     unrated_index_ = map(int, unrated_index)
    #     rates_lst = Collaborative_Filtering_recommendation.CF_recommend_estimate(train, idx, unrated_index_, 50)
    #
    #     train.loc[idx, unrated_index] = rates_lst
    #
    #     count += 1
    #     if count % 100 == 0:
    #         presentage = round((count / total) * 100)
    #         print 'Completed %d' % presentage + '%'
    #
    # train.to_csv('../data/Moivelens/ml-latest-small/pred_ratings_CF.csv')


    # 计算基于内容的推荐和协同过滤的MSE和RMSE
    # pred_CB = pd.read_csv('../data/Moivelens/ml-latest-small/pred_ratings_CB.csv', index_col=0)
    # pred_CF = pd.read_csv('../data/Moivelens/ml-latest-small/pred_ratings_CF.csv', index_col=0)
    #
    # nonzero_index = user_rating.values.nonzero()
    #
    # actual = user_rating.values[nonzero_index[0], nonzero_index[1]]
    # pred_CB = pred_CB.values[nonzero_index[0], nonzero_index[1]]
    # pred_CF = pred_CF.values[nonzero_index[0], nonzero_index[1]]
    #
    # print 'MSE of CB is %s' % measure_method.comp_mse(pred_CB, actual)
    # print 'RMSE of CB is %s' % measure_method.comp_rmse(pred_CB, actual)
    # print 'MSE of CF is %s' % measure_method.comp_mse(pred_CF, actual)
    # print 'RMSE of CF is %s' % measure_method.comp_rmse(pred_CF, actual)


    # 使用矩阵分解算法来估计评分
    MF_estimate = Matrix_Factorization.Matrix_Factorization(K=10, epoch=5)
    MF_estimate.fit(train)
    R_hat = MF_estimate.start()
    non_index = test.values.nonzero()
    pred_MF = R_hat[non_index[0], non_index[1]]
    actual = test.values[non_index[0], non_index[1]]

    print 'MSE of MF is %s' % measure_method.comp_mse(pred_MF, actual)
    print 'RMSE of MF is %s' % measure_method.comp_rmse(pred_MF, actual)
