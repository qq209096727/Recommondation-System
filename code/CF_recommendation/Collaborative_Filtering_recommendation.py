#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Feiniu


import numpy as np
import pandas as pd
from measure import measure_method


def user_similarity(user_rating, user):
    """
    计算user-user之间的相似度
    :param user_rating: user评分矩阵
    :param user: user编号
    :return: 该user与其余users的相似度
    """
    user_feature = user_rating.loc[user, :]
    user_index = user_rating.index.values
    # item_index = user_rating.index.values

    user_rating_feature = np.matrix(user_feature.values)
    user_rating_matrix = np.matrix(user_rating)

    #使用余弦夹角作为度量标准计算相似度
    similarity_uu = np.array(measure_method.cos_measure(user_rating_feature, user_rating_matrix))[0]
    similarity_uu_df = pd.Series(similarity_uu, index=user_index).drop(user)

    return similarity_uu_df


def top_k_similar(rated_sim, user_rate, K):
    """
    得到top k 的对应的评分以及相似度
    :param rated_sim: 相似度向量
    :param K: 相似度最大的K个
    :return: top k 的对应的评分以及相似度
    """
    #TOP K user的对应引索和评分
    top_k_user = rated_sim.iloc[: K]
    top_index = top_k_user.index
    top_rating = user_rate.loc[top_index].values
    top_sim = top_k_user.values

    return top_rating, top_sim


def estimate_rate(rating, similarity, naive=False):
    """
    计算得到评分的估计值
    :param rating: 相似的评分向量
    :param similarity: 相似度向量
    :param naive: 是否采用直接平均的方法
    :return: 评分的估计值
    """
    if naive == False:
        rate_hat = rating.dot(similarity) / similarity.sum()
    else:
        rate_hat = rating.mean()

    return rate_hat


def CF_recommend_estimate(user_rating, user, item_lst, K):

    user_similarity_lst = user_similarity(user_rating, user)

    rate_hat_lst = []
    for i in item_lst:
        # 得到该user未评分item但其他已评分的user
        user_unrate_item = user_rating.loc[:, str(i)]
        other_user_rate = user_unrate_item[user_unrate_item > 0]

        # 对该item已评分uesr的引索和对应评分
        other_user_rate_index = other_user_rate.index

        # 得到待评分item下已评分uesr与该user的相似度,并且以降序排序
        rated_user_sim = user_similarity_lst.loc[other_user_rate_index].sort_values(ascending=False)

        if rated_user_sim.shape[0] < K:

            if rated_user_sim.shape[0] == 0:
                rate_hat = 0

            else:
                top_rating, top_sim = top_k_similar(rated_user_sim, other_user_rate, K)
                rate_hat = estimate_rate(top_rating, top_sim)

        else:
            top_rating, top_sim = top_k_similar(rated_user_sim, other_user_rate, K)
            rate_hat = estimate_rate(top_rating, top_sim)

        rate_hat_lst.append(rate_hat)

    return rate_hat_lst


if __name__ == '__main__':

    movies_feature = pd.read_csv('../data/Moivelens/ml-latest-small/movies_feature.csv', index_col=0)
    user_rating = pd.read_csv('../data/Moivelens/ml-latest-small/user-rating.csv', index_col=0)

    print CF_recommend_estimate(user_rating, 1, [10, 17], 50)

