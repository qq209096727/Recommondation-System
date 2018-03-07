#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Feiniu

import numpy as np
import pandas as pd


def cos_measure(feature_vector, feature_matrix):
    """
    计算item之间的余弦夹角相似度
    :param feature_vector: 待测量的item特征向量
    :param feature_matrix: 用户已评分的items的特征矩阵
    :return: 待计算item与用户已评分的items的余弦夹角相识度的向量
    """
    x_c = (feature_vector * feature_matrix.T) + 0.0000001
    mod_x = np.sqrt(feature_vector * feature_vector.T)
    mod_c = np.sqrt((feature_matrix * feature_matrix.T).diagonal())
    cos_xc = x_c / (mod_x * mod_c)

    return cos_xc


def comp_mse(pred, actual):
    """
    计算根均方误差
    :param pred: 预测值
    :param actual: 真实值
    :return: 根均方误差
    """
    error = pred - actual
    count = error.nonzero()[0].shape
    MSE = (error).dot((error).T) / count

    return MSE

def comp_rmse(pred, actual):
    """
    计算根均方误差
    :param pred: 预测值
    :param actual: 真实值
    :return: 根均方误差
    """
    error = pred - actual
    count = error.nonzero()[0].shape
    RMSE = np.sqrt((error).dot((error).T) / count)

    return RMSE