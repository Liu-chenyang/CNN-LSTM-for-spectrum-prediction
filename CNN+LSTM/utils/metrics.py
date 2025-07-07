import torch



def accuracy(pred, y):
    """
        :param pred: predictions
        :param y: ground truth
        :return: accuracy, defined as 1 - (norm(y - pred) / norm(y))
        """
    return 1 - torch.linalg.norm(y - pred, "fro") / torch.linalg.norm(y, "fro")

# 这个 r2 函数用于计算 决定系数，也称为 R 平方（R²），它是回归模型中常用的一种衡量模型拟合效果的指标。
# R² 值表示模型预测值与实际值之间的差异，取值范围为 0 到 1，值越接近 1 表示模型的拟合效果越好。
def r2(pred, y):
    """
    :param y: ground truth
    :param pred: predictions
    :return: R square (coefficient of determination)
    """
    return 1 - torch.sum((y - pred) ** 2) / torch.sum((y - torch.mean(pred)) ** 2)

# explained_variance() 函数用于计算 解释方差，这是回归模型评估中常用的一个指标。
# 解释方差 衡量的是模型预测结果中可以通过真实值解释的方差比例。它的取值范围通常在 0 到 1 之间，值越接近 1 表示模型的解释能力越强。
def explained_variance(pred, y):
    return 1 - torch.var(y - pred) / torch.var(y)

# MAPE 函数用于计算 平均绝对百分比误差（Mean Absolute Percentage Error, MAPE），这是回归问题中常用的误差度量指标。它通过计算预测值与真实值之间的相对误差（按百分比），然后取其平均值。
def MAPE(v, v_):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAPE averages on all elements of input.
    '''

    return torch.mean(torch.abs((v_-v)/(v+1)))

# MAE（Mean Absolute Error，平均绝对误差）是回归问题中常用的一种损失度量方式。它衡量预测值与真实值之间的误差的绝对值，并取平均值。MAE 反映了模型预测结果与真实值之间的平均偏差。
def MAE(v, v_):
    '''
    Mean absolute error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAE averages on all elements of input.
    '''
    return torch.mean(torch.abs((v_-v)))


# RMSE（Root Mean Squared Error，均方根误差）是回归问题中常用的一种误差度量方式。它衡量预测值与真实值之间误差的平方平均值，并对结果开平方，使其具有与原始数据相同的单位。
# RMSE 可以有效地反映预测值与真实值之间的平均偏差，同时对较大的误差有更大的惩罚作用（因为平方增大了误差的影响）。
def RMSE(v, v_):
    '''
    Mean squared error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, RMSE averages on all elements of input.
    '''
    return torch.sqrt(torch.mean((v_ - v) ** 2))