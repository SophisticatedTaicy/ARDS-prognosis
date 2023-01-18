from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score


def grid_search(model, params, cv):
    '''
    :param model: 模型
    :param params: 需要优化的超参数
    :param cv: 交叉验证次数
    :return: 对应网格搜索
    '''
    return GridSearchCV(model, params, cv)


def random_search(model, params, n_iter):
    '''
    :param model: 模型
    :param params: 需要优化的超参数
    :param n_iter: 迭代次数
    :return: 对应模型随机搜索算法
    '''
    return RandomizedSearchCV(model, params, n_iter)


def bayesian_optimization(model, data, label, algorithm, params, max_evals):
    '''
    :param model:
    :param algorithm:
    :param params:
    :param max_evals:
    :return:
    '''
    # 定义目标损失函数
    scores = cross_val_score(model, data, label, cv=5)
    loss = 1 - scores.mean()
    # 定义超参数搜索空间
    # space = {'kernel': hp.choice('kernel', ['linear', 'rbf']),
    #          'C': hp.uniform('C', 1, 10)}
    space = params
    best = fmin(loss, space, algorithm, max_evals)
    return best


def hyperband_optimization():
    '''

    :return:
    '''
    return None
