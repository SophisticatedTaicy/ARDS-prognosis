import os

import numpy as np
from matplotlib import pyplot as plt
from numpy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from filter.common import read_file, concat_array, init_component, data_split_and_combine, judge_label_balance, \
    format_label, standard_data_by_white
from filter.param import outcome_dict, colors, marks
from ml.classification.classify_parameter import GBDT, GBDT_param

base_picture_path = './ARDS/combine/pictures/'
base_csv_path = './ARDS/combine/csvfiles'
base_path = os.path.dirname(os.path.abspath(__file__)) + '\ARDS'
from ARDS.data_process.process import Processing


class model_process:
    def __init__(self, datatype=None, insertnum=20):
        self.datatype = datatype
        # respontaneous recovery
        self.mean_tpr_0, self.mean_fpr_0, self.tprs_0, self.combine_mean_tpr_0, self.combine_mean_fpr_0, self.combine_tprs_0 = init_component(
            insertnum)
        # long stay
        self.mean_tpr_1, self.mean_fpr_1, self.tprs_1, self.combine_mean_tpr_1, self.combine_mean_fpr_1, self.combine_tprs_1 = init_component(
            insertnum)
        # rapid death
        self.mean_tpr_2, self.mean_fpr_2, self.tprs_2, self.combine_mean_tpr_2, self.combine_mean_fpr_2, self.combine_tprs_2 = init_component(
            insertnum)
        # variance
        self.variance_0, self.variance_1, self.variance_2, self.combine_variance_0, self.combine_variance_1, self.combine_variance_2 = [], [], [], [], [], []

    def train_model(self, x_train, y_train, x_combine_train, y_combine_train, x_test, y_test, label):
        # gclf = RandomizedSearchCV(xgb.XGBClassifier(objective="binary:logistic", tree_method='gpu_hist'), XGB_param,
        #                           scoring='roc_auc', n_jobs=12, cv=5)
        gclf = RandomizedSearchCV(GBDT, GBDT_param, scoring='roc_auc', n_jobs=10, cv=5)
        # train on single dataset
        x_test_new = np.array(x_test)
        x_train, x_test = standard_data_by_white(x_train, x_test)
        gclf.fit(x_train, y_train)
        # get best estimator
        model = gclf.best_estimator_
        # predict
        y_predict = model.predict_proba(x_test)
        fpr, tpr, threshold = roc_curve(y_test, y_predict[:, 1])
        if label == 0:
            self.variance_0.append(auc(fpr, tpr))
            self.tprs_0.append(interp(self.mean_fpr_0, fpr, tpr))
            self.mean_tpr_0 = np.mean(self.tprs_0, axis=0)
        elif label == 1:
            self.variance_1.append(auc(fpr, tpr))
            self.tprs_1.append(interp(self.mean_fpr_1, fpr, tpr))
            self.mean_tpr_1 = np.mean(self.tprs_1, axis=0)
        elif label == 2:
            self.variance_2.append(auc(fpr, tpr))
            self.tprs_2.append(interp(self.mean_fpr_2, fpr, tpr))
            self.mean_tpr_2 = np.mean(self.tprs_2, axis=0)
        # train on combine dataset
        # combine_gclf = RandomizedSearchCV(xgb.XGBClassifier(objective="binary:logistic", tree_method='gpu_hist'),
        #                                   XGB_param, scoring='roc_auc', n_jobs=12, cv=5)
        combine_gclf = RandomizedSearchCV(GBDT, GBDT_param, scoring='roc_auc', n_jobs=10, cv=5)
        x_combine_train, x_test_new = standard_data_by_white(x_combine_train, x_test_new)
        combine_gclf.fit(x_combine_train, y_combine_train)
        # 得到最佳模型
        combine_model = combine_gclf.best_estimator_
        # 训练集性能
        y_combine_predict = combine_model.predict_proba(x_test_new)
        fpr_combine, tpr_combine, threshold_combine_ = roc_curve(y_test, y_combine_predict[:, 1])
        if label == 0:
            self.combine_variance_0.append(auc(fpr_combine, tpr_combine))
            self.combine_tprs_0.append(interp(self.combine_mean_fpr_0, fpr_combine, tpr_combine))
            self.combine_mean_tpr_0 = np.mean(self.combine_tprs_0, axis=0)
        elif label == 1:
            self.combine_variance_1.append(auc(fpr_combine, tpr_combine))
            self.combine_tprs_1.append(interp(self.combine_mean_fpr_1, fpr_combine, tpr_combine))
            self.combine_mean_tpr_1 = np.mean(self.combine_tprs_1, axis=0)
        elif label == 2:
            self.combine_variance_2.append(auc(fpr_combine, tpr_combine))
            self.combine_tprs_2.append(interp(self.combine_mean_fpr_2, fpr_combine, tpr_combine))
            self.combine_mean_tpr_2 = np.mean(self.combine_tprs_2, axis=0)

    # 绘制每种数据集不同预后的结果
    def plot_images(self):
        plt.figure(0)
        mean_auc = round(auc(self.mean_fpr_0, self.mean_tpr_0), 3)
        combine_mean_auc = round(auc(self.combine_mean_fpr_0, self.combine_mean_tpr_0), 3)
        plt.plot(self.mean_fpr_0, self.mean_tpr_0, label=r'%s test (area= %s)' % (self.datatype, mean_auc),
                 color='deepskyblue')
        plt.plot(self.combine_mean_fpr_0, self.combine_mean_tpr_0, label=r'combine test (area= %s)' % combine_mean_auc,
                 color='salmon')
        plt.plot([0, 1], [0, 1], linestyle='--', lw=3, color='gray', alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False positive rate', fontsize=15)
        plt.ylabel('True positive rate', fontsize=15)
        plt.title('Spontaneous recovery ROCs', fontweight='bold', fontsize=15)
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig(base_picture_path + str(self.datatype) + '_' + 'Spontaneous_recovery_24.jpg')
        plt.show()
        plt.figure(1)
        print('mean_fpr_1 : %s mean_tpr_1 : %s ' % (self.mean_fpr_1, self.mean_tpr_1))
        mean_auc = round(auc(self.mean_fpr_1, self.mean_tpr_1), 3)
        combine_mean_auc = round(auc(self.combine_mean_fpr_1, self.combine_mean_tpr_1), 3)
        plt.plot(self.mean_fpr_1, self.mean_tpr_1, label=r'%s test (area= %s)' % (self.datatype, mean_auc),
                 color='deepskyblue')
        plt.plot(self.combine_mean_fpr_1, self.combine_mean_tpr_1, label=r'combine test (area= %s)' % combine_mean_auc,
                 color='salmon')
        plt.plot([0, 1], [0, 1], linestyle='--', lw=3, color='gray', alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False positive rate', fontsize=15)
        plt.ylabel('True positive rate', fontsize=15)
        plt.title('Long stay ROCs', fontweight='bold', fontsize=15)
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig(base_picture_path + str(self.datatype) + '_' + 'Long_stay_24.jpg')
        plt.show()
        plt.figure(2)
        mean_auc = round(auc(self.mean_fpr_2, self.mean_tpr_2), 3)
        combine_mean_auc = round(auc(self.combine_mean_fpr_2, self.combine_mean_tpr_2), 3)
        plt.plot(self.mean_fpr_2, self.mean_tpr_2, label=r'%s test (area= %s)' % (self.datatype, mean_auc),
                 color='deepskyblue')
        plt.plot(self.combine_mean_fpr_2, self.combine_mean_tpr_2, label=r'combine test (area= %s)' % combine_mean_auc,
                 color='salmon')
        plt.plot([0, 1], [0, 1], linestyle='--', lw=3, color='gray', alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False positive rate', fontsize=15)
        plt.ylabel('True positive rate', fontsize=15)
        plt.title('Rapid death ROCs', fontweight='bold', fontsize=15)
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig(base_picture_path + str(self.datatype) + '_' + 'Rapid_death_24.svg', format='svg')
        plt.show()

    # 在多个机器学习模型上融合训练数据，在统一测试集上查看模型性能
    # 使用不同模型测试融合数据性能
    def various_model(self, outcome, x_train, x_test, y_train, y_test):
        from ml.classification.classify_parameter import base_models, searchCVnames
        # 将训练集和测试集白化
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        x_train, x_test = standard_data_by_white(np.array(x_train), np.array(x_test))
        for name, model, color, mark in zip(searchCVnames, base_models, colors[:len(base_models)],
                                            marks[:len(base_models)]):
            model.fit(x_train, y_train)
            if name == 'Perceptron':
                test_predict_proba = model._predict_proba_lr(x_test)
                fpr, tpr, threshold = roc_curve(y_test, test_predict_proba[:, 1], pos_label=1)
            elif name == 'LinearRegression' or name == 'BayesianRidge':
                test_predict_proba = model.predict(x_test)
                fpr, tpr, threshold = roc_curve(y_test, test_predict_proba, pos_label=1)
            else:
                test_predict_proba = model.predict_proba(x_test)
                fpr, tpr, threshold = roc_curve(y_test, test_predict_proba[:, 1], pos_label=1)
            test_auc = round(auc(fpr, tpr), 3)
            plt.plot(fpr[:-1:30], tpr[:-1:30], label=r'%s (area=%.3f)' % (name, test_auc), color=color, marker=mark,
                     lw=1.5)
        plt.title(r'%s' % outcome, fontweight='bold', fontsize=15)
        plt.xlabel('False positive rate', fontsize=15, fontweight='bold')
        plt.ylabel('True positive rate', fontsize=15, fontweight='bold')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.grid()
        plt.legend(loc=4, fontsize=7)
        plt.savefig(base_picture_path + 'combine_models_' + str(outcome) + '.png')
        plt.show()


if __name__ == '__main__':
    eicu_path = os.path.join('eicu', 'csvfiles')
    mimic3_path = os.path.join('mimic', 'mimic3', 'csvfiles')
    mimic4_path = os.path.join('mimic', 'mimic4', 'csvfiles')
    base_name = 'new_1228'
    eicu = read_file(path=os.path.join(base_path, eicu_path), filename=base_name)
    mimic3 = read_file(path=os.path.join(base_path, mimic3_path), filename=base_name)
    mimic4 = read_file(path=os.path.join(base_path, mimic4_path), filename=base_name)
    # extract common columns
    common_columns = list(set(eicu.columns).intersection(set(mimic3.columns), set(mimic4.columns)))
    print('common columns : %s, common columns length : %s' % (common_columns, len(common_columns)))
    processer = Processing()
    # respectively extract common columns data  from eicu,mimic iii, mimic iv
    eicu_label, mimic3_label, mimic4_label, eicu_data, mimic3_data, mimic4_data = data_split_and_combine(eicu,
                                                                                                         mimic3,
                                                                                                         mimic4,
                                                                                                         common_columns)
    eicu_processer = model_process(datatype='eICU')
    mimic3_processer = model_process(datatype='MIMIC III')
    mimic4_processer = model_process(datatype='MIMIC IV')
    for outcome, label in outcome_dict.items():
        i = 0
        while i < 10:
            # split train and test set for training in each dataset
            eicu_x_train, eicu_x_test, eicu_y_train_ori, eicu_y_test_ori = train_test_split(eicu_data, eicu_label,
                                                                                            test_size=0.2)
            mimic3_x_train, mimic3_x_test, mimic3_y_train_ori, mimic3_y_test_ori = train_test_split(mimic3_data,
                                                                                                    mimic3_label,
                                                                                                    test_size=0.2)
            mimic4_x_train, mimic4_x_test, mimic4_y_train_ori, mimic4_y_test_ori = train_test_split(mimic4_data,
                                                                                                    mimic4_label,
                                                                                                    test_size=0.2)
            # confirm that each train and test set labels are not single one classes
            if judge_label_balance(eicu_y_train_ori, eicu_y_test_ori) and judge_label_balance(
                    mimic3_y_train_ori, mimic3_y_test_ori) and judge_label_balance(
                mimic4_y_train_ori, mimic4_y_test_ori):
                eicu_y_train = format_label(eicu_y_train_ori, label)
                eicu_y_test = format_label(eicu_y_test_ori, label)
                mimic3_y_train = format_label(mimic3_y_train_ori, label)
                mimic3_y_test = format_label(mimic3_y_test_ori, label)
                mimic4_y_train = format_label(mimic4_y_train_ori, label)
                mimic4_y_test = format_label(mimic4_y_test_ori, label)
                # 融合数据
                combine_x_train = concat_array([eicu_x_train, mimic3_x_train, mimic4_x_train])
                combine_y_train = concat_array([eicu_y_train, mimic3_y_train, mimic4_y_train])
                combine_x_test = concat_array([eicu_x_test, mimic3_x_test, mimic4_x_test])
                combine_y_test = concat_array([eicu_y_test, mimic3_y_test, mimic4_y_test])
                eicu_processer.train_model(eicu_x_train, eicu_y_train, combine_x_train, combine_y_train, eicu_x_test,
                                           eicu_y_test, label)
                mimic3_processer.train_model(mimic3_x_train, mimic3_y_train, combine_x_train, combine_y_train,
                                             mimic3_x_test, mimic3_y_test, label)
                mimic4_processer.train_model(mimic4_x_train, mimic4_y_train, combine_x_train, combine_y_train,
                                             mimic4_x_test, mimic4_y_test, label)
                i += 1
    # plot average performance by tprs and fprs with 10 times for each prognosis
    eicu_processer.plot_images()
    mimic3_processer.plot_images()
    mimic4_processer.plot_images()
