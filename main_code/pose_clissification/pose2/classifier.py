#author: wmingzhu
#date: 2024/07/02
import torch
from torch.utils.data import Dataset,DataLoader
import random
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from pose_utils import count_samples
import lightgbm as lgb
import pandas as pd
import numpy as np
import joblib
class LightGBM:
    def __init__(self,params,train_data="./train_data/train_indexed.txt",model_save_path="./models/lgbm.txt",num_boost_round=1000,best_model="./models/best_gbm.joblib"):
        self.params = params
        self.train_data = train_data
        self.model_save_path = model_save_path
        self.num_boost_round = num_boost_round
        self.best_model = best_model
    def load_lgb_data(self,test_size=0.4):
        df = pd.read_csv(self.train_data,header=None,sep=",")
        data = df.iloc[:,1:].values
        label = df.iloc[:,0].values
        train_data = data
        train_label = label
        _, val_data, _, val_label = train_test_split(data,label,test_size=test_size, random_state=42)
        return train_data, val_data, train_label,val_label

    def train(self):
        train_data, val_data, train_label,val_label = self.load_lgb_data()
        print(len(train_data),len(val_data))
        lgb_train = lgb.Dataset(train_data, train_label)
        lgb_eval = lgb.Dataset(val_data, val_label, reference=lgb_train)

        gbm = lgb.train(self.params,
                        lgb_train,
                        num_boost_round=self.num_boost_round,
                        valid_sets=lgb_eval)

        gbm.save_model(self.model_save_path)
        y_pred = gbm.predict(val_data, num_iteration=gbm.best_iteration)
        y_pred_class = np.argmax(y_pred, axis=1)
        # 特征名称
        print('特征名称:')
        print(gbm.feature_name())
        # 特征重要度
        print('特征重要度:')
        print(list(gbm.feature_importance()))
        print("Accuracy:", accuracy_score(val_label, y_pred_class))
        print(classification_report(val_label, y_pred_class))
        return gbm
    def train_grid(self):
        # param_grid = {
        #     'num_leaves': [31, 50,70, 100],
        #     'learning_rate': [0.009,0.02,0.05, 0.1, 0.2],
        #     'n_estimators': [20, 30,40, 50,80],
        #     'subsample': [0.5, 0.7, 0.9,1],
        #     'colsample_bytree': [0.5, 0.7, 0.9,1],
        #     'bagging_freq': [0,5,8,10],
        #     'lambda_l2': [0,0.001,0.005]
        # }
        param_grid = {
            'num_leaves': [20,31],
            'learning_rate': [0.009,0.05],
            'n_estimators': [20, 30],
            'subsample': [0.5, 0.7, 0.9],
            'colsample_bytree': [0.5, 0.7, 0.9],
            'bagging_freq': [5],
            'num_boost_round': [self.num_boost_round]
        }
        estimator = lgb.LGBMClassifier(objective='multiclass', metric='multi_logloss', num_class=6)
        grid_search = GridSearchCV(estimator, param_grid, scoring='accuracy', cv=3, n_jobs=-1)
        train_data, test_data, train_label, test_label = self.load_lgb_data()
        print(len(train_data),len(test_data))
        grid_search.fit(train_data, train_label)
        # 输出最佳参数组合
        print("Best parameters found: ", grid_search.best_params_)
        print("Best score found: ", grid_search.best_score_)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(test_data)
        print("Accuracy on test set: ", accuracy_score(test_label, y_pred))
        print("Classification Report: \n", classification_report(test_label, y_pred))
        joblib.dump(best_model, self.best_model)

class PoseDataset(Dataset):
    def __init__(self,data_path=None,mode="train"):
        with open(data_path,"r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            # print("各类别总样本个数:{}".format(count_samples(lines)))
            total = len(lines)
            random.shuffle(lines)#实际上既然是shuffle的，下面的按照前后百分比取意义不大
            if mode == "train":
                # lines = lines[:int(0.8*len(lines))]
                lines = lines #数据少，暂时拿全部数据训练
            elif mode == "val":
                lines = lines[int(0.6 * len(lines)):]

        # print("各类别{}样本个数:{}".format(mode,count_samples(lines)))
        self.data_source = lines

    def __getitem__(self,item):
        line = self.data_source[item].split(",")
        label = torch.tensor(int(line[0]))#默认是torch.int64。范围是0到number_classes-1
        data = torch.tensor([float(i) for i in line[1:]])#默认是torch.float32
        return data,label

    def __len__(self):
        return len(self.data_source)

class PoseClassifier(torch.nn.Module):
    def __init__(self, input_dim,num_classes):
        super(PoseClassifier, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim,128)
        self.activation = torch.nn.LeakyReLU(0.01)
        self.drop = torch.nn.Dropout(0.2)
        self.linear2 = torch.nn.Linear(128,256)
        # self.linear3 = torch.nn.Linear(256,128)
        self.output = torch.nn.Linear(256,num_classes)

    def forward(self, x):
        #多分类损失函数nn.CrossEntropyLoss()内部做了softmax和标签的one-hot，所以不用我们显式做这两个东西
        out = self.linear1(x)
        out = self.activation(out)
        # # out = self.drop(out)
        out = self.linear2(out)
        out = self.activation(out)
        # # out = self.drop(out)
        # out = self.linear3(out)
        # out = self.activation(out)
        out = self.output(out)
        return out


if __name__ == "__main__":
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 6,
        'metric': {'multi_logloss'},
        'num_leaves': 31,
        'learning_rate': 0.009,
        'colsample_bytree': 0.9,
        'subsample': 0.9,
        'bagging_freq': 5,
        'lambda_l2': 0,
        'verbose': 0
    }
    lgbm = LightGBM(params=params,num_boost_round=1000,train_data="./train_data/merged_indexed.txt")
    gbm_model = lgbm.train()
    # model = lgb.Booster(model_file="./models/lgbm.txt")
    # results = model.predict([[0.0,0.0,0.44522902,0.42797977,0.42548296,0.42939463,0.43283972,0.44478908,0.0,0.0,0.44119215,0.49945438,0.0,0.0,0.4552446,0.52110904,0.39579314,0.5014287,0.400371,0.5122012,0.43441874,0.5071763,0.44037935,0.5308573,0.4124673,0.5521924,0.41139397,0.5661066]])
    # print(np.argmax(results, axis=1))
    # print(model.params)

    # lgbm.train_grid()
    # model = joblib.load("./models/best_gbm.joblib")


