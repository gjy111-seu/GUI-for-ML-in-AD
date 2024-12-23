import pandas as pd
import os
import numpy as np
import xgboost as xgb

from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 允许所有来源的跨域请求


def model_save_load(model_path):
    clf = xgb.XGBRegressor(verbosity=0,
                           n_jobs=1,
                           learning_rate=0.15,  #0.15
                           n_estimators=200,
                           max_depth=2,
                           min_child_weight=10,
                           subsample=0.8)
    clf.load_model(model_path)
    return clf


@app.route('/', methods=['GET', 'POST'])
def home():
    data_get = request.get_json()  # 获取JSON数据
    data_list = data_get['data']
    print(data_list)
    data = {
        'SSA': [float(data_list[1])],  # 示例数据，你可以根据需要添加更多数据
        'APD': [float(data_list[2])],
        'pH1': [float(data_list[3])],
        'EC': [float(data_list[4])],
        'PS': [float(data_list[5])],
        'C': [float(data_list[6])],
        'H': [float(data_list[7])],
        'O': [float(data_list[8])],
        'N': [float(data_list[9])],
        'VS/TS': [float(data_list[10])],
        'S/I': [float(data_list[11])],
        'Dose': [float(data_list[12])],
        'Temp': [float(data_list[13])],
        'pH2': [float(data_list[14])],
    }

    data_manual = pd.DataFrame(data)
    res_result = [0, 0, 0]

    # data1 = pd.read_csv('./data_demo.csv', sep='\t')
    j = 0
    for i in ['BMY', 'ICR', 'LAG']:
        model_path = './{}_XGB'.format(i)
        model_new = model_save_load(model_path)
        y_predict_demo = model_new.predict(data_manual)
        print(y_predict_demo)
        result = y_predict_demo
        res_result[j] = result[0]
        j = j + 1

    # data = pd.read_csv('./data_demo.csv', sep='\t')
    # model_path = './xgboost_classifier_model.model'
    # model_new = model_save_load(model_path)
    # y_predict_demo = model_new.predict(data_manual)
    # result = y_predict_demo
    print(str(res_result))
    return str(res_result)


if __name__ == "__main__":
    app.run(debug=True)
    # print("hello")
    # data = pd.read_csv('./data_demo.csv',sep='\t')
    # for i in ['BMP','CPR','λ']:
    #     model_path = './xgboost_classifier_model_{}.model'.format(i)
    #     model_new = model_save_load(model_path)
    #
    #     y_predict_demo = model_new.predict(data)
    #
    #     result = y_predict_demo
    #     print('new_result_{}:'.format(i),result)
