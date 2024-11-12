import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# 读取CSV文件
data = pd.read_csv("300.csv")

# 分离特征和标签
X = data.iloc[:, 0:4].astype(float)  # 取前四列作为特征，并转换为浮点数
y = data.iloc[:, 4].astype(float)    # 取第五列作为标签，并转换为浮点数

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=46)

# 特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled = scaler.transform(X)  # 对完整数据集进行缩放
# 定义SVR模型
svr_model = SVR()

# 定义参数搜索范围
param_grid = {
    'C': [0.1, 1, 10, 100],  # 惩罚参数
    'gamma': [0.001, 0.01, 0.1, 1],  # 核函数的系数
    'kernel': ['rbf', 'linear', 'poly']  # 核函数类型
}

# 创建GridSearchCV对象
grid_search = GridSearchCV(svr_model, param_grid, cv=5, scoring='neg_mean_squared_error')

# 在训练集上进行参数搜索
grid_search.fit(X_train_scaled, y_train)

# 输出最佳参数组合和对应的负均方误差
print("最佳参数组合：", grid_search.best_params_)
print("最佳负均方误差：", grid_search.best_score_)

# 使用最佳参数组合的模型进行预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
y_pred_full = best_model.predict(X_scaled)  # 在完整数据集上进行预测

# 计算评估指标
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"在测试集上的均方误差：{mse:.2f}")
print(f"在测试集上的均方根误差：{rmse:.2f}")
print(f"在测试集上的平均绝对误差：{mae:.2f}")
print(f"在测试集上的R²得分：{r2:.2f}")

# 新数据点
new_data = np.array([[0.5, 0.5, 200, 300]])
new_data_scaled = scaler.transform(new_data)

# 使用之前训练好的模型进行预测
predicted_value = best_model.predict(new_data_scaled)
print("预测值：", predicted_value)

# 特征重要性曲线
# 使用随机森林回归计算特征重要性
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
importances = rf_model.feature_importances_

# 绘制特征重要性曲线
plt.figure(figsize=(8, 6))
plt.barh(range(len(importances)), importances, align='center')
plt.yticks(range(len(importances)), data.columns[:4])
plt.title('Importance of feature(300μm dataset)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# 绘制真实值与预测值的散点图
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y)), y, color='blue', edgecolor='k', alpha=0.7, label='True Value')
plt.scatter(range(len(y)), y_pred_full, color='red', edgecolor='k', alpha=0.7, label='Predict Value')
plt.xlabel('Index')
plt.ylabel('LL')
plt.title('Predict/True Value(300μm dataset)')
# 设置y轴的最大值为3.5
plt.ylim(top=3.5)

plt.legend()
plt.grid(True)
plt.show()