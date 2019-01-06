import pandas as pd
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus
from sklearn.externals.six import StringIO
#导入第三方包
from sklearn import ensemble

# 构建随机森林
RF_class = ensemble.RandomForestClassifier(n_estimators=200, random_state=1234)
# 随机森林的拟合
RF_class.fit(X_train, y_train)
# 模型在测试集上的预测
RFclass_pred = RF_class.predict(X_test)
# 模型的准确率
print('模型在测试集的预测准确率：\n', metrics.accuracy_score(y_test, RFclass_pred))

# 计算绘图数据
y_score = RF_class.predict_proba(X_test)[:, 1]
fpr, tpr, threshold = metrics.roc_curve(y_test, y_score)
roc_auc = metrics.auc(fpr, tpr)
# 绘图
plt.stackplot(fpr, tpr, color='steelblue', alpha=0.5, edgecolor='black')
plt.plot(fpr, tpr, color='black', lw=1)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.text(0.5, 0.3, 'ROC curve (area = %0.2f)' % roc_auc)
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.show()

# 变量的重要性程度值
importance = RF_class.feature_importances_
# 构建含序列用于绘图
Impt_Series = pd.Series(importance, index=X_train.columns)
# 对序列排序绘图
Impt_Series.sort_values(ascending=True).plot('barh')
plt.show()

# 读入数据
NHANES = pd.read_excel(r'F:\pycharm workspace\test\NHANES.xlsx')
NHANES.head()
print(NHANES.shape)

# 取出自变量名称
predictors = NHANES.columns[:-1]
# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = model_selection.train_test_split(NHANES[predictors], NHANES.CKD_epi_eGFR,
                                                                    test_size=0.25, random_state=1234)

# 预设各参数的不同选项值
max_depth = [18, 19, 20, 21, 22]
min_samples_split = [2, 4, 6, 8]
min_samples_leaf = [2, 4, 8]
parameters = {'max_depth': max_depth, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}
# 网格搜索法，测试不同的参数值
grid_dtreg = GridSearchCV(estimator=tree.DecisionTreeRegressor(), param_grid=parameters, cv=10)
# 模型拟合
grid_dtreg.fit(X_train, y_train)
# 返回最佳组合的参数值
grid_dtreg.best_params_

# 构建用于回归的决策树
CART_Reg = tree.DecisionTreeRegressor(max_depth=20, min_samples_leaf=2, min_samples_split=4)
# 回归树拟合
CART_Reg.fit(X_train, y_train)
# 模型在测试集上的预测
pred = CART_Reg.predict(X_test)
# 计算衡量模型好坏的MSE值
metrics.mean_squared_error(y_test, pred)

# 构建用于回归的随机森林
RF = ensemble.RandomForestRegressor(n_estimators=200, random_state=1234)
# 随机森林拟合
RF.fit(X_train, y_train)
# 模型在测试集上的预测
RF_pred = RF.predict(X_test)
# 计算模型的MSE值
metrics.mean_squared_error(y_test, RF_pred)

# 构建变量重要性的序列
importance = pd.Series(RF.feature_importances_, index=X_train.columns)
# 排序并绘图
importance.sort_values().plot('barh')
plt.show()		