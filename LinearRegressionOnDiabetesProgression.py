#linear regression on diabetes dataset for predicting extent of diabetes progression after one year by considering attributes Age, Sex, BMI, Average Blood pressure, 6 serum levels

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
diabetes = load_diabetes()

X_train, X_test, y_train, y_test = train_test_split(diabetes['data'],diabetes['target'], 
                                                    random_state = 42)

lr = LinearRegression().fit(X_train,y_train)

print("lr coefficients: {}".format(lr.coef_))
print("lr interecept: {}".format(lr.intercept_))

print("lr training score: {:.2f}".format(lr.score(X_train,y_train)))
print("lr test score: {:.2f}".format(lr.score(X_test,y_test)))

#bad values of R^2 score of 0.52, 0.48 (train and test scores) 
#imply that this is not a great fit
