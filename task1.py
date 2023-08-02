from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

dataFrame= pd.read_csv("C:\Users\Harshal\Downloads\prodgytask1")
dataFrame.head()
print(dataFrame['SalePrice'].describe())

A= dataFrame[['GrLivArea', 'BsmtFullBath', 'BsmtHalfBath','BedroomAbvGr','FullBath','HalfBath']]
B= dataFrame['SalePrice']

A_train, A_text, B_train, B_test = train_test_split(A,B, random_state=1)

mode1 = LinearRegression()
mode1.fit(A_train,B_train)

y_pred = mode1.predict(A_text)

print('Mean squared error: %.2f' % mean_squared_error(B_test,y_pred))
print('Coefficiant of determination: %.2f' % r2_score(B_test,y_pred))