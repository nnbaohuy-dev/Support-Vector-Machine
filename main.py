from data_preprocessing import data_preprocessing
from SVM import SVM

X_train, y_train, X_test, y_test, X_val, y_val = data_preprocessing()
print("Hoàn thành xử lý dữ liệu")

model = SVM(150, 0.0001, 0.005)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

model.evaluate(y_test, y_pred)