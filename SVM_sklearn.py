from data_preprocessing import data_preprocessing
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score

X_train, y_train, X_test, y_test, X_val, y_val = data_preprocessing()
print("Hoàn thành xử lý dữ liệu")

model = SVC()
model = SVC(kernel='linear', C=1)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"precision: {precision_score(y_test, y_pred)}")
print(f"recall: {recall_score(y_test, y_pred)}")
print(f"f1_score: {f1_score(y_test, y_pred)}")