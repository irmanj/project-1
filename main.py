import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# data sederhana
data = {
    'jam_belajar': [1,2,3,4,5,6,7,8],
    'kehadiran': [50,60,70,80,85,90,95,100],
    'lulus': [0,0,0,1,1,1,1,1]
}

df = pd.DataFrame(data)

X = df[['jam_belajar', 'kehadiran']]
y = df['lulus']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# coba prediksi baru
print("Prediksi:", model.predict([[5, 85]]))