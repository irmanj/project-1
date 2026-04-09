import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ======================
# LOAD DATA (CSV)
# ======================
def load_data():
    df = pd.read_csv("data.csv")
    return df

# ======================
# PREPROCESSING
# ======================
def preprocess(df):
    X = df[['jam_belajar', 'kehadiran']]
    y = df['lulus']
    return X, y

# ======================
# TRAIN MODEL
# ======================
def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

# ======================
# EVALUATION
# ======================
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ======================
# VISUALIZATION
# ======================
def visualize(df):
    plt.scatter(df['jam_belajar'], df['lulus'])
    plt.xlabel("Jam Belajar")
    plt.ylabel("Lulus")
    plt.title("Hubungan Jam Belajar vs Kelulusan")
    plt.show()

# ======================
# USER INPUT PREDICTION
# ======================
def predict_user(model):
    try:
        jam = float(input("Masukkan jam belajar: "))
        hadir = float(input("Masukkan kehadiran: "))

        if jam < 0 or hadir < 0:
            print("Input tidak valid!")
            return

        result = model.predict([[jam, hadir]])
        print("Prediksi Lulus:", result[0])

    except:
        print("Input harus angka!")

# ======================
# MAIN PROGRAM
# ======================
def main():
    df = load_data()
    X, y = preprocess(df)

    visualize(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    # simpan model
    joblib.dump(model, "model.pkl")

    # prediksi user
    predict_user(model)

if __name__ == "__main__":
    main()