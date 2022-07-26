import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

#mengambil sebagian kolom untuk dipisahkan data kolom label dan pixel
def input_excel():
    file_excel = pd.read_excel('dataset.xlsx')
    label = file_excel[file_excel.columns[2:786]]
    pixel = file_excel[file_excel.columns[1]]
    return label, pixel

#mengubah dataframe menjadi array
def ubah_array(kolom):
    kolom = np.array(kolom)
    return kolom

#convert array 1 dimensi
def flat_array(flatted):
    flatted = flatted.flatten()
    return flatted

#Program membagi data latih dan data uji dengan komposisi 80%/20%
def split_data(label, pixel):
    X_train, X_test, y_train, y_test = train_test_split(label,pixel,test_size=0.2, random_state=35)
    return X_train, X_test, y_train, y_test

#Program melakukan normalisasi data menggunakan standard scaler
def normalisasi(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

#mengambil sebagian kolom untuk dipisahkan data kolom label dan pixel
label, pixel = input_excel()

#mengubah dataframe menjadi array
label = ubah_array(label)
pixel = ubah_array(pixel)

#convert array 1 dimensi
pixel = flat_array(pixel)

#Program membagi data latih dan data uji dengan komposisi 80%/20%
X_train, X_test, y_train, y_test = split_data(label, pixel)

#Program melakukan normalisasi data menggunakan standard scaler
X_train, X_test = normalisasi(X_train, X_test)

#list accuracy_arr
accuracy_arr = []

#perhitungan KNN (1, 3, 5, 7, 9) dengan 5 Fold Cross Validation
for i in range(1,10,2):
    knn = KNeighborsClassifier(n_neighbors=i, metric='chebyshev').fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    result = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    accuracy_arr.append(result.mean())

#Membuat dataframe dengan tiap kolom yang berisi hasil kalkulasi KNN k=1
data1 = pd.DataFrame({
    'idData' : np.arange(1, len(y_test) + 1),
    'label aktual' : y_test,
    'prediksi' : y_pred,
    'label luaran' : np.NaN
    })
data1.at[0,'label luaran'] = accuracy_arr[0]

#Membuat dataframe dengan tiap kolom yang berisi hasil kalkulasi KNN k=3
data2 = pd.DataFrame({
    'idData' : np.arange(1, len(y_test) + 1),
    'label aktual' : y_test,
    'prediksi' : y_pred,
    'label luaran' : np.NaN
    })
data2.at[0,'label luaran'] = accuracy_arr[1]

#Membuat dataframe dengan tiap kolom yang berisi hasil kalkulasi KNN k=5
data3 = pd.DataFrame({
    'idData' : np.arange(1, len(y_test) + 1),
    'label aktual' : y_test,
    'prediksi' : y_pred,
    'label luaran' : np.NaN
    })
data3.at[0,'label luaran'] = accuracy_arr[2]

#Membuat dataframe dengan tiap kolom yang berisi hasil kalkulasi KNN k=7
data4 = pd.DataFrame({
    'idData' : np.arange(1, len(y_test) + 1),
    'label aktual' : y_test,
    'prediksi' : y_pred,
    'label luaran' : np.NaN
    })
data4.at[0,'label luaran'] = accuracy_arr[3]

#Membuat dataframe dengan tiap kolom yang berisi hasil kalkulasi KNN k=9
data5 = pd.DataFrame({
    'idData' : np.arange(1, len(y_test) + 1),
    'label aktual' : y_test,
    'prediksi' : y_pred,
    'label luaran' : np.NaN
    })
data5.at[0,'label luaran'] = accuracy_arr[4]

#menggabungkan setiap dataframe pada sheet yang berbeda dalam file OutputValidasi.xlsx
with pd.ExcelWriter(r"Supremum\OutputValidasi.xlsx") as hasil:
    data1.to_excel(hasil, sheet_name="k=1", index=False)
    data2.to_excel(hasil, sheet_name="k=3", index=False)
    data3.to_excel(hasil, sheet_name="k=5", index=False)
    data4.to_excel(hasil, sheet_name="k=7", index=False)
    data5.to_excel(hasil, sheet_name="k=9", index=False)

#perhitungan KNN (1 sampai 5) dengan 5 Fold Cross Validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
for i in range(1,10,2):
    knn = KNeighborsClassifier(n_neighbors=i, metric='minkowski').fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    result = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    accuracy_arr.append(result.mean())

#Membuat dataframe dengan tiap kolom yang berisi hasil kalkulasi KNN k=1
data1 = pd.DataFrame({
    'idData' : np.arange(1, len(y_test) + 1),
    'label aktual' : y_test,
    'prediksi' : y_pred,
    'label luaran' : np.NaN
    })
data1.at[0,'label luaran'] = accuracy_arr[5]

#Membuat dataframe dengan tiap kolom yang berisi hasil kalkulasi KNN k=3
data2 = pd.DataFrame({
    'idData' : np.arange(1, len(y_test) + 1),
    'label aktual' : y_test,
    'prediksi' : y_pred,
    'label luaran' : np.NaN
    })
data2.at[0,'label luaran'] = accuracy_arr[6]

#Membuat dataframe dengan tiap kolom yang berisi hasil kalkulasi KNN k=5
data3 = pd.DataFrame({
    'idData' : np.arange(1, len(y_test) + 1),
    'label aktual' : y_test,
    'prediksi' : y_pred,
    'label luaran' : np.NaN
    })
data3.at[0,'label luaran'] = accuracy_arr[7]

#Membuat dataframe dengan tiap kolom yang berisi hasil kalkulasi KNN k=7
data4 = pd.DataFrame({
    'idData' : np.arange(1, len(y_test) + 1),
    'label aktual' : y_test,
    'prediksi' : y_pred,
    'label luaran' : np.NaN
    })
data4.at[0,'label luaran'] = accuracy_arr[8]

#Membuat dataframe dengan tiap kolom yang berisi hasil kalkulasi KNN k=9
data5 = pd.DataFrame({
    'idData' : np.arange(1, len(y_test) + 1),
    'label aktual' : y_test,
    'prediksi' : y_pred,
    'label luaran' : np.NaN
    })
data5.at[0,'label luaran'] = accuracy_arr[9]

#menggabungkan setiap dataframe pada sheet yang berbeda dalam file OutputValidasi.xlsx
with pd.ExcelWriter(r"Minkowski\OutputValidasi.xlsx") as hasil:
    data1.to_excel(hasil, sheet_name="k=1", index=False)
    data2.to_excel(hasil, sheet_name="k=3", index=False)
    data3.to_excel(hasil, sheet_name="k=5", index=False)
    data4.to_excel(hasil, sheet_name="k=7", index=False)
    data5.to_excel(hasil, sheet_name="k=9", index=False)