from sklearn.calibration import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import joblib
import matplotlib.pyplot as plt

class Bayes:
    def __init__(self) -> None:
        pass

    def start(self, df, percentage):
        X = df[['tgl_faktur', 'NIK', 'Alamat']]
        y = df['Tindak lanjut']
        
        # Inisialisasi label encoder
        label_encoder = LabelEncoder()

        # Melakukan label encoding pada fitur
        X['tgl_faktur'] = label_encoder.fit_transform(X['tgl_faktur'])
        X['NIK'] = label_encoder.fit_transform(X['NIK'])
        X['Alamat'] = label_encoder.fit_transform(X['Alamat'])

        # Membagi data menjadi data pelatihan dan data pengujian
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=percentage, random_state=42)

        # Inisialisasi model Naive Bayes
        model = MultinomialNB()

        # Melatih model menggunakan data pelatihan
        model.fit(X_train, y_train)

        # Menguji model pada data pengujian
        y_pred = model.predict(X_test)

        # Mengukur kinerja model
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(conf_matrix)
        conf_matrix_html = conf_matrix.tolist()

        precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1score': f1_score,
            'support': support,
            'conf_matrix': conf_matrix_html,
        }

        # Simpan model dan label encoder menggunakan joblib
        joblib.dump(model, 'bayes_model.pkl')
        joblib.dump(label_encoder, 'label_encoder.pkl')

        # Opsional: Simpan juga objek svd (jika digunakan)
        svd = TruncatedSVD(n_components=2)
        X_pca = svd.fit_transform(X)

        joblib.dump(svd, 'svd_model.pkl')

        # Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(X_pca[y == 'ya', 0], X_pca[y == 'ya', 1], color='blue', label='Ya')
        plt.scatter(X_pca[y == 'tidak', 0], X_pca[y == 'tidak', 1], color='red', label='Tidak')
        plt.title('Hasil')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plot_filename = 'scatterplot.png'
        plt.savefig(f'static/{plot_filename}')  # Simpan plot ke dalam direktori static
        plt.close()

        return {
            'metrics': metrics,
            # 'model_filename': 'bayes_model.pkl',
            # 'label_encoder_filename': 'label_encoder.pkl',
            # 'svd_model_filename': 'svd_model.pkl',  # Opsional
            'plot_filename': plot_filename,
        }

    def predict(self, input_df):
            # Load model Naive Bayes
        model = joblib.load('bayes_model.pkl')

        # Load label encoder
        label_encoder = joblib.load('label_encoder.pkl')
            # Transformasi data input menggunakan label encoder
        input_df['tgl_faktur'] = label_encoder.transform(input_df['tgl_faktur'])
        input_df['NIK'] = label_encoder.transform(input_df['NIK'])
        input_df['Alamat'] = label_encoder.transform(input_df['Alamat'])
        prediction = model.predict(input_df)
        return prediction