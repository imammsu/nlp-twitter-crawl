import streamlit as st
import numpy as np
import pandas as pd
import joblib, os
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fifa_value_model.pkl")

def load_prediction_model(model_file):
    loaded_model = joblib.load(open(model_file, "rb"))
    return loaded_model

RATING_BUCKETS = pd.DataFrame({
    "Rentang Overall": ["<60", "60-64", "65-69", "70-74", "75-79", "80-84", "85+"],
    "Median Nilai Pasar (juta $)": [0.25, 0.68, 1.40, 2.90, 9.50, 26.75, 64.50],
    "Jumlah Pemain": [6359, 5258, 3744, 2236, 1229, 420, 91]
})

MODEL_R2 = 0.882   
MODEL_MAE = 0.71   

SAMPLE_POINTS = [
    [67.0,2.2],[70.0,1.4],[61.0,0.75],[65.0,0.85],[59.0,0.52],[57.0,0.4],[58.0,0.48],
    [61.0,0.72],[67.0,2.5],[62.0,0.28],[69.0,1.4],[76.0,15.5],[53.0,0.19],[55.0,0.28],[65.0,1.1],
    [62.0,0.4],[76.0,17.0],[56.0,0.32],[59.0,0.5],[64.0,1.2],[63.0,0.23],[60.0,0.52],[62.0,0.52],
    [59.0,0.55],[50.0,0.06],[60.0,0.52],[61.0,0.68],[51.0,0.15],[66.0,0.55],[68.0,1.2],[59.0,0.19],
    [57.0,0.35],[69.0,1.0],[65.0,1.6],[65.0,0.7],[61.0,0.55],[69.0,3.0],[52.0,0.18],[66.0,1.8],
    [62.0,1.0],[53.0,0.24],[55.0,0.28],[61.0,0.55],[56.0,0.45],[68.0,0.52],[53.0,0.1],[52.0,0.09],
    [66.0,0.82],[66.0,0.85],[53.0,0.21],[53.0,0.22],[64.0,0.95],[61.0,0.35],[63.0,0.6],[61.0,0.72],
    [63.0,1.0],[59.0,0.22],[54.0,0.25],[80.0,22.5],[49.0,0.07],[62.0,0.95],[62.0,0.42],[67.0,2.3],
    [62.0,0.4],[63.0,1.0],[70.0,2.2],[66.0,0.65],[62.0,0.9],[73.0,3.0],[76.0,10.0],[60.0,0.5],
    [72.0,1.9],[72.0,5.0],[55.0,0.38],[71.0,2.0],[52.0,0.21],[60.0,0.3],[64.0,1.1],[67.0,1.0],
    [53.0,0.21],[63.0,1.1],[63.0,0.7],[64.0,0.85],[72.0,0.4],[64.0,0.5],[64.0,0.9],[57.0,0.35],
    [72.0,1.6],[53.0,0.24],[52.0,0.17],[58.0,0.45],[48.0,0.06],[59.0,0.55],[66.0,1.8],[51.0,0.15],
    [75.0,0.38],[65.0,1.6],[57.0,0.62],[49.0,0.08],[60.0,0.35],[61.0,0.35],[58.0,0.42],[59.0,0.45],
    [73.0,7.0],[56.0,0.3],[68.0,1.2],[52.0,0.09],[60.0,0.52],[64.0,1.3],[57.0,0.38],[54.0,0.24],
    [75.0,6.5],[60.0,0.6],[69.0,3.3],[53.0,0.22],[66.0,2.0],[66.0,0.72],[62.0,0.95],[57.0,0.32],
    [71.0,3.5],[73.0,6.5],[64.0,1.2],[53.0,0.22],[59.0,0.45],[74.0,8.0],[64.0,1.2],[69.0,2.8],
    [62.0,0.57],[57.0,0.19],[55.0,0.28],[65.0,0.88],[55.0,0.35],[55.0,0.28],[57.0,0.5],[66.0,0.45],
    [61.0,0.16],[63.0,0.57],[63.0,1.1],[52.0,0.08],[64.0,0.72],[57.0,0.32],[74.0,5.0],[52.0,0.19],
    [53.0,0.28],[61.0,0.62],[60.0,0.57],[54.0,0.21],[60.0,0.32],[68.0,0.85],[64.0,0.6]
]

def get_bucket(overall):
    if overall < 60: return "<60"
    elif overall < 65: return "60-64"
    elif overall < 70: return "65-69"
    elif overall < 75: return "70-74"
    elif overall < 80: return "75-79"
    elif overall < 85: return "80-84"
    else: return "85+"

def main():
    st.title("Prediksi Nilai Pasar Pemain Sepak Bola")

    html_templ = """
    <div style="background-color:#726E6D;padding:10px;">
    <h3 style="color:white">Prediksi Nilai Pasar Pemain Menggunakan Multiple Linear Regression</h3>
    </div>
    """
    st.markdown(html_templ, unsafe_allow_html=True)

    activity = ["Prediksi Nilai Pasar Pemain", "Apa itu Regresi?"]
    choice = st.sidebar.selectbox("Menu", activity)

    if choice == 'Prediksi Nilai Pasar Pemain':

        st.subheader("Masukkan Data Pemain")

        player_name = st.text_input("Nama pemain (opsional)", placeholder="e.g., Ruud Van Nistelrooy")
        st.caption("Nama hanya sebagai label, tidak memengaruhi hasil prediksi.")

        col1, col2 = st.columns(2)
        with col1:
            overall = st.slider("Overall Rating (kemampuan saat ini)", 44, 91, 70)
            age = st.slider("Usia pemain", 15, 44, 24)
        with col2:
            stats = st.slider("Total Stats Score (jumlah seluruh statistik)", 745, 2324, 1535)

        if st.button("Proses"):

            regressor = load_prediction_model(MODEL_PATH)

            input_data = np.array([[overall, age, stats]])
            pred_log = regressor.predict(input_data)
            pred_value = 10 ** pred_log[0][0]

            label = player_name.strip() if player_name.strip() else "Pemain ini"
            st.info(f"Estimasi nilai pasar **{label}**: **${pred_value:,.2f} juta**")
            st.caption(f"Rentang wajar: sekitar ±${MODEL_MAE:.2f} juta")

            bucket = get_bucket(overall)
            ref_row = RATING_BUCKETS[RATING_BUCKETS["Rentang Overall"] == bucket].iloc[0]
            median_kelompok = ref_row["Median Nilai Pasar (juta $)"]
            selisih = pred_value - median_kelompok

            col1, col2 = st.columns(2)
            col1.metric(f"Median kelompok (Overall {bucket})", f"${median_kelompok:.2f} juta")
            col2.metric("Selisih dari median", f"${selisih:+.2f} juta")

            if age <= 23:
                usia_tag = "usia berkembang"
            elif age <= 29:
                usia_tag = "usia puncak"
            else:
                usia_tag = "melewati puncak"
            st.caption(f"Usia {age} tahun — {usia_tag}")

            sample_df = pd.DataFrame(SAMPLE_POINTS, columns=["Overall_Rating", "Value"])

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(sample_df["Overall_Rating"], sample_df["Value"],
                       color="gray", alpha=0.5, label="Pemain lain (sampel data)")
            ax.scatter([overall], [pred_value],
                       color="orange", s=120, edgecolor="black", zorder=5, label="Prediksi Anda")
            if player_name.strip():
                ax.annotate(player_name.strip(), (overall, pred_value),
                            textcoords="offset points", xytext=(8, 8), fontsize=9)

            ax.set_xlabel("Overall Rating")
            ax.set_ylabel("Nilai Pasar (juta $)")
            ax.set_title("Overall Rating vs Nilai Pasar")
            ax.legend()
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)
    else:
        st.subheader("Apa itu Regresi Linear?")
        st.write("""
        Regresi linear adalah metode statistik untuk memprediksi suatu nilai (Y) 
        berdasarkan satu atau lebih variabel lain (X), dengan mengasumsikan hubungan berbentuk garis lurus.

        Model ini memakai **tiga variabel independen**:
        - **Overall Rating** — kemampuan pemain saat ini
        - **Age** — usia pemain
        - **Total Stats Score** — jumlah seluruh statistik pemain

        untuk memprediksi **Value Per M$**, yaitu nilai pasar pemain dalam juta dolar.
        """)

        st.subheader("Kenapa Pakai Transformasi Logaritma?")
        st.write("""
        Nilai pasar pemain sangat timpang: kebanyakan pemain bernilai di bawah 1 juta dolar, 
        sementara beberapa bintang bernilai lebih dari 100 juta dolar. Kalau dilatih langsung 
        dari angka aslinya, R² model hanya sekitar 0,34 dan bisa menghasilkan prediksi negatif.

        Setelah target diubah menjadi log10(Value), hubungannya menjadi jauh lebih lurus, 
        dan R² model naik menjadi sekitar 0,96 (skala log) atau 0,88 (setelah dikembalikan ke skala dolar). 
        Hasil akhir yang ditampilkan ke pengguna selalu dikembalikan ke skala dolar dengan `10 ** hasil_prediksi`.
        """)

        st.subheader("Performa Model")
        st.write(f"- R² (skala asli, juta $): **{MODEL_R2}**")
        st.write(f"- MAE (skala asli, juta $): **±{MODEL_MAE}**")
        st.caption("Koefisien Total_Stats Score sangat kecil (≈0,0001), karena informasinya tumpang tindih dengan Overall_Rating (multikolinearitas).")


if __name__ == '__main__':
    main()