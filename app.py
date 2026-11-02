"""
Dashboard Streamlit - Klasifikasi Objek Laut (OceanicLife) Menggunakan CNN
Jalankan dengan:  streamlit run app.py

Struktur file yang diharapkan (semua bisa diubah di sidebar):
    app.py
    model_emotion_cnn.h5 (atau .keras)   -> model.save(...) dari notebook
    history.json (atau history.csv)       -> riwayat training dari notebook
    dataset/
        train/<nama_kelas>/*.jpg
        test/<nama_kelas>/*.jpg
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# ----------------------------------------------------------------------------
# Konfigurasi halaman
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard CNN - Klasifikasi Objek Laut",
    page_icon="🌊",
    layout="wide",
)

# Urutan kelas HARUS sama dengan urutan saat training (output/class_names.json),
# bukan urutan abjad folder.
def _load_class_names():
    for p in ("output/class_names.json", "class_names.json"):
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    return ["other", "fish", "seal", "coral", "crab", "shark", "turtle",
            "octopus", "starfish", "jellyfish"]


DEFAULT_CLASSES = _load_class_names()
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
EMOJI = {
    "other": "🌊", "fish": "🐟", "seal": "🦭", "coral": "🪸", "crab": "🦀",
    "shark": "🦈", "turtle": "🐢", "octopus": "🐙", "starfish": "⭐", "jellyfish": "🪼",
}


# ----------------------------------------------------------------------------
# Fungsi bantu
# ----------------------------------------------------------------------------
def list_classes(directory):
    if directory and os.path.isdir(directory):
        return sorted(
            d for d in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, d))
        )
    return []


def count_per_class(directory, classes):
    counts = {}
    for c in classes:
        p = os.path.join(directory, c)
        counts[c] = (
            len([f for f in os.listdir(p) if f.lower().endswith(IMG_EXT)])
            if os.path.isdir(p) else 0
        )
    return counts


@st.cache_resource(show_spinner="Memuat model...")
def load_model(path):
    from tensorflow.keras.models import load_model as _load
    return _load(path)


def load_history(path, uploaded=None):
    try:
        if uploaded is not None:
            if uploaded.name.endswith(".json"):
                return pd.DataFrame(json.load(uploaded))
            return pd.read_csv(uploaded)
        if path and os.path.exists(path):
            if path.endswith(".json"):
                with open(path) as f:
                    return pd.DataFrame(json.load(f))
            return pd.read_csv(path)
    except Exception as e:  # noqa: BLE001
        st.warning(f"Gagal membaca history: {e}")
    return None


def preprocess(img, size, channels, rescale):
    img = ImageOps.exif_transpose(img)
    img = img.convert("L" if channels == 1 else "RGB")
    img = img.resize(size)
    arr = np.asarray(img, dtype="float32")
    if rescale:
        arr = arr / 255.0
    if channels == 1:
        arr = arr[..., np.newaxis]
    return arr


@st.cache_data(show_spinner="Mengevaluasi model pada data uji...")
def evaluate_on_test(_model, model_key, test_dir, classes, size, channels, rescale):
    X, y = [], []
    for idx, c in enumerate(classes):
        p = os.path.join(test_dir, c)
        if not os.path.isdir(p):
            continue
        for fn in os.listdir(p):
            if fn.lower().endswith(IMG_EXT):
                try:
                    X.append(preprocess(Image.open(os.path.join(p, fn)), size, channels, rescale))
                    y.append(idx)
                except Exception:  # noqa: BLE001
                    continue
    if not X:
        return None, None
    X = np.stack(X)
    probs = _model.predict(X, batch_size=64, verbose=0)
    return np.array(y), probs.argmax(axis=1)


def fig_bar(names, values, title, color="#4C78A8", ylabel="Jumlah gambar"):
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, values, color=color)
    ax.bar_label(bars, padding=2, fontsize=9)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Sidebar - pengaturan
# ----------------------------------------------------------------------------
st.sidebar.header("⚙️ Pengaturan")
model_path = st.sidebar.text_input("Path model (.h5 / .keras)", "model_emotion_cnn.h5")
history_path = st.sidebar.text_input("Path history (.json / .csv)", "history.json")
history_upload = st.sidebar.file_uploader("…atau upload history", type=["json", "csv"])
train_dir = st.sidebar.text_input("Folder data latih", "dataset/train")
test_dir = st.sidebar.text_input("Folder data uji", "dataset/test")
rescale = st.sidebar.checkbox("Normalisasi piksel (÷255)", value=True,
                              help="Aktifkan jika saat training memakai rescale=1/255.")
st.sidebar.caption(
    "Pastikan preprocessing di sini sama persis dengan saat training di notebook."
)

# Kelas
classes = list(DEFAULT_CLASSES)  # urutan sesuai training

# Model
model = None
if os.path.exists(model_path):
    try:
        model = load_model(model_path)
    except Exception as e:  # noqa: BLE001
        st.sidebar.error(f"Gagal memuat model: {e}")
else:
    st.sidebar.warning("File model belum ditemukan.")

img_size, channels = (48, 48), 1
if model is not None:
    shp = model.input_shape  # (None, H, W, C)
    img_size, channels = (int(shp[2]), int(shp[1])), int(shp[3])
    if len(classes) != model.output_shape[-1]:
        st.sidebar.warning(
            f"Jumlah kelas folder ({len(classes)}) ≠ output model "
            f"({model.output_shape[-1]}). Memakai label default."
        )
        classes = DEFAULT_CLASSES[: model.output_shape[-1]]

# ----------------------------------------------------------------------------
# 1. Judul dan deskripsi
# ----------------------------------------------------------------------------
st.title("🌊 Klasifikasi Objek Laut Menggunakan CNN")
st.markdown(
    """
Dashboard ini menampilkan hasil eksperimen **Convolutional Neural Network (CNN)**
untuk mengenali objek laut (ikan, hiu, penyu, dll.) dari gambar. Model dilatih pada
dataset OceanicLife, lalu dievaluasi pada data uji yang terpisah. Di bawah ini tersedia
ringkasan dataset, grafik pelatihan, confusion matrix, metrik evaluasi, serta fitur
**uji prediksi langsung** dengan mengunggah gambar baru.
"""
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Dataset", "📈 Pelatihan", "🧪 Evaluasi", "🔍 Uji Prediksi"]
)

# ----------------------------------------------------------------------------
# 2-3. Ringkasan & distribusi dataset
# ----------------------------------------------------------------------------
with tab1:
    st.subheader("Ringkasan Dataset")
    train_counts = count_per_class(train_dir, classes) if os.path.isdir(train_dir) else {}
    test_counts = count_per_class(test_dir, classes) if os.path.isdir(test_dir) else {}
    n_train, n_test = sum(train_counts.values()), sum(test_counts.values())
    total = n_train + n_test

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah kelas", len(classes))
    c2.metric("Total gambar", f"{total:,}" if total else "-")
    c3.metric("Data latih", f"{n_train:,}" if n_train else "-")
    c4.metric("Data uji", f"{n_test:,}" if n_test else "-")

    if total:
        pct_tr, pct_te = n_train / total * 100, n_test / total * 100
        st.markdown(f"**Proporsi data latih : uji = {pct_tr:.1f}% : {pct_te:.1f}%**")
        st.write("Kelas:", ", ".join(f"{EMOJI.get(c, '')} {c}" for c in classes))

        st.subheader("Distribusi Data per Kelas")
        df_dist = pd.DataFrame({
            "Kelas": classes,
            "Latih": [train_counts.get(c, 0) for c in classes],
            "Uji": [test_counts.get(c, 0) for c in classes],
        })
        df_dist["Total"] = df_dist["Latih"] + df_dist["Uji"]

        g1, g2 = st.columns(2)
        with g1:
            st.pyplot(fig_bar(df_dist["Kelas"], df_dist["Latih"], "Distribusi Data Latih"))
        with g2:
            st.pyplot(fig_bar(df_dist["Kelas"], df_dist["Uji"], "Distribusi Data Uji", "#F58518"))
        st.dataframe(df_dist, use_container_width=True, hide_index=True)

        # Analisis
        base = df_dist["Latih"] if n_train else df_dist["Total"]
        mx, mn = base.idxmax(), base.idxmin()
        ratio = base.max() / max(base.min(), 1)
        kesimpulan = (
            "Distribusi relatif **seimbang**." if ratio < 1.5 else
            "Terdapat **ketidakseimbangan sedang**." if ratio < 3 else
            "Terdapat **ketidakseimbangan kelas yang signifikan**."
        )
        st.info(
            f"**Analisis:** Kelas terbanyak adalah **{df_dist.loc[mx, 'Kelas']}** "
            f"({base[mx]:,} gambar) dan paling sedikit **{df_dist.loc[mn, 'Kelas']}** "
            f"({base[mn]:,} gambar), rasio ≈ {ratio:.1f}×. {kesimpulan} "
            "Pada kelas minoritas model cenderung memiliki recall lebih rendah karena "
            "kurangnya contoh; solusinya dapat berupa augmentasi, oversampling, "
            "atau pemberian class weight."
            + (f" Proporsi latih:uji {pct_tr:.0f}:{pct_te:.0f} memberi data uji yang "
               "cukup untuk evaluasi yang andal." if pct_te >= 10 else
               " Data uji relatif sedikit, sehingga hasil evaluasi kurang stabil.")
        )
    else:
        st.warning("Folder dataset tidak ditemukan. Atur path di sidebar "
                   "(struktur: folder/<nama_kelas>/gambar).")

# ----------------------------------------------------------------------------
# 4. Grafik pelatihan
# ----------------------------------------------------------------------------
with tab2:
    st.subheader("Hasil Pelatihan (Akurasi & Loss)")
    hist = load_history(history_path, history_upload)
    if hist is not None:
        hist.columns = [c.lower() for c in hist.columns]
        acc_k = "accuracy" if "accuracy" in hist else ("acc" if "acc" in hist else None)
        vacc_k = "val_accuracy" if "val_accuracy" in hist else ("val_acc" if "val_acc" in hist else None)
        epochs = np.arange(1, len(hist) + 1)

        g1, g2 = st.columns(2)
        if acc_k:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(epochs, hist[acc_k], label="Train", marker="o", ms=3)
            if vacc_k:
                ax.plot(epochs, hist[vacc_k], label="Validasi", marker="o", ms=3)
            ax.set(title="Akurasi per Epoch", xlabel="Epoch", ylabel="Akurasi")
            ax.legend(); ax.grid(alpha=.3)
            g1.pyplot(fig)
        if "loss" in hist:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(epochs, hist["loss"], label="Train", marker="o", ms=3)
            if "val_loss" in hist:
                ax.plot(epochs, hist["val_loss"], label="Validasi", marker="o", ms=3)
            ax.set(title="Loss per Epoch", xlabel="Epoch", ylabel="Loss")
            ax.legend(); ax.grid(alpha=.3)
            g2.pyplot(fig)

        # Analisis
        teks = [f"Model dilatih selama **{len(hist)} epoch**."]
        if acc_k:
            teks.append(f"Akurasi train akhir **{hist[acc_k].iloc[-1]:.2%}**.")
        if vacc_k:
            be = int(hist[vacc_k].idxmax())
            gap = hist[acc_k].iloc[-1] - hist[vacc_k].iloc[-1]
            teks.append(
                f"Akurasi validasi terbaik **{hist[vacc_k].max():.2%}** pada epoch "
                f"**{be + 1}**; akhir **{hist[vacc_k].iloc[-1]:.2%}**."
            )
            if gap > 0.15:
                teks.append(f"Selisih train–validasi {gap:.1%} → indikasi **overfitting**. "
                            "Pertimbangkan dropout, augmentasi, regularisasi, atau early stopping.")
            elif gap < 0.03 and hist[vacc_k].iloc[-1] < 0.6:
                teks.append("Kedua kurva rendah dan berdekatan → indikasi **underfitting**; "
                            "model perlu lebih dalam/lebih lama dilatih.")
            else:
                teks.append(f"Selisih train–validasi {gap:.1%} → generalisasi **cukup baik**.")
        if "val_loss" in hist:
            bl = int(hist["val_loss"].idxmin())
            teks.append(f"Val loss terendah **{hist['val_loss'].min():.4f}** pada epoch {bl + 1}; "
                        "jika setelahnya naik sementara train loss turun, itu tanda overfitting.")
        st.info("**Analisis:** " + " ".join(teks))
        with st.expander("Lihat tabel history"):
            st.dataframe(hist, use_container_width=True)
    else:
        st.warning(
            "File history tidak ditemukan. Simpan dari notebook dengan:\n\n"
            "```python\nimport json\n"
            "json.dump(history.history, open('history.json','w'))\n```\n"
            "lalu letakkan di folder yang sama atau upload di sidebar."
        )

# ----------------------------------------------------------------------------
# 5-6. Confusion matrix & metrik evaluasi
# ----------------------------------------------------------------------------
with tab3:
    st.subheader("Evaluasi Model pada Data Uji")
    if model is None:
        st.warning("Model belum dimuat. Periksa path model di sidebar.")
    elif not os.path.isdir(test_dir):
        st.warning("Folder data uji tidak ditemukan.")
    else:
        y_true, y_pred = evaluate_on_test(
            model, os.path.getmtime(model_path), test_dir,
            tuple(classes), img_size, channels, rescale,
        )
        if y_true is None:
            st.warning("Tidak ada gambar pada folder data uji.")
        else:
            labels = list(range(len(classes)))
            cm = confusion_matrix(y_true, y_pred, labels=labels)

            # Confusion matrix
            st.markdown("#### Confusion Matrix")
            cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            g1, g2 = st.columns(2)
            for col, mat, ttl, fmt in (
                (g1, cm, "Confusion Matrix (jumlah)", "d"),
                (g2, cm_norm, "Confusion Matrix (dinormalisasi per baris)", ".2f"),
            ):
                fig, ax = plt.subplots(figsize=(6.5, 5.5))
                im = ax.imshow(mat, cmap="Blues")
                ax.set_xticks(labels); ax.set_yticks(labels)
                ax.set_xticklabels(classes, rotation=45, ha="right")
                ax.set_yticklabels(classes)
                ax.set(xlabel="Prediksi", ylabel="Label sebenarnya", title=ttl)
                thr = mat.max() / 2
                for i in labels:
                    for j in labels:
                        ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center",
                                fontsize=8, color="white" if mat[i, j] > thr else "black")
                fig.colorbar(im, ax=ax, fraction=0.046)
                fig.tight_layout()
                col.pyplot(fig)

            recall_cls = np.diag(cm_norm)
            best, worst = int(recall_cls.argmax()), int(recall_cls.argmin())
            off = cm.copy(); np.fill_diagonal(off, 0)
            ti, tj = np.unravel_index(off.argmax(), off.shape)
            st.info(
                f"**Analisis:** Diagonal utama menunjukkan prediksi benar. Kelas yang paling "
                f"baik dikenali adalah **{classes[best]}** (recall {recall_cls[best]:.1%}), "
                f"sedangkan yang paling sulit adalah **{classes[worst]}** "
                f"(recall {recall_cls[worst]:.1%}). Kesalahan terbanyak: gambar "
                f"**{classes[ti]}** diprediksi sebagai **{classes[tj]}** sebanyak "
                f"{off[ti, tj]} kali — kemungkinan karena kedua kelas mirip secara "
                "visual atau jumlah data salah satu kelas kurang."
            )

            # Metrik
            st.markdown("#### Metrik Evaluasi")
            acc = accuracy_score(y_true, y_pred)
            p_m, r_m, f_m, _ = precision_recall_fscore_support(
                y_true, y_pred, average="macro", zero_division=0)
            p_w, r_w, f_w, _ = precision_recall_fscore_support(
                y_true, y_pred, average="weighted", zero_division=0)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{acc:.2%}")
            m2.metric("Precision (macro)", f"{p_m:.2%}")
            m3.metric("Recall (macro)", f"{r_m:.2%}")
            m4.metric("F1-score (macro)", f"{f_m:.2%}")
            st.caption(f"Weighted avg → Precision {p_w:.2%} | Recall {r_w:.2%} | F1 {f_w:.2%}")

            rep = classification_report(
                y_true, y_pred, labels=labels, target_names=classes,
                output_dict=True, zero_division=0)
            df_rep = pd.DataFrame(rep).T.drop(index=["accuracy"], errors="ignore")
            df_rep["support"] = df_rep["support"].astype(int)
            st.dataframe(
                df_rep.style.format({"precision": "{:.2%}", "recall": "{:.2%}", "f1-score": "{:.2%}"}),
                use_container_width=True,
            )

            per = df_rep.loc[classes]
            fig, ax = plt.subplots(figsize=(9, 4))
            x = np.arange(len(classes)); w = 0.27
            ax.bar(x - w, per["precision"], w, label="Precision")
            ax.bar(x, per["recall"], w, label="Recall")
            ax.bar(x + w, per["f1-score"], w, label="F1-score")
            ax.set_xticks(x); ax.set_xticklabels(classes, rotation=30, ha="right")
            ax.set(ylim=(0, 1.05), title="Metrik per Kelas")
            ax.legend(); ax.grid(axis="y", alpha=.3)
            st.pyplot(fig)

            bf = per["f1-score"].idxmax(); wf = per["f1-score"].idxmin()
            gap_mw = abs(f_m - f_w)
            st.info(
                f"**Analisis:** Accuracy keseluruhan **{acc:.2%}**. F1 terbaik pada kelas "
                f"**{bf}** ({per.loc[bf, 'f1-score']:.2%}) dan terendah pada **{wf}** "
                f"({per.loc[wf, 'f1-score']:.2%}). "
                + ("Selisih macro dan weighted F1 kecil, artinya performa merata antar kelas."
                   if gap_mw < 0.03 else
                   f"Selisih macro vs weighted F1 {gap_mw:.1%} menandakan performa tidak merata "
                   "— kelas dengan data banyak mendominasi skor.")
                + " Precision rendah berarti banyak salah tebak ke kelas tersebut; recall rendah "
                  "berarti banyak sampel kelas tersebut terlewat."
            )

# ----------------------------------------------------------------------------
# 7. Upload gambar & prediksi real-time
# ----------------------------------------------------------------------------
with tab4:
    st.subheader("Uji Prediksi Gambar Baru")
    if model is None:
        st.warning("Model belum dimuat. Periksa path model di sidebar.")
    else:
        mode = st.radio("Sumber gambar", ["Upload file", "Kamera"], horizontal=True)
        src = (st.file_uploader("Unggah gambar objek laut", type=["jpg", "jpeg", "png", "bmp", "webp"])
               if mode == "Upload file" else st.camera_input("Ambil foto"))

        if src is not None:
            img = Image.open(src)
            arr = preprocess(img, img_size, channels, rescale)
            probs = model.predict(arr[np.newaxis], verbose=0)[0]
            order = np.argsort(probs)[::-1]
            top = int(order[0])

            c1, c2 = st.columns([1, 1.4])
            with c1:
                st.image(img, caption="Gambar input", use_container_width=True)
                st.image(
                    np.clip(arr.squeeze() if channels == 1 else arr, 0, 1 if rescale else 255)
                    / (1 if rescale else 255),
                    caption=f"Setelah preprocessing ({img_size[0]}×{img_size[1]}, "
                            f"{'grayscale' if channels == 1 else 'RGB'})",
                    width=150,
                )
            with c2:
                st.success(f"### Prediksi: {EMOJI.get(classes[top], '')} **{classes[top].upper()}**")
                st.metric("Tingkat keyakinan", f"{probs[top]:.2%}")
                fig, ax = plt.subplots(figsize=(6, 3.8))
                colors = ["#2CA02C" if i == top else "#9ecae1" for i in range(len(classes))]
                bars = ax.barh(classes, probs, color=colors)
                ax.bar_label(bars, labels=[f"{p:.1%}" for p in probs], padding=3, fontsize=9)
                ax.set_xlim(0, 1.15); ax.invert_yaxis()
                ax.set(title="Probabilitas per Kelas", xlabel="Probabilitas")
                ax.spines[["top", "right"]].set_visible(False)
                fig.tight_layout()
                st.pyplot(fig)

            second = int(order[1])
            margin = probs[top] - probs[second]
            yakin = ("sangat yakin" if probs[top] >= 0.8 else
                     "cukup yakin" if probs[top] >= 0.5 else "kurang yakin")
            st.info(
                f"**Analisis:** Model memprediksi **{classes[top]}** dengan probabilitas "
                f"{probs[top]:.1%} ({yakin}). Kandidat kedua adalah **{classes[second]}** "
                f"({probs[second]:.1%}), selisih {margin:.1%}. "
                + ("Selisih tipis menunjukkan gambar ambigu atau mirip dua kelas."
                   if margin < 0.2 else "Selisih besar menunjukkan prediksi dominan dan stabil.")
                + " Untuk hasil terbaik gunakan foto dengan pencahayaan cukup dan objek "
                  "yang jelas serta memenuhi frame."
            )
        else:
            st.caption("Unggah gambar untuk melihat hasil prediksi secara langsung.")

st.divider()
st.caption("Dashboard klasifikasi objek laut dengan CNN • dibangun menggunakan Streamlit")