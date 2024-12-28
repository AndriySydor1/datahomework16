import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Завантаження моделей та історії навчання
@st.cache_resource
def load_models():
    model_conv = load_model("model_conv.keras")  # Модель Частини 1
    model_vgg16 = load_model("model_vgg16.keras")  # Модель Частини 2
    return model_conv, model_vgg16

@st.cache_resource
def load_histories():
    history_conv = np.load("history_conv.npy", allow_pickle=True).item()
    history_vgg16 = np.load("history_vgg16.npy", allow_pickle=True).item()
    return history_conv, history_vgg16

model_conv, model_vgg16 = load_models()
history_conv, history_vgg16 = load_histories()

# Інтерфейс
st.title("Класифікація зображень за допомогою нейронних мереж")
model_choice = st.selectbox("Оберіть модель", ("Модель Частини 1", "Модель Частини 2"))

uploaded_file = st.file_uploader("Завантажте зображення для класифікації", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Завантажене зображення", use_column_width=True)

    # Обробка зображення
    if model_choice == "Модель Частини 1":
        img = img.resize((28, 28))
        img_array = np.array(img.convert("L")).reshape(1, 28, 28, 1) / 255.0
        model = model_conv
    else:
        img = img.resize((64, 64))
        img_array = np.array(img).reshape(1, 64, 64, 3) / 255.0
        model = model_vgg16

    # Передбачення
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    class_names = ["T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

    st.subheader("Результати класифікації")
    st.write(f"Ймовірності для кожного класу: {predictions}")
    st.write(f"Передбачений клас: {predicted_class} ({class_names[predicted_class]})")

# Графіки навчання
st.subheader("Графіки навчання моделі")
def plot_history(history, title):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    # Точність
    ax[0].plot(history['accuracy'], label='Точність на тренуванні')
    ax[0].plot(history['val_accuracy'], label='Точність на валідації')
    ax[0].set_title(f'{title} - Точність')
    ax[0].set_xlabel('Епоха')
    ax[0].set_ylabel('Точність')
    ax[0].legend()

    # Втрата
    ax[1].plot(history['loss'], label='Втрата на тренуванні')
    ax[1].plot(history['val_loss'], label='Втрата на валідації')
    ax[1].set_title(f'{title} - Втрата')
    ax[1].set_xlabel('Епоха')
    ax[1].set_ylabel('Втрата')
    ax[1].legend()

    st.pyplot(fig)

if model_choice == "Модель Частини 1":
    plot_history(history_conv, "Модель Частини 1")
else:
    plot_history(history_vgg16, "Модель Частини 2")
    