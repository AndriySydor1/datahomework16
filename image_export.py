import os
from tensorflow.keras.datasets import fashion_mnist
from PIL import Image

# Завантаження датасету
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Мапа назв класів із заміною "/" на "-"
class_names = [
    "T-shirt-Top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle-Boot"
]

# Функція експорту зображень
def export_images(data, labels, output_dir, dataset_type):
    for i, (img, label) in enumerate(zip(data, labels)):
        class_name = class_names[label]
        class_dir = os.path.join(output_dir, dataset_type, class_name)
        
        # Створюємо каталог, якщо він не існує
        os.makedirs(class_dir, exist_ok=True)
        
        img_name = f"{dataset_type}_{class_name}_{i + 1}.png"
        img_path = os.path.join(class_dir, img_name)
        
        # Збереження зображення
        Image.fromarray(img).save(img_path)

# Вихідний каталог для зображень
output_directory = "fashion_mnist_images"

# Експорт навчальних даних
export_images(X_train, y_train, output_directory, "train")

# Експорт тестових даних
export_images(X_test, y_test, output_directory, "test")

print("Експорт зображень завершено!")
