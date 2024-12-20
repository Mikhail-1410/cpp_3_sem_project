import torch
from torchvision import datasets, transforms
import os

# Параметры
DATA_DIR = 'data/mnist'  # Папка для хранения данных
BATCH_SIZE = 64           # Размер батча
DOWNLOAD = True           # Скачивать данные, если их нет

# Трансформации данных
transform = transforms.Compose([
    transforms.ToTensor(),  # Преобразовать изображения в тензоры
    transforms.Normalize((0.1307,), (0.3081,))  # Нормализация
])

# Загрузка обучающего набора данных
train_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=True,
    download=DOWNLOAD,
    transform=transform
)

# Загрузка тестового набора данных
test_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=False,
    download=DOWNLOAD,
    transform=transform
)

# Создание загрузчиков данных
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Пример использования: вывод количества образцов в обучающем и тестовом наборах
if __name__ == "__main__":
    print(f"Количество обучающих образцов: {len(train_dataset)}")
    print(f"Количество тестовых образцов: {len(test_dataset)}")

    # Пример: получение одного батча
    data_iter = iter(train_loader)
    images, labels = data_iter.next()
    print(f"Форма изображений батча: {images.shape}")
    print(f"Форма меток батча: {labels.shape}")