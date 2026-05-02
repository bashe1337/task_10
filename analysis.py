import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import tensorflow as tf

def analyze_models():
    """
    Раздел 1. Сравнение моделей классификации (из практических работ 2-5).
    """
    print("Начинаем анализ моделей...")
    
    # 1. Загрузка тестового датасета (пример: Fashion MNIST)
    # ВАЖНО: Замените на ваш датасет, если он другой!
    (_, _), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_test = x_test.astype('float32') / 255.0
    x_test = np.expand_dims(x_test, -1) # (10000, 28, 28, 1)

    # 2. Пути к вашим обученным моделям (замените на реальные пути)
    # Пример ожидаемых названий
    model_paths = {
        "Model_Prac2": "models/model_prac2.keras",
        "Model_Prac3": "models/model_prac3.keras",
        "Model_Prac4": "models/model_prac4.keras",
        "Model_Prac5": "models/model_prac5.keras"
    }

    results = []
    best_f1 = -1
    best_model_name = ""
    best_model_obj = None

    # Оцениваем каждую модель
    for name, path in model_paths.items():
        if not os.path.exists(path):
            print(f"Модель {name} не найдена по пути {path}. Пропускаем.")
            continue
            
        print(f"\nЗагрузка и оценка {name}...")
        model = tf.keras.models.load_model(path)
        
        # Замер времени инференса
        start_time = time.time()
        preds_proba = model.predict(x_test, batch_size=128)
        inference_time = (time.time() - start_time) / len(x_test) * 1000 # мс на изображение
        
        preds_classes = np.argmax(preds_proba, axis=1)
        
        # 3. Подсчет метрик
        acc = accuracy_score(y_test, preds_classes)
        rec = recall_score(y_test, preds_classes, average='weighted')
        prec = precision_score(y_test, preds_classes, average='weighted', zero_division=0)
        f1 = f1_score(y_test, preds_classes, average='weighted')
        
        results.append({
            "Модель": name,
            "Accuracy": acc,
            "Recall": rec,
            "Precision": prec,
            "F1-score": f1,
            "Inference Time (ms)": inference_time
        })
        
        # 6. Построение матрицы ошибок
        cm = confusion_matrix(y_test, preds_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Матрица ошибок: {name}')
        plt.ylabel('Истинный класс')
        plt.xlabel('Предсказанный класс')
        plt.savefig(f'models/confusion_matrix_{name}.png')
        plt.close()
        
        # Запоминаем лучшую модель по F1
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model_obj = model
            
    # 4. Создание сводного DataFrame
    if not results:
        print("\nНет результатов для анализа. Убедитесь, что модели существуют.")
        return
        
    df_results = pd.DataFrame(results)
    print("\nСводная таблица метрик:")
    print(df_results.to_markdown())
    
    # 5. Визуализация метрик
    df_results.set_index("Модель")[["Accuracy", "Recall", "Precision", "F1-score"]].plot(kind='bar', figsize=(10, 6))
    plt.title('Сравнение моделей по метрикам')
    plt.ylabel('Значение метрики')
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig('models/metrics_comparison.png')
    plt.close()
    
    # Визуализация времени
    df_results.set_index("Модель")[["Inference Time (ms)"]].plot(kind='bar', color='orange', figsize=(6, 4))
    plt.title('Время инференса (мс/изображение)')
    plt.tight_layout()
    plt.savefig('models/inference_time_comparison.png')
    plt.close()
    
    # 7. Выбор и 8. Сохранение лучшей модели
    print(f"\nЛучшая модель по F1-мере: {best_model_name} (F1: {best_f1:.4f})")
    best_model_path = "models/best_classification_model.keras"
    best_model_obj.save(best_model_path)
    print(f"Лучшая модель сохранена в {best_model_path}")

if __name__ == "__main__":
    analyze_models()