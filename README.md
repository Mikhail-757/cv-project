# Учебная практика

**Выполнил:** студент группы

## Тема: Сравнительный анализ моделей детекции объектов на датасете COCO

Проект посвящен обучению и сравнению пяти архитектур нейронных сетей для задачи детекции (обнаружения) животных на изображениях. Используется подмножество датасета COCO 2017, отфильтрованное по четырем классам животных.

## Целевые классы

- bird (птица)
- cat (кошка)
- dog (собака)
- horse (лошадь)

## Исследуемые модели

| #   | Модель           | Архитектура                            | Ноутбук                          |
| --- | ---------------- | -------------------------------------- | -------------------------------- |
| 1   | **YOLOv8**       | One-stage, anchor-free                 | `notebooks/01_yolov8.ipynb`      |
| 2   | **Faster R-CNN** | Two-stage, ResNet50-FPN v2             | `notebooks/02_faster_rcnn.ipynb` |
| 3   | **SSD**          | One-stage, VGG16                       | `notebooks/03_ssd.ipynb`         |
| 4   | **RetinaNet**    | One-stage, ResNet50-FPN v2, Focal Loss | `notebooks/04_retinanet.ipynb`   |
| 5   | **DETR**         | Transformer-based, ResNet50 backbone   | `notebooks/05_detr.ipynb`        |

## Датасет

- **Источник:** COCO 2017 (train + val)
- **Фильтрация:** из 80 классов COCO выбраны 4 класса животных
- **Train:** 14 177 изображений, 27 375 аннотаций
- **Val:** 296 изображений, 556 аннотаций
- **Test:** 296 изображений, 563 аннотации

## Пайплайн каждого ноутбука

1. **Setup** — установка зависимостей
2. **Download COCO 2017** — загрузка датасета
3. **Filter Animal Classes** — фильтрация аннотаций по целевым классам
4. **Dataset Statistics** — визуализация распределения классов
5. **Pretrained Evaluation** — оценка предобученной модели (baseline)
6. **Training** — дообучение (fine-tuning) на отфильтрованном датасете
7. **Evaluation** — оценка на val/test выборках
8. **Plots** — графики потерь, метрик, примеры детекций

## Параметры обучения

| Параметр      | Значение          |
| ------------- | ----------------- |
| Эпохи         | 5                 |
| Batch size    | 8                 |
| Оптимизатор   | Adam              |
| Learning rate | 0.0005            |
| Weight decay  | 0.0005            |
| Scheduler     | Cosine            |
| Input size    | 320 (256 для SSD) |

## Метрики оценки

- **mAP@0.5** — mean Average Precision при IoU = 0.5
- **mAP@0.5:0.95** — mean Average Precision усредненная по IoU от 0.5 до 0.95
- **Precision** — точность
- **Recall** — полнота
- **F1** — гармоническое среднее Precision и Recall

## Результаты YOLOv8 (из ноутбука)

| Метрика      | Val    | Test   |
| ------------ | ------ | ------ |
| mAP@0.5      | 0.6189 | 0.6164 |
| mAP@0.5:0.95 | 0.4347 | 0.4316 |
| Precision    | 0.6855 | 0.6924 |
| Recall       | 0.5632 | 0.5519 |
| F1           | 0.6183 | 0.6142 |

## Структура проекта

```
├── configs/
│   └── default.yaml          # конфигурация обучения
├── data/
│   ├── raw/                  # исходный датасет COCO 2017
│   └── processed/            # отфильтрованные аннотации и изображения
├── results/
│   ├── plots/                # графики обучения и метрик
│   ├── logs/                 # логи обучения
│   ├── yolov8/               # результаты YOLOv8
│   ├── faster_rcnn/          # результаты Faster R-CNN
│   ├── ssd/                  # результаты SSD
│   ├── retinanet/            # результаты RetinaNet
│   └── detr/                 # результаты DETR
├── notebooks/
│   ├── 01_yolov8.ipynb        # YOLOv8
│   ├── 02_faster_rcnn.ipynb   # Faster R-CNN
│   ├── 03_ssd.ipynb           # SSD
│   ├── 04_retinanet.ipynb     # RetinaNet
│   ├── 05_detr.ipynb          # DETR
│   └── notebook_utils.py      # утилиты для ноутбуков
├── src/
│   ├── dataset/
│   │   ├── prepare_data.py    # загрузка и фильтрация COCO
│   │   ├── dataset.py         # PyTorch Dataset и DataLoader
│   │   └── augmentations.py   # аугментации
│   ├── models/
│   │   ├── yolo.py            # обертка YOLOv8
│   │   ├── faster_rcnn.py     # Faster R-CNN
│   │   ├── ssd.py             # SSD
│   │   ├── retinanet.py       # RetinaNet
│   │   └── detr.py            # DETR
│   ├── training/
│   │   ├── train.py           # цикл обучения
│   │   └── logger.py          # логирование
│   ├── evaluation/
│   │   ├── metrics.py         # вычисление метрик (mAP, precision, recall)
│   │   └── visualize.py       # визуализация результатов
│   └── utils/
│       └── utils.py           # вспомогательные функции
├── main.py                    # точка входа (CLI)
├── setup.py
└── requirements.txt
```

## Установка и запуск

```bash
pip install -r requirements.txt
pip install -e .
```

### Запуск через CLI

```bash
# Подготовка данных
python main.py --prepare-data

# Обучение одной модели
python main.py --model yolov8

# Обучение всех моделей
python main.py --model all
```

### Запуск через ноутбуки

Ноутбуки предназначены для запуска в Google Colab с GPU (Tesla T4). Каждый ноутбук является самодостаточным и включает загрузку данных, обучение и оценку.

## Зависимости

- Python 3.12+
- PyTorch >= 2.0
- torchvision >= 0.15
- Ultralytics >= 8.0
- pycocotools >= 2.0.7
- matplotlib, numpy, Pillow, OpenCV, PyYAML, tqdm, scipy
