## 📑 Оглавление

- [Описание](#описание)
- [Результаты](#результаты)
- [Архитектура пайплайна](#архитектура-пайплайна)
- [Структура репозитория](#структура-репозитория)
- [Инструкция по запуску](#инструкция-по-запуску)
- [Требования](#требования)
- [Контакты](#контакты)

---

## 📝 Описание

Выпускная квалификационная работа по теме **«Машинное обучение как инструмент предотвращения мошенничества в финансовых системах»**.

Разработан многоуровневый воспроизводимый пайплайн для выявления финансового мошенничества в кредитных транзакциях, адаптированный к условиям российского финансового сектора. Система объединяет детерминированные правила первого уровня, классические алгоритмы машинного обучения (XGBoost, LightGBM, CatBoost), нейросетевые модели и графовые нейронные сети (GraphSAGE, CARE-GNN), построенные на связях транзакций через общие пользовательские идентификаторы. Дополнительно реализован модуль псевдоразметки данных с использованием большой языковой модели в режиме слабого надзора.

Особое внимание уделено интерпретируемости решений (TreeSHAP, GNNExplainer), калибровке вероятностей и мониторингу дрейфа данных в соответствии с регуляторными требованиями.

---

## 📊 Результаты

Метрики на отложенной тестовой выборке (порог по максимуму F₂ на валидации):


| Модель                | F₂         | ROC-AUC    | PR-AUC     | Recall     |
| --------------------- | ---------- | ---------- | ---------- | ---------- |
| LogReg (baseline)     | 0.4359     | 0.8327     | 0.3044     | 0.5836     |
| XGBoost               | 0.4838     | 0.8579     | 0.3000     | 0.6877     |
| LightGBM              | 0.4704     | 0.8567     | 0.3411     | 0.6431     |
| CatBoost              | 0.4821     | 0.8617     | 0.3011     | 0.6691     |
| NN Classifier         | 0.4579     | 0.8455     | 0.3114     | 0.5613     |
| GraphSAGE             | 0.4976     | 0.8706     | 0.3248     | 0.7063     |
| **Stacking Ensemble** | **0.5151** | **0.8848** | **0.3619** | **0.6357** |


Экономический эффект на тестовом периоде: чистая экономия **7 530 000 у.е.**, ROI = **1.27**.

---

## 🏗️ Архитектура пайплайна

```
Входящий поток транзакций
           │
           ▼
 ┌─────────────────────┐
 │   Правила 1-го ур.  │  extreme_velocity_1h · glue_size_spike · glue_ead_spike
 │   (P95 / P99.5)     │  Precision до 0.93, Recall = 0.115
 └──────────┬──────────┘
            │ ~98% заказов передаётся дальше
            ▼
 ┌─────────────────────┐
 │    ML-модели        │  XGBoost · LightGBM · CatBoost · NN Classifier
 └──────────┬──────────┘
            │
            ▼
 ┌─────────────────────┐
 │    GNN-модели       │  GraphSAGE (индуктивный) · CARE-GNN (межтиповая аттенция)
 │    Граф транзакций  │  11 типов рёбер по общим идентификаторам
 └──────────┬──────────┘
            │
            ▼
 ┌─────────────────────┐
 │  Stacking Ensemble  │  LogReg мета-классификатор на валидационных скорах
 │  ŷ = σ(β₀ + Σβᵢpᵢ)  │  Champion-Challenger: смена при ΔF₂ > 0.001
 └──────────┬──────────┘
            │
            ▼
 ┌─────────────────────┐
 │   Постобработка     │  Изотонная калибровка · SHAP · GNNExplainer
 │   Мониторинг        │  PSI · KS-тест · JS-дивергенция → OK / MONITOR / RETRAIN
 └─────────────────────┘
```

---

## 📁 Структура репозитория

```
├── antifraud_ml_pipeline.ipynb          # Основной ноутбук — запуск полного пайплайна
│
├── antifraud_pipeline/                  # Оркестрация и библиотека
│   ├── antifraud_stages.py              # Стейджи 01–12: EDA → обучение → ансамбль → отчёт
│   ├── antifraud_lib.py                 # FeatureEngineer, ModelTrainer, DriftMonitor и др.
│   ├── antifraud_pipeline_modeltrainer.py
│   ├── antifraud_pipeline_rules_train.py
│   └── antifraud_pipeline_tail.py
│
├── antifraud_components/                # Реализации компонентов
│   ├── models/
│   │   ├── gnn.py                       # FraudGraphSAGE: архитектура, граф, обучение
│   │   ├── care_gnn.py                  # CARE-GNN: двухуровневая аттенция по типам рёбер
│   │   ├── gbm.py                       # Обёртка над градиентными бустингами
│   │   └── stacking.py                  # Стекинг: мета-классификатор
│   ├── labeling/
│   │   ├── eliza_client.py              # HTTP-клиент к LLM API
│   │   ├── llm_labeler.py               # Промптинг и парсинг ответов LLM
│   │   └── label_model.py               # Weak supervision: объединение шумных меток
│   ├── rules/
│   │   └── l1_rules.py                  # Подбор и валидация правил первого уровня
│   ├── explain/
│   │   ├── gnn_explain.py               # GNNExplainer wrapper
│   │   └── shap_utils.py                # TreeSHAP: summary plot + локальные объяснения
│   ├── monitoring/
│   │   └── drift.py                     # PSI, KS-тест, JS-дивергенция скоров
│   └── features/
│       ├── velocity.py                  # Velocity-метрики (1h / 24h / 2d окна)
│       ├── glue.py                      # Признаки склеек (glue_ead, glue_size)
│       └── temporal.py                  # Календарные и временны́е признаки
│
└── notebooks_experiments/
    ├── gnn_sage_train.ipynb             # Отдельное обучение GraphSAGE
    └── llm_labeling.ipynb               # Эксперимент с LLM-разметкой (Qwen3.5-397B)
```

---

## 🚀 Инструкция по запуску

### 1. Клонируй репозиторий

```bash
git clone https://github.com/elizavetabolotnikova/Machine-Learning-as-a-Tool-for-Fraud-Prevention-in-Financial-Systems.git
cd Machine-Learning-as-a-Tool-for-Fraud-Prevention-in-Financial-Systems
```

### 2. Создай виртуальное окружение

```bash
python3 -m venv .venv
source .venv/bin/activate
```

*(на Windows: `.\.venv\Scripts\activate`)*

### 3. Установи зависимости

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Для работы с ноутбуками:

```bash
pip install jupyter ipykernel
python -m ipykernel install --user --name antifraud-env --display-name "Antifraud Env"
```

### 4. Настрой переменные окружения

```bash
cp .env.example .env
# заполни .env: YQL_TOKEN и SOY_TOKEN
```

### 5. Запусти пайплайн

Открой `antifraud_ml_pipeline.ipynb` и выполни ячейки последовательно — пайплайн пройдёт все стейджи от загрузки данных до итогового отчёта с бизнес-метриками.

Бекап чекпойнтов моделей:

```bash
python scripts/backup_checkpoints.py save "experiment_name"
python scripts/backup_checkpoints.py list
python scripts/backup_checkpoints.py restore latest
```

---

## ⚙️ Требования

- Python 3.10+
- PyTorch 2.0+
- PyTorch Geometric
- XGBoost, LightGBM, CatBoost
- scikit-learn
- pandas, numpy, scipy
- shap
- matplotlib, seaborn
- jupyter, ipykernel

---

## 📬 Контакты

- **Почта:** [liza.bolotnikova@gmail.com](mailto:liza.bolotnikova@gmail.com)
- **Telegram:** @liza_bolotnikova

