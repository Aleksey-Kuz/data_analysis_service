import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

# Установите тип задачи: 'classification' или 'regression'
task_type = 'regression'  # или 'regression'

# Метрики и значения
if task_type == 'classification':
    metrics = {
        'Accuracy': 0.95,
        'Precision': 0.93,
        'Recall': 0.92,
        'F1-score': 0.925,
        'AUC': 0.96
    }
elif task_type == 'regression':
    metrics = {
        'R^2': 0.91,
        'MAE': 1.8,
        'MSE': 4.2,
        'RMSE': 2.05,
        'MAPE': 5.1
    }
else:
    raise ValueError("Unsupported task type. Use 'classification' or 'regression'.")

# Подготовка графика
fig, ax = plt.subplots(figsize=(8, 6))
metric_names = list(metrics.keys())
metric_values = list(metrics.values())

bars = ax.bar(metric_names, metric_values, color='skyblue')
ax.set_ylim([0, max(metric_values) * 1.2])
ax.set_ylabel("Metric Value")
ax.set_title("")

# Добавление значений над столбиками
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')

# Заголовок
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
model_name = "CatBoostModel"
title = f"{model_name} - {task_type.capitalize()} \n{now}"
plt.suptitle(title, fontsize=14)

# Сохранение в PDF
pdf_filename = "model_metrics_report.pdf"
with PdfPages(pdf_filename) as pdf:
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

print(f"PDF-файл успешно сохранён как '{pdf_filename}'")
