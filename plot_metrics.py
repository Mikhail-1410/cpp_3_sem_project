import pandas as pd
import matplotlib.pyplot as plt

# Допустим, ваш файл: training_metrics.csv
# Формат: Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Acc,Train_F1,Val_F1,Train_ROC,Val_ROC
df = pd.read_csv("training_metrics.csv")

plt.figure(figsize=(10,6))
# Кривая точности
plt.plot(df["Epoch"], df["Train_Acc"], label="Train Accuracy")
plt.plot(df["Epoch"], df["Val_Acc"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epoch")
plt.legend()
plt.grid(True)
plt.savefig("accuracy_plot.png")

plt.figure(figsize=(10,6))
# Кривая потерь
plt.plot(df["Epoch"], df["Train_Loss"], label="Train Loss")
plt.plot(df["Epoch"], df["Val_Loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")
plt.legend()
plt.grid(True)
plt.savefig("loss_plot.png")

plt.show()