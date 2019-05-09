import pandas as pd
import sys
import matplotlib.pyplot as plt

file_address = sys.argv[1]

df = pd.read_csv(file_address)
df['val_acc'].plot()
df['acc'].plot()
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Rate(%)')
plt.legend()
plt.show()

df['val_loss'].plot()
df['loss'].plot()
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('value')
plt.legend()
plt.show()


