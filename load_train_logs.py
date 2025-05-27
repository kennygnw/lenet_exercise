from tbparse import SummaryReader
import matplotlib.pyplot as plt
import numpy as np
# open tensorboard data
accuracy_train_sr = SummaryReader("runs/20250526_2237/Accuracy_train")
train_acc =accuracy_train_sr.scalars['value'].values  # should show your logged scalar data
accuracy_val_sr = SummaryReader("runs/20250526_2237/Accuracy_val")
validation_acc =accuracy_val_sr.scalars['value'].values  # should show your logged scalar data
epochs = [i for i in range(len(validation_acc))]
# plot
plt.plot(epochs,train_acc,label='Training')
plt.plot(epochs,validation_acc,label='Validation')
plt.title('Accuracy Values')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("documentation_imgs/Accuracy Values.png")
plt.close()
# open tensorboard data
loss_train_sr = SummaryReader("runs/20250526_2237/Loss_train")
train_loss =loss_train_sr.scalars['value'].values  # should show your logged scalar data
loss_val_sr = SummaryReader("runs/20250526_2237/Loss_val")
validation_loss =loss_val_sr.scalars['value'].values  # should show your logged scalar data
# plot
plt.plot(epochs,train_loss,'g',label='Training',)
plt.plot(epochs,validation_loss,'r',label='Validation',)
plt.title('Loss Values')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("documentation_imgs/Loss Values.png")
plt.close()
