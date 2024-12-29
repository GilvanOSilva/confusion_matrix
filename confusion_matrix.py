from tensorflow.keras import datasets, layers, models
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
import tensorflow as tf

import numpy as np

import seaborn as sns

import pandas as pd

import io

tf.__version__

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard

logdir='log'

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

train_images, test_images = train_images / 255.0, test_images / 255.0

classes=[0,1,2,3,4,5,6,7,8,9]

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x=train_images,
            y=train_labels,
            epochs=5,
            validation_data=(test_images, test_labels))

y_true = test_labels
y_pred = np.argmax(model.predict(test_images), axis=-1)

classes=[0,1,2,3,4,5,6,7,8,9]

con_mat = tf.math.confusion_matrix(labels=y_true, predictions=y_pred).numpy()
con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

con_mat_df = pd.DataFrame(con_mat_norm,
                     index = classes,
                     columns = classes)

figure = plt.figure(figsize=(8, 8))
sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

model1 = models.Sequential()
model1.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Conv2D(64, (3, 3), activation='relu'))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Conv2D(64, (3, 3), activation='relu'))

model1.add(layers.Flatten())
model1.add(layers.Dense(64, activation='relu'))
model1.add(layers.Dense(10, activation='softmax'))

model1.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

file_writer = tf.summary.create_file_writer(logdir + '/cm')

def log_confusion_matrix(epoch, logs):
  # Use the model to predict the values from the validation dataset.
  test_pred = np.argmax(model1.predict(test_images), axis=-1)

  con_mat = tf.math.confusion_matrix(labels=test_labels, predictions=test_pred).numpy()
  con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

  con_mat_df = pd.DataFrame(con_mat_norm,
                     index = classes,
                     columns = classes)

  figure = plt.figure(figsize=(8, 8))
  sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')

  buf = io.BytesIO()
  plt.savefig(buf, format='png')

  plt.close(figure)
  buf.seek(0)
  image = tf.image.decode_png(buf.getvalue(), channels=4)

  image = tf.expand_dims(image, 0)

  # Log the confusion matrix as an image summary.
  with file_writer.as_default():
    tf.summary.image("Confusion Matrix", image, step=epoch)


logdir='logs/images'

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

model1.fit(
    train_images,
    train_labels,
    epochs=5,
    verbose=0,
    callbacks=[tensorboard_callback, cm_callback],
    validation_data=(test_images, test_labels),
)

# Commented out IPython magic to ensure Python compatibility.
# Start TensorBoard.
# %tensorboard --logdir logs/images

#function for getting the confusion matrix.
def confusion_matrix_for(cls, cm):
    TP = cm[cls, cls]
    FN = cm[cls].sum() - TP
    FP = cm[:, cls].sum() - TP
    TN = cm.sum() - TP - FN - FP
    return np.array([[TP, FN], [FP, TN]])

#Extracting the values of confusion matrix for metrics values.
for cls in range(con_mat.shape[0]):
    print(f'[Class {cls} vs other Classes]')
    TP, FN, FP, TN = confusion_matrix_for(cls, con_mat).ravel()
    print(f'VP: {TP}, FN: {FN}, FP: {FP}, VN: {TN}')

def sensiblity_for(cls, cm):
    TP, FN, FP, TN = confusion_matrix_for(cls, cm).ravel()
    return TP / (TP + FN)

for cls in range(con_mat.shape[0]):
    print(f'[Class {cls} sensibility]')
    average =+ sensiblity_for(cls, con_mat)
    print(sensiblity_for(cls, con_mat))
    print()
ROC_sens = average
print(f'Average sensibility: {average/(cls + 1)}')

def specificity_for(cls, cm):
    TP, FN, FP, TN = confusion_matrix_for(cls, cm).ravel()
    return TN / (TN + FP)

for cls in range(con_mat.shape[0]):
    print(f'[Class {cls} specificity]')
    average =+ specificity_for(cls, con_mat)
    print(specificity_for(cls, con_mat))
    print()
ROC_spec = 1 - average
print(f'Average specificity: {average/(cls + 1)}')

def accuracy_for(cls, cm):
    TP, FN, FP, TN = confusion_matrix_for(cls, cm).ravel()
    return (TP + TN) / (TP + FN + FP + TN)

for cls in range(con_mat.shape[0]):
    print(f'[Class {cls} accuracy]')
    average =+ accuracy_for(cls, con_mat)
    print(accuracy_for(cls, con_mat))
    print()
print(f'Average accuracy: {average/(cls + 1)}')

def precision_for(cls, cm):
    TP, FN, FP, TN = confusion_matrix_for(cls, cm).ravel()
    return TP / (TP + FP)

for cls in range(con_mat.shape[0]):
    print(f'[Class {cls} precision]')
    average =+ precision_for(cls, con_mat)
    print(precision_for(cls, con_mat))
    print()
print(f'Average precision: {average/(cls + 1)}')

def f1_score_for(cls, cm):
    precision = precision_for(cls, cm)
    sensiblity = sensiblity_for(cls, cm)
    return 2 * (precision * sensiblity) / (precision + sensiblity)

for cls in range(con_mat.shape[0]):
    print(f'[Class {cls} f1_score]')
    average =+ f1_score_for(cls, con_mat)
    print(f1_score_for(cls, con_mat))
    print()
print(f'Average f1_score: {average/(cls + 1)}')

print('Plotting the ROC curve')
colors = ('blue', 'red', 'yellow', 'green', 'cyan', 'orange', 'purple', 'pink', 'brown', 'gray')
y_pred_prob = model1.predict(test_images)
for c in classes:
    fpr, tpr, thresholds = roc_curve(test_labels, y_pred_prob[:, c], pos_label=0)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color = colors[c], label = 'AUC = %0.2f' % roc_auc)
plt.title('Receiver Operating Characteristic')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()