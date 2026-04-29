import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall

warnings.filterwarnings("ignore")

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.keras.mixed_precision.set_global_policy("float32")

IMG_SIZE = (299, 299)
BATCH_SIZE = 32

def load_datasets(train_dir, test_dir):
    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )

    test_ds_raw = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )

    class_names = train_ds_raw.class_names
    val_batches = int(0.5 * len(test_ds_raw))
    val_ds = test_ds_raw.take(val_batches)
    test_ds = test_ds_raw.skip(val_batches)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds_raw.cache().shuffle(500).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names

def plot_training_history(history):
    tr_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    tr_loss = history.history['loss']
    val_loss = history.history['val_loss']
    tr_prec = history.history['precision']
    val_prec = history.history['val_precision']
    tr_recall = history.history['recall']
    val_recall = history.history['val_recall']

    epochs = np.arange(1, len(tr_acc) + 1)
    plt.figure(figsize=(20, 12))
    plt.style.use('fivethirtyeight')

    metrics = [
        ('Loss', tr_loss, val_loss),
        ('Accuracy', tr_acc, val_acc),
        ('Precision', tr_prec, val_prec),
        ('Recall', tr_recall, val_recall)
    ]

    for i, (name, train_vals, val_vals) in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        plt.plot(epochs, train_vals, 'r', label=f'Train {name}')
        plt.plot(epochs, val_vals, 'g', label=f'Val {name}')
        plt.title(f'{name} over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel(name)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

def build_model(num_classes=4, img_shape=(299, 299, 3), learning_rate=0.001):
    base_model = tf.keras.applications.Xception(
        include_top=False, weights="imagenet", input_shape=img_shape, pooling='max'
    )
    base_model.trainable = False  # Freeze pretrained layers

    model = Sequential([
        base_model,
        Flatten(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.25),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adamax(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', Precision(), Recall()]
    )
    return model

def evaluate_model(model, datasets, class_names):
    for name, ds in datasets.items():
        loss, acc, prec, rec = model.evaluate(ds, verbose=1)
        print(f"\n{name} Results:")
        print(f"  Loss: {loss:.4f}")
        print(f"  Accuracy: {acc*100:.2f}%")
        print(f"  Precision: {prec*100:.2f}%")
        print(f"  Recall: {rec*100:.2f}%")
        print("-" * 40)

def plot_confusion_matrix(model, dataset, class_labels):
    preds = model.predict(dataset)
    y_true = np.concatenate([y for x, y in dataset], axis=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = np.argmax(y_true, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_labels))

def predict_image(model, img_path, class_labels):
    img = Image.open(img_path).resize(IMG_SIZE)
    img_array = np.expand_dims(np.asarray(img) / 255.0, axis=0)
    probs = model.predict(img_array)[0]

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(2, 1, 2)
    bars = plt.barh(class_labels, probs)
    plt.xlabel('Probability', fontsize=12)
    plt.title('Prediction Probabilities')
    plt.gca().bar_label(bars, fmt='%.2f')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_ds, val_ds, test_ds, class_names = load_datasets("Training", "Testing")

    print(f"\nClasses Detected: {class_names}\n")

    model = build_model(num_classes=len(class_names))
    history = model.fit(train_ds, validation_data=val_ds, epochs=10)

    model.save("brain_tumor_xception.h5")
    print("Model saved as brain_tumor_xception.h5")

    import json
    with open("class_names.json", "w") as f:
        json.dump(class_names, f)
    print("Class names saved to class_names.json")
    
    plot_training_history(history)
    evaluate_model(model, {'Train': train_ds, 'Validation': val_ds, 'Test': test_ds}, class_names)
    plot_confusion_matrix(model, test_ds, class_names)

    predict_image(model, "Testing/meningioma/Te-meTr_0000.jpg", class_names)
    predict_image(model, "Testing/pituitary/Te-piTr_0001.jpg", class_names)