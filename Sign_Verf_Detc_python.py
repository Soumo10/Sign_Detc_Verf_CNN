# %% [markdown]
# # Offline Handwritten Signature Detection & Verification
# 
# ## Project Overview
# This project builds an offline signature verification system using a Convolutional Neural Network (CNN).  
# **Problem**: Binary classification to distinguish between **Genuine** and **Forged** signatures.
# 
# ## 1. Project Setup
# We start by setting up our environment, importing necessary libraries, and ensuring reproducibility.

# %%
import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns

# Set Random Seeds for Reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Check for GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU Detected: {gpus}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected. Running on CPU.")

print(f"TensorFlow Version: {tf.__version__}")

# %% [markdown]
# ## 2. Data Preparation
# The dataset is located in the `signatures` directory. It is organized by user ID (e.g., `0001` for genuine, `0001_forg` for forged).  
# We need to reorganize this into a standard format for Keras:
# ```
# data/
#   train/
#     genuine/
#     forged/
#   test/
#     genuine/
#     forged/
# ```
# The following script performs this split (80% Train, 20% Test).

# %%
SOURCE_DIR = "signatures"
DATA_DIR = "data"

def prepare_dataset(source_dir, data_dir, split_ratio=0.8):
    if os.path.exists(data_dir):
        print(f"'{data_dir}' already exists. Skipping data preparation to avoid overwriting.")
        return

    print(f"Preparing data from '{source_dir}' to '{data_dir}'...")
    
    # Create target directories
    for split in ['train', 'test']:
        for category in ['genuine', 'forged']:
            os.makedirs(os.path.join(data_dir, split, category), exist_ok=True)

    # Get all subdirectories
    subdirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    count_genuine = 0
    count_forged = 0

    for folder in subdirs:
        src_folder_path = os.path.join(source_dir, folder)
        
        # Determine category based on folder name suffix
        if folder.endswith('_forg'):
            category = 'forged'
        else:
            category = 'genuine'
            
        files = [f for f in os.listdir(src_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        random.shuffle(files)
        
        split_index = int(len(files) * split_ratio)
        train_files = files[:split_index]
        test_files = files[split_index:]
        
        # Copy files
        for f in train_files:
            shutil.copy(os.path.join(src_folder_path, f), os.path.join(data_dir, 'train', category, f))
            
        for f in test_files:
            shutil.copy(os.path.join(src_folder_path, f), os.path.join(data_dir, 'test', category, f))

        if category == 'genuine': count_genuine += len(files)
        else: count_forged += len(files)

    print(f"Data Prep Complete!")
    print(f"Total Genuine: {count_genuine}, Total Forged: {count_forged}")

# Run Data Prep
prepare_dataset(SOURCE_DIR, DATA_DIR)

# %% [markdown]
# ## 3. Data Loading
# We use `image_dataset_from_directory` to load images efficiently. We also create a validation set from the training data.

# %%
BATCH_SIZE = 16
IMG_SIZE = (224, 224)
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

print("Loading Training Set:")
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

print("\nLoading Validation Set:")
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

print("\nLoading Test Set:")
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=False # Important for evaluation metrics later
)

class_names = train_ds.class_names
print(f"\nClass Names: {class_names}") # Should be ['forged', 'genuine']

# %% [markdown]
# ### visualize Samples

# %%
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"), cmap='gray')
        plt.title(class_names[int(labels[i])])
        plt.axis("off")
plt.show()

# %% [markdown]
# ## 4. Preprocessing & Augmentation
# We build a preprocessing pipeline that:
# 1. Rescales pixel values to [0, 1] (Normalization). 
# 2. Applies data augmentation (Random Flip, Rotation, Zoom) to prevent overfitting.

# %%
# Improve performance with buffered prefetching
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache(filename='train_cache.tf').prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache(filename='val_cache.tf').prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache(filename='test_cache.tf').prefetch(buffer_size=AUTOTUNE)

# Data Augmentation Layer
data_augmentation = models.Sequential([
    layers.Rescaling(1./255),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1)
])

# Preprocessing Layer for Test (Only Rescaling)
resize_and_rescale = models.Sequential([
    layers.Rescaling(1./255)
])

# %% [markdown]
# ## 5. CNN Model Design
# We will design a custom CNN architecture suitable for binary image classification.
# - **Conv2D**: Extracts features (edges, curves).
# - **MaxPooling2D**: Reduces spatial dimensions.
# - **Dropout**: Prevents overfitting.
# - **Dense**: Classification layers.
# - **Sigmoid Output**: Returns probability between 0 and 1.

# %%
def build_cnn_model():
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        data_augmentation, # Apply augmentation on input
        
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid') # Binary output
    ])
    return model

cnn_model = build_cnn_model()
cnn_model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])
cnn_model.summary()


# %%
# --- Train Custom CNN ---
cnn_callbacks = [
    callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    callbacks.ModelCheckpoint('signature_model_cnn.keras', save_best_only=True, monitor='val_loss')
]

print("Training Custom CNN...")
cnn_history = cnn_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=cnn_callbacks
)


# %% [markdown]
# ## 5.1 Alternative: Transfer Learning (MobileNetV2)
# This model uses a pre-trained backbone (MobileNetV2) to improve accuracy.
# 

# %%
def build_mobilenet_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    return tf.keras.Model(inputs, outputs)

mobilenet_model = build_mobilenet_model()

mobilenet_model.summary()

# %%
mobilenet_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)


# %% [markdown]
# ## 6. Training
# We train the model using:
# - **Loss**: Binary Crossentropy
# - **Optimizer**: Adam
# - **Callbacks**: EarlyStopping (to stop when validation loss stops improving) and ModelCheckpoint.

# %%
callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    callbacks.ModelCheckpoint('signature_model_mobilenet.keras', save_best_only=True, monitor='val_loss')
]

EPOCHS = 5 # Adjust as needed

print("--- Phase 1: Training Top Layers ---")
mobilenet_history = mobilenet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks_list
)

print("\n--- Phase 2: Fine-Tuning ---")
mobilenet_model.trainable = True # Unfreeze all layers
mobilenet_model.compile(optimizer=optimizers.Adam(learning_rate=1e-5), 
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])

FINE_TUNE_EPOCHS = 2
total_epochs = EPOCHS + FINE_TUNE_EPOCHS

mobilenet_history_fine = mobilenet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=total_epochs,
    initial_epoch=mobilenet_history.epoch[-1],
    callbacks=callbacks_list
)

# %%
# Plot Training History (Combined)
acc = mobilenet_history.history['accuracy'] + mobilenet_history_fine.history['accuracy']
val_acc = mobilenet_history.history['val_accuracy'] + mobilenet_history_fine.history['val_accuracy']
loss = mobilenet_history.history['loss'] + mobilenet_history_fine.history['loss']
val_loss = mobilenet_history.history['val_loss'] + mobilenet_history_fine.history['val_loss']

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.plot([len(mobilenet_history.history['accuracy'])-1, len(mobilenet_history.history['accuracy'])-1], plt.ylim(), label='Start Fine Tuning', linestyle='--')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.plot([len(mobilenet_history.history['accuracy'])-1, len(mobilenet_history.history['accuracy'])-1], plt.ylim(), label='Start Fine Tuning', linestyle='--')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# %% [markdown]
# ## 7. Evaluation
# We evaluate the model on the unseen test set using Confusion Matrix and ROC Curve.

# %%
print("--- Evaluate Custom CNN ---")
loss, accuracy, precision, recall = cnn_model.evaluate(test_ds)
print(f"Custom CNN Test Accuracy: {accuracy*100:.2f}%")


# %%
# Evaluate on Test Set
loss, accuracy, precision, recall = mobilenet_model.evaluate(test_ds)
print(f"MobileNetV2 Test Accuracy: {accuracy*100:.2f}%")

# Get predictions
predictions = mobilenet_model.predict(test_ds)
y_pred = (predictions > 0.5).astype(int).flatten()

# Get true labels
y_true = np.concatenate([y for x, y in test_ds], axis=0).flatten()

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print(classification_report(y_true, y_pred, target_names=class_names))

# %%
# ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, predictions)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# %% [markdown]
# ## 8. Prediction & Verification Logic
# We create a standalone function to verify a single signature image.

# %%
def verify_signature(image_path, model, threshold=0.5):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create batch axis
    img_array = img_array / 255.0 # Rescale manually to match model input

    prediction = model.predict(img_array, verbose=0)[0][0]
    
    label = "Genuine" if prediction > threshold else "Forged"
    confidence = prediction if label == "Genuine" else 1 - prediction
    
    plt.imshow(img)
    plt.title(f"{label} ({confidence*100:.2f}% confidence)")
    plt.axis("off")
    plt.show()
    
    return label, float(prediction)

# Test with a sample from the test set
sample_image_path = os.path.join(DATA_DIR, 'test', 'genuine', os.listdir(os.path.join(DATA_DIR, 'test', 'genuine'))[0])
print(f"Testing: {sample_image_path}")
verify_signature(sample_image_path, mobilenet_model)

# %% [markdown]
# ## 9. Offline Usage
# The model is saved as `signature_model.keras`. You can load it later without retraining.

# %%
# Load the model
loaded_model = tf.keras.models.load_model('signature_model_mobilenet.keras')
print("Model loaded successfully!")

# Use loaded model
verify_signature(sample_image_path, loaded_model)


