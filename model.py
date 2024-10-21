import os
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

# Constants
BATCH_SIZE = 8
N_CLASSES = 2
IMG_SIZE = (224, 224)
N_SPLITS = 5
RANDOM_STATE = 42

# Save path for the models and results (anonymous)
save_path = './results'
os.makedirs(save_path, exist_ok=True)

# Image directory paths for each class (anonymous)
class_dirs = {
    0: "./data/class_0",
    1: "./data/class_1"
}

# Load and preprocess images
def load_images(class_dirs, img_size):
    images, labels = [], []
    for class_label, class_dir in class_dirs.items():
        for filename in os.listdir(class_dir):
            if filename.endswith('.jpeg'):
                image_path = os.path.join(class_dir, filename)
                image = Image.open(image_path).resize(img_size)
                images.append(np.array(image))
                labels.append(class_label)
                image.close()
    return np.array(images), np.array(labels)

# Build ResNet50-based model
def model_resnet50(in_shape=(224, 224, 3), n_classes=2):
    input_tensor = Input(shape=in_shape)
    base_model = ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=False)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(16, activation='relu')(x)
    output = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=input_tensor, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train and evaluate using K-Fold Cross-Validation
def run_kfold(images, labels, save_path):
    kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    for fold, (train_indices, val_indices) in enumerate(kfold.split(images, labels)):
        print(f"Fold {fold + 1}")
        train_images, val_images = images[train_indices], images[val_indices]
        train_labels = to_categorical(labels[train_indices], num_classes=N_CLASSES)
        val_labels = to_categorical(labels[val_indices], num_classes=N_CLASSES)

        model = model_resnet50(in_shape=(224, 224, 3), n_classes=N_CLASSES)

        # Add ModelCheckpoint callback
        checkpoint_path = os.path.join(save_path, f'best_model_fold_{fold + 1}.h5')
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)

        model.fit(train_images, train_labels,
                  epochs=100, batch_size=BATCH_SIZE,
                  validation_data=(val_images, val_labels),
                  callbacks=[checkpoint])

        # Load best model and evaluate
        best_model = load_model(checkpoint_path)
        loss, accuracy = best_model.evaluate(val_images, val_labels)
        print(f"Validation Loss: {loss}, Validation Accuracy: {accuracy}")

        # Save classification report
        val_pred_classes = np.argmax(best_model.predict(val_images), axis=1)
        val_true_classes = np.argmax(val_labels, axis=1)
        report = classification_report(val_true_classes, val_pred_classes, target_names=["Class 0", "Class 1"], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(os.path.join(save_path, f'classification_report_fold_{fold + 1}.csv'), index=True)

# Load and process images
images, labels = load_images(class_dirs, IMG_SIZE)

# Run K-Fold cross-validation
run_kfold(images, labels, save_path)
