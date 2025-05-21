'''import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# === Step 1: Load CSV ===
csv_path = "D:/train.csv"
df = pd.read_csv(csv_path)

# === Step 2: Prepare image data ===
image_dir = "D:/train_images"
img_size = 227  # AlexNet input size
X = []
y = []

for _, row in df.iterrows():
    img_path = os.path.join(image_dir, row['id_code'] + ".png")
    try:
        image = load_img(img_path, target_size=(img_size, img_size))
        image = img_to_array(image) / 255.0  # Normalize to [0, 1]
        X.append(image)
        y.append(row['diagnosis'])
    except Exception as e:
        print(f"Error loading {img_path}: {e}")

X = np.array(X)
y = to_categorical(y, num_classes=5)

# === Step 3: Split Data ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# === Step 4: Define AlexNet-like Model ===
model = Sequential()

# Layer C1
model.add(Conv2D(96, (11, 11), strides=(4, 4), padding='valid', activation='relu', input_shape=(227, 227, 3)))
# Layer S2
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
# Layer C3
model.add(Conv2D(256, (5, 5), padding='same', activation='relu'))
# Layer S4
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
# Layer C5
model.add(Conv2D(384, (3, 3), padding='same', activation='relu'))
# Layer C6
model.add(Conv2D(384, (3, 3), padding='same', activation='relu'))
# Layer C7
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
# Layer S8`
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

model.add(Flatten())
# Layer F8
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
# Layer F9
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
# Output
model.add(Dense(5, activation='softmax'))

# === Step 5: Compile ===
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === Step 6: Train ===
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

model.save("MY_MODEL.h5")'''
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pandas as pd
import os

# Load the saved model
model = load_model("MY_MODEL.h5")


image_path = "D:/train_images/12ae44be0d38.png"
img_size = 227

# Load and preprocess
img = load_img(image_path, target_size=(img_size, img_size))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)

print("Predicted Class:", predicted_class)
import matplotlib.pyplot as plt

plt.imshow(img)
plt.title(f"Predicted Class: {predicted_class}")
plt.axis('off')
plt.show()
