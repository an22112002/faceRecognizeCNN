# with open("D:/SavePythonFile/TestTensorFlow/image/label.txt", "r") as file:
    #     l = file.readline()
    #     data = l.split(',')
    #     data.remove('')
    #     for i in range(0, len(data)):
    #         data[i] = data[i].split("|")
    #         data[i][1] = int(data[i][1])
    #     print(data)

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# Đường dẫn tới thư mục chứa ảnh và file label.txt
image_dir = './image'
label_file_path = os.path.join(image_dir, 'label.txt')

# Đọc file label.txt và tạo danh sách tên file và nhãn
image_paths = []
labels = []

with open(label_file_path, 'r') as f:
    content = f.read().strip()  # Đọc toàn bộ nội dung và loại bỏ các khoảng trắng ở cuối
    entries = content.split(',')  # Tách các mục dựa trên dấu phẩy
    
    for entry in entries:
        if entry:  # Kiểm tra mục không phải là chuỗi rỗng
            img_name, label = entry.split('|')
            image_paths.append(os.path.join(image_dir, img_name))
            labels.append(int(label))  # Chuyển nhãn thành số nguyên

# Chuyển đổi danh sách ảnh và nhãn thành numpy arrays
images = []
for img_path in image_paths:
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(256, 256))  # Resize ảnh
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    images.append(img_array)

X = np.array(images)
y = np.array(labels)

# Chuẩn hóa dữ liệu (rescale từ 0-255 về 0-1)
X = X / 255.0

# Chia dữ liệu thành tập huấn luyện và kiểm tra (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# Tạo ImageDataGenerator cho việc tăng cường dữ liệu
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

# Tạo generator từ dữ liệu huấn luyện và kiểm tra
train_generator = train_datagen.flow(X_train, y_train, batch_size=10)
val_generator = val_datagen.flow(X_val, y_val, batch_size=10)

# Xây dựng mô hình CNN đơn giản với đầu vào là 256x256
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Đầu ra nhị phân (yes/no)
])

# Biên dịch mô hình
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // 10,
    validation_data=val_generator,
    validation_steps=len(X_train) // 10,
    epochs=100  # Số epoch có thể điều chỉnh
)

# Lưu mô hình đã huấn luyện
model.save('my_face_recognize2.h5')

# Test
predictions = model.predict(X)
print(f"Predict: {predictions}")
print(f"Label: {y}")
