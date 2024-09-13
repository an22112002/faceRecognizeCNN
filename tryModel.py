import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
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

# Tải lại mô hình từ file
model = tf.keras.models.load_model('my_face_recognize.h5')

# In ra cấu trúc của mô hình để kiểm tra
model.summary()

# Sử dụng mô hình để dự đoán
# Ví dụ: Dự đoán cho một ảnh mới (ảnh đã qua tiền xử lý)
# img_array: là một ảnh đã được chuyển đổi thành numpy array và chuẩn hóa

predictions = model.predict(X)
for i in range(0, len(X)):
    print(f"Predict: {predictions[i]} - Label: {y[i]}")