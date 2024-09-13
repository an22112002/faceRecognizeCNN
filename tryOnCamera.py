import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import tensorflow as tf
import numpy as np
model = tf.keras.models.load_model('my_face_recognize2.h5')
model.summary()

# Mở camera (camera ID là 0, nếu bạn có nhiều camera thì có thể đổi ID này)
cap = cv2.VideoCapture(0)

while True:
    # Đọc một khung hình từ camera
    ret, frame = cap.read()

    if not ret:
        print("Không thể nhận hình ảnh từ camera")
        break

    # Phân loại ảnh
    resized_frame  = cv2.resize(frame, (256, 256))
    input_data = np.array(resized_frame, dtype=np.float32)
    input_data = np.expand_dims(input_data, axis=0)
    input_data /= 255.0
    predict = model.predict(input_data, verbose=0)[0][0]
    
    if predict >= 0.99:
        print("Nhận diện")

    # Hiển thị hình ảnh
    cv2.imshow('Camera', frame)

    # Nhấn phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng camera và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()