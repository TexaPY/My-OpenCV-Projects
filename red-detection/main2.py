import cv2
import numpy as np

# Video dosyasını yükle (veya 0 kullanarak webcam'den görüntü alabilirsiniz)
cap = cv2.VideoCapture(0)

while True:
    # Videodan kareyi oku
    ret, frame = cap.read()

    if not ret:
        break

    # Görüntüyü HSV renk uzayına dönüştür
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Kırmızı renk için alt ve üst sınırlar
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)

    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)

    # İki maskeyi birleştir
    mask = mask1 + mask2

    # Sonucu elde et
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Maskeyi ve sonucu göster
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)

    # Çıkmak için 'q' tuşuna basın
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Her şeyi temizle
cap.release()
cv2.destroyAllWindows()
