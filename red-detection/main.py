import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # blue
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)
    # red
    lower_red = np.array([160, 90, 90])
    upper_red = np.array([190, 255, 255])
    red_mask = cv2.inRange(hsv_frame, lower_red, upper_red)

    kernal = np.ones((1, 1), "uint8")
    red_mask = cv2.dilate(red_mask, kernal)
    red = cv2.bitwise_and(frame, frame, mask=red_mask)

    blue_mask = cv2.dilate(blue_mask, kernal)
    blue = cv2.bitwise_and(frame, frame, mask=blue_mask)

    # referans noktları2
    frame = cv2.line(frame, (640, 340), (640, 380), (0, 0, 0), 2)
    frame = cv2.line(frame, (620, 360), (660, 360), (0, 0, 0), 2)

    # su bırakma
    contours, hierarchy = cv2.findContours(
        red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            orjinx = int(x + (w / 2))
            orjiny = int(y + (h / 2))
            frame = cv2.circle(frame, (orjinx, orjiny), 10, (0, 255, 0), 4)
            line_uzunluk = math.sqrt(
                (x - 640 + (w / 2)) * (x - 640 + (w / 2))
                + (y - 360 + (h / 2)) * (y - 360 + (h / 2))
            )
            x_koordinat = x - 640 + (w / 2)
            y_koordinat = (y + (h / 2) - 360) * (-1)
            if line_uzunluk < 50:
                cv2.putText(
                    frame,
                    "Kargo Birakma Konumu",
                    (900, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
            cv2.putText(
                frame,
                "Kargo Birakma Konumu",
                (900, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                "Uzaklik => " + str(line_uzunluk),
                (900, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                "X Koordinat => " + str(x_koordinat),
                (900, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                "Y Koordinat => " + str(y_koordinat),
                (900, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            frame = cv2.line(
                frame, (640, 360), (x + int(w / 2), y + int(h / 2)), (0, 255, 0), 1
            )
            cv2.putText(
                frame,
                "Kargo Birakma Alani",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
            )

    cv2.imshow("webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
