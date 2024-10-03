import cv2
from keras.models import load_model
import numpy as np
from collections import deque

# Load the pre-trained model
model = load_model('devanagari.keras')
print(model)

# Define the mapping from class indices to Devanagari characters
letter_count = {
    0: 'CHECK', 1: '01_ka', 2: '02_kha', 3: '03_ga', 4: '04_gha', 5: '05_kna',
    6: '06_cha', 7: '07_chha', 8: '08_ja', 9: '09_jha', 10: '10_yna', 11: '11_tta',
    12: '12_ttha', 13: '13_dda', 14: '14_ddha', 15: '15_adna', 16: '16_ta',
    17: '17_tha', 18: '18_da', 19: '19_dha', 20: '20_na', 21: '21_pa', 22: '22_pha',
    23: '23_ba', 24: '24_bha', 25: '25_ma', 26: '26_ya', 27: '27_ra', 28: '28_la',
    29: '29_va', 30: '30_sha', 31: '31_shaa', 32: '32_sa', 33: '33_ha',
    34: '34_ksha', 35: '35_tra', 36: '36_gnya', 37: 'CHECK'
}

def keras_predict(model, image):
    processed = keras_process_image(image)
    pred_probab = model.predict(processed)[0]
    pred_class = np.argmax(pred_probab)
    return max(pred_probab), pred_class

def keras_process_image(img):
    image_x, image_y = 32, 32
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img

# Initialize video capture
cap = cv2.VideoCapture(0)
lower_green = np.array([110, 50, 50])
upper_green = np.array([130, 255, 255])
pts = deque(maxlen=512)
blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
digit = np.zeros((200, 200, 3), dtype=np.uint8)

# Initialize prediction variables
pred_class = None
pred_probab = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv, lower_green, upper_green)
    blur = cv2.GaussianBlur(cv2.medianBlur(mask, 15), (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if contours:
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) > 250:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            cv2.circle(frame, center, int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            pts.appendleft(center)

            for i in range(1, len(pts)):
                if pts[i - 1] is None or pts[i] is None:
                    continue
                cv2.line(blackboard, pts[i - 1], pts[i], (255, 255, 255), 10)
                cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), 5)
    else:
        if pts:
            blackboard_gray = cv2.cvtColor(blackboard, cv2.COLOR_BGR2GRAY)
            blur1 = cv2.GaussianBlur(cv2.medianBlur(blackboard_gray, 15), (5, 5), 0)
            thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            blackboard_cnts, _ = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            if blackboard_cnts:
                cnt = max(blackboard_cnts, key=cv2.contourArea)
                if cv2.contourArea(cnt) > 2000:
                    x, y, w, h = cv2.boundingRect(cnt)
                    digit = blackboard_gray[y:y + h, x:x + w]
                    pred_probab, pred_class = keras_predict(model, digit)
                    print(f"Predicted class: {pred_class}, Probability: {pred_probab}")

        pts = deque(maxlen=512)
        blackboard.fill(0)

    cv2.putText(frame, "USE COLOR - BLUE", (150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    if pred_class is not None:
        cv2.putText(frame, f"Conv Network: {letter_count.get(pred_class, '')}", (20, 470), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow("Frame", frame)
    cv2.imshow("Contours", thresh)

    if cv2.waitKey(10) == 27:
        break

cap.release()
cv2.destroyAllWindows()
