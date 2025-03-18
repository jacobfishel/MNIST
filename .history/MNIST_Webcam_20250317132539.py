import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not fetch webcam feed.")
        break

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()