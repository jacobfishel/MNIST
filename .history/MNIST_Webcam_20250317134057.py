import cv2
import torchvision.transforms as transforms

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not fetch webcam feed.")
        break

    cv2.imshow("Webcam", frame)
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])

    # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # transformed = transform(frame_gray)

    # print(transformed.shape)
    
    #First just try to flatten the image and convert it to a tensor












    if cv2.waitKey(1) == ord('q'):
        break
# cap.release()
cv2.destroyAllWindows()