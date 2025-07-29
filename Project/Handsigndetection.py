import cv2
import mediapipe as md

# Initilize mediapipe
md_hands = md.solutions.hands
hands = md_hands.Hands(max_num_hands=1,min_detection_confidence=0.7)
md_draw =md.solutions.drawing_utils

# start capturing video
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            md_draw.draw_landmarks(img, hand_landmarks, md_hands.HAND_CONNECTIONS)

            # Print landmark positions (normalized coordinates)
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(f"ID: {id}, X: {cx}, Y: {cy}")

    cv2.imshow("Hand Tracking", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

