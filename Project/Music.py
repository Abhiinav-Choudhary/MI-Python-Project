import cv2
import mediapipe as mp
import pygame

# Init Pygame Mixer
pygame.mixer.init()
sounds = {
    "thumb": pygame.mixer.Sound("sa.wav"),
    "index": pygame.mixer.Sound("re.wav"),
    "middle": pygame.mixer.Sound("ga.wav"),
    "ring": pygame.mixer.Sound("ma.wav"),
    "pinky": pygame.mixer.Sound("pa.wav"),
}

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            lm = hand_landmarks.landmark
            h, w, _ = img.shape

            # --- Thumb detection (x-axis based) ---
            if lm[4].x > lm[3].x:  # Right hand logic
                print("thumb is up")
                sounds["thumb"].play()

            # --- Other fingers (y-axis based) ---
            finger_tips = [8, 12, 16, 20]
            finger_lower = [6, 10, 14, 18]
            finger_names = ["index", "middle", "ring", "pinky"]

            for tip_id, lower_id, name in zip(finger_tips, finger_lower, finger_names):
                if lm[tip_id].y < lm[lower_id].y:
                    print(f"{name} is up")
                    sounds[name].play()

    cv2.imshow("Gesture Piano", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
