import mediapipe as mp
import cv2

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

if not (cap.isOpened()):
    print("Could not open video device")
while True: 
    ret,frame= cap.read()
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = hands.process(frame_rgb)

    # Display results
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on frame
            mpDraw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Tracking', frame)

    cv2.waitKey(1)
cv2.destroyAllWindows()
