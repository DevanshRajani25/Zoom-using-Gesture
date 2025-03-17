import cv2
from cvzone.HandTrackingModule import HandDetector

# Capture Video from User Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Make Detector
detector = HandDetector(detectionCon=0.8)
StartDist = None
scale = 0
cx, cy = 500, 500

# Load Image
img1 = cv2.imread('img_1.png')
if img1 is None:
    raise ValueError("Error: 'img.png' not found or could not be loaded.")

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from webcam")
        continue  # Skip this iteration if no frame is captured

    hands, img = detector.findHands(img)

    # Check if 2 hands are detected
    if len(hands) == 2:
        fingers1 = detector.fingersUp(hands[0])
        fingers2 = detector.fingersUp(hands[1])

        print(f"Fingers Up (Hand 1): {fingers1}")
        print(f"Fingers Up (Hand 2): {fingers2}")

        if fingers1 == [1, 1, 0, 0, 0] and fingers2 == [1, 1, 0, 0, 0]:
            lmList1 = hands[0]["lmList"]
            lmList2 = hands[1]["lmList"]

            # Ensure both hands have at least 9 landmarks
            if len(lmList1) > 8 and len(lmList2) > 8:
                p1 = lmList1[8][:2]  # Index finger of hand 1 (x, y)
                p2 = lmList2[8][:2]  # Index finger of hand 2 (x, y)

                # Calculate distance between index fingertips
                if StartDist is None:
                    length, info, img = detector.findDistance(p1, p2, img)
                    StartDist = length

                length, info, img = detector.findDistance(p1, p2, img)
                scale = int((length - StartDist) // 2)
                cx, cy = info[4:]  # Center position between fingers

                print(f"Scale: {scale}, Center: ({cx}, {cy})")

    else:
        StartDist = None  # Reset scale when hands are not detected

    # Resize img1
    h1, w1, _ = img1.shape
    newH, newW = h1 + scale, w1 + scale

    # Ensure new dimensions are even numbers to avoid shape mismatch ( otherwise 1 px will lose )
    newH = max(1, (newH // 2) * 2)
    newW = max(1, (newW // 2) * 2)

    img1_resized = cv2.resize(img1, (newW, newH))

    # Ensure cropping region stays within bounds of img
    h, w, _ = img.shape  # Get frame size

    x1, x2 = max(0, cx - newW // 2), min(w, cx + newW // 2)
    y1, y2 = max(0, cy - newH // 2), min(h, cy + newH // 2)

    # Ensure the img1 overlay fits inside img correctly
    img[y1:y2, x1:x2] = img1_resized[:y2 - y1, :x2 - x1]

    cv2.imshow("Image", img)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
