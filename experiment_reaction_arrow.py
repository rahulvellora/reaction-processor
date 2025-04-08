import cv2
import matplotlib.pyplot as plt

image_path = "out/image_2025_03_26T05_28_03_548Z_reaction.png"
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, threshold1=50, threshold2=150)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / float(h)
    if aspect_ratio > 4 and 50 < w < 500 and h < 10:
        print(x)
        print(x + w)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Detected Arrow Region")
plt.axis("off")
plt.show()
