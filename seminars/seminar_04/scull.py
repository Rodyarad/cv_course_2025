import cv2
import numpy as np
import matplotlib.pyplot as plt

original_image = cv2.imread('data/the_ambassadors_skull_transformed.jpg')
distorted_image = cv2.imread('data/the_ambassadors.jpg')

original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
distorted_image_rgb = cv2.cvtColor(distorted_image, cv2.COLOR_BGR2RGB)

def select_points(image, num_points=4):
    plt.imshow(image)
    plt.title(f'Выберите {num_points} точки')
    points = plt.ginput(num_points, timeout=0) 
    plt.close()
    return np.array(points, dtype=np.float32)

print("Выберите точки на оригинальном изображении:")
original_points = select_points(original_image_rgb, num_points=4)

print("Выберите точки на искаженном изображении:")
distorted_points = select_points(distorted_image_rgb, num_points=4)

transformation_matrix = cv2.getPerspectiveTransform(distorted_points, original_points)

height, width, _ = original_image.shape
corrected_image = cv2.warpPerspective(distorted_image, transformation_matrix, (width, height))

plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(original_image_rgb)
plt.title('Оригинальное изображение')

plt.subplot(1, 3, 2)
plt.imshow(distorted_image_rgb)
plt.title('Искаженное изображение')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB))
plt.title('Исправленное изображение')

plt.show()

cv2.imwrite('corrected_skull.jpg', corrected_image)