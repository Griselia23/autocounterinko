import cv2

# Membaca gambar masjid.png
image = cv2.imread('masjid.png')

# Mengubah gambar menjadi grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Menyimpan hasil gambar grayscale
cv2.imwrite('masjid_grayscale.png', gray_image)

# Menampilkan gambar hasil
cv2.imshow('Grayscale Image', gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
