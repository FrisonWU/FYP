import cv2
import numpy as np
import matplotlib.pyplot as plt

def homomorphic_filter(src, d0=10, r1=0.5, rh=2, c=4, h=2.0, l=0.5):

    gray = src.copy()
    if len(src.shape) > 2:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)
    rows, cols = gray.shape

    gray_fft = np.fft.fft2(gray)
    gray_fftshift = np.fft.fftshift(gray_fft)
    dst_fftshift = np.zeros_like(gray_fftshift)
    M, N = np.meshgrid(np.arange(-cols // 2, cols // 2), np.arange(-rows // 2, rows // 2))
    D = np.sqrt(M ** 2 + N ** 2)

    Z = (rh - r1) * (1 - np.exp(-c * (D ** 2 / d0 ** 2))) + r1
    dst_fftshift = Z * gray_fftshift
    dst_fftshift = (h - l) * dst_fftshift + l
    dst_ifftshift = np.fft.ifftshift(dst_fftshift)
    dst_ifft = np.fft.ifft2(dst_ifftshift)
    dst = np.real(dst_ifft)
    dst = np.uint8(np.clip(dst, 0, 255,out=dst))

    plt.figure(1)
    plt.title("Fitler Relation")
    plt.plot(D,Z)
    return dst

img1= cv2.imread('box1.jpg')
img1_filtered = homomorphic_filter(img1).copy()
#src = cv2.GaussianBlur(img1, (3, 3), 0)
#gray = cv2.cvtColor(img1_filtered, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(img1_filtered, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow('homofilter',img1_filtered)
cv2.imshow("binary", binary)

