import cv2

# 读取图片
img = cv2.imread("Marsha.jpg")

# 判断是否读取成功
if img is None:
    print("图片读取失败")
    exit()

# 灰度化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 保存灰度图
cv2.imwrite("Marsha_gray.jpg", gray)

print("灰度图已保存为 Marsha_gray.jpg")