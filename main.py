import cv2

# 读取图片
img = cv2.imread("Marsha.jpg")

# 判断是否读取成功
if img is None:
    print("图片读取失败")
    exit()

# 灰度化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 显示原图和灰度图
cv2.imshow("Original", img)
cv2.imshow("Gray", gray)

# 等待按键
cv2.waitKey(0)

# 关闭窗口
cv2.destroyAllWindows()