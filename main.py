import cv2
from util import Tools

def main():
    t = Tools()
    imgIds = t.imgIds

    N = len(imgIds)
    print(N)

    t.saveAnns()

    # for i in range(0, N):
    #     I = t.drawBBox(i)
    #     cv2.imshow("asdf", I)
    #     cv2.waitKey(3000)

if __name__ == '__main__':
    main()