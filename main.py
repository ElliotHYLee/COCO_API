import cv2
from util import Tools

def main():
    t = Tools()
    imgIds = t.coco.getImgIds()

    N = len(imgIds)
    for i in range(0, 10):
        I = t.drawBBox(i)
        cv2.imshow("asdf", I)
        cv2.waitKey(1000)




if __name__ == '__main__'\
        :
    main()