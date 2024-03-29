import skimage.io as io
from pycocotools.coco import COCO
import numpy as np
import cv2
import json

class Tools:
    def __init__(self):
        dataDir = '../..'
        dataType = 'train2014'
        annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
        self.coco = COCO(annFile)
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self.imgIds = self.coco.getImgIds()
        self.N = len(self.imgIds)

    def getImgUrl(self, index):
        imgInfo = self.coco.loadImgs(self.imgIds)[index]
        return self.getImgId, cv2.cvtColor(io.imread(imgInfo['coco_url']), cv2.COLOR_RGB2BGR)

    def getImgId(self, index):
        imgInfo = self.coco.loadImgs(self.imgIds)[index]
        imgId = imgInfo['id']
        return imgId

    def getAnns(self, imgId):
        annIds = self.coco.getAnnIds(imgIds=imgId)
        return self.coco.loadAnns(annIds)

    def getBBox(self, imgId):
        anns = self.getAnns(imgId)
        objNames = self.getCatNames(anns)
        N = len(anns)
        bboxes = np.zeros((N, 4), dtype=int)
        for i in range(0, N):
            bboxes[i,:] = np.array(anns[i]['bbox'], dtype = int)
        return bboxes, objNames

    def getCatNames(self, anns):
        class_id = []
        for i in range(0, len(anns)):
            class_id.append([x['name'] for x in self.cats if x['id'] == anns[i]['category_id']][0])
        return class_id

    def getImgAnns(self, index):
        imgId, I = self.getImgUrl(index)
        imgInfo = self.coco.loadImgs(self.imgIds)[index]
        bboxes, objNames = self.getBBox(imgInfo['id'])
        return imgId, I, bboxes, objNames

    def drawBBox(self, index):
        imgId, I, bboxes, objNames = self.getImgAnns(index)
        for i in range(0, bboxes.shape[0]):
            x, y = bboxes[i, 0], bboxes[i, 1]
            x2, y2 = x+bboxes[i, 2], y+bboxes[i, 3]
            I = cv2.rectangle(I, (x, y), (x2, y2), (0, 0, 255), 2)
            I = cv2.putText(I, objNames[i], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        return I

    def saveAnns(self):
        igmInfos =  self.coco.loadImgs(self.imgIds)
        dataList = []
        for i in range(0, self.N):
            imgId = self.getImgId(i)
            anns = self.getAnns(imgId)
            imgInfo = igmInfos[i]
            bboxes, objNames = self.getBBox(imgInfo['id'])
            objIds = [x['category_id'] for x in anns]
            data = {"imgId": imgId,
                    "objId": objIds,
                    "objName": objNames,
                    "bboxes": bboxes.tolist()}
            dataList.append(data)
            if np.mod(i,100) == 0:
                print(i)
        with open('train.json', 'w') as f:
            json.dump(dataList, f)





