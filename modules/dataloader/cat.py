import glob
import math
import os
import shutil
import subprocess
import sys
from multiprocessing.dummy import Pool as ThreadPool
from sys import platform
from zipfile import ZipFile

import cv2
from torchvision import datasets, transforms

from base import BaseDataLoader


class CatDataLoader(BaseDataLoader):
    @property
    def dataset(self):
        _transforms = [
            transforms.Resize(tuple(self.size[1:])),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]

        _is_valid_file = None

        return Cat(root=self.paths.dataset,
                   train=self.is_train,
                   transform=transforms.Compose(_transforms),
                   download=self.is_download,
                   is_valid_file=_is_valid_file)


class Cat(datasets.ImageFolder):
    FILES = ["CAT_DATASET_01.zip", "CAT_DATASET_02.zip"]
    URL = "https://archive.org/download/CAT_DATASET/"

    def __init__(self, root, train=True, transform=None, download=False, is_valid_file=None):
        self.root = root

        if download and not os.path.exists(self.processed_data):
            self.download()
            self.file_adjusment()
            self.preprocess()
        super(Cat, self).__init__(self.processed_data, transform=transform, is_valid_file=is_valid_file)

    def download(self):
        if self._check_raw_exists:
            return
        else:
            os.makedirs(self.raw_folder)
        if platform == 'linux' or platform == 'linux2':
            def call_wget(zip_data):
                url = self.URL + zip_data
                file_output = os.path.join(self.root, self.__class__.__name__, zip_data)
                subprocess.call('wget -nc ' + url + " -O " + file_output, shell=True)

            if not self._check_zip_exists:
                pool = ThreadPool(4)  # Sets the pool size to 4
                # Open the urls in their own threads
                # and return the results
                pool.map(call_wget, self.FILES + ["00000003_015.jpg.cat"])
                # close the pool and wait for the work to finish
                pool.close()
                pool.join()

        print("Please wait, extract files")
        if not os.path.exists(self.tmp_raw_folder):
            os.makedirs(self.tmp_raw_folder)

        for file in self.zip_data:
            with ZipFile(file, 'r') as zipObj:
                # Extract all the contents of zip file in raw folder
                zipObj.extractall(self.tmp_raw_folder)
        print("extract done!")

    def file_adjusment(self):
        # move all sub directory to raw_folder
        for f in os.listdir(self.tmp_raw_folder):
            path_f = os.path.join(self.tmp_raw_folder, f)
            for _f in os.listdir(path_f):
                path__f = os.path.join(path_f, _f)
                if os.path.exists(os.path.join(self.raw_folder, _f)):
                    continue
                else:
                    shutil.move(path__f, self.raw_folder)

        # Error correction

        os.remove(os.path.join(self.raw_folder, "00000003_019.jpg.cat"))
        shutil.move(os.path.join(self.root, self.__class__.__name__, "00000003_015.jpg.cat"), self.raw_folder)

        # Removing outliers
        for c in self.corrupted_data:
            os.remove(os.path.join(self.raw_folder, c))

        # Make cats
        if not os.path.exists(self.data64):
            os.makedirs(self.data64)

        if not os.path.exists(self.data128):
            os.makedirs(self.data128)

    def preprocess(self):
        print("start preprocess")
        for imagePath in glob.glob(self.raw_folder + '/*.jpg'):
            # Open the '.cat' annotation file associated with this
            # image.
            input = open('%s.cat' % imagePath, 'r')
            # Read the coordinates of the cat features from the
            # file. Discard the first number, which is the number
            # of features.
            coords = [int(i) for i in input.readline().split()[1:]]
            # Read the image.
            image = cv2.imread(imagePath)
            # Straighten and crop the cat face.
            crop = preprocessCatFace(coords, image)
            if crop is None:
                print(f'Failed to preprocess image at {imagePath}.', file=sys.stderr)
                continue
            # Save the crop to folders based on size
            h, w, colors = crop.shape
            if min(h, w) >= 64:
                Path1 = imagePath.replace(self.raw_folder, self.data64)
                cv2.imwrite(Path1, crop)
            if min(h, w) >= 128:
                Path2 = imagePath.replace(self.raw_folder, self.data128)
                cv2.imwrite(Path2, crop)
        print("preprocess done!")

    @property
    def data(self):
        return self.imgs

    @property
    def _check_raw_exists(self):
        return os.path.exists(self.raw_folder)

    @property
    def _check_zip_exists(self):
        for file in self.FILES:
            if not os.path.exists(file):
                return False
        return True

    @property
    def _check_processed_data_exists(self):
        return os.path.exists(self.processed_data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def tmp_raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'tmp_raw')

    @property
    def processed_data(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def data64(self):
        return os.path.join(self.processed_data, 'data_64')

    @property
    def data128(self):
        return os.path.join(self.processed_data, 'data_128')

    @property
    def train_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'cats_bigger_than_64x64')

    @property
    def zip_data(self):
        return tuple([os.path.join(self.root, self.__class__.__name__, f) for f in self.FILES])

    @property
    def corrupted_data(self):
        corrupt_data = "00000004_007.jpg 00000007_002.jpg 00000045_028.jpg 00000050_014.jpg 00000056_013.jpg " \
                       "00000059_002.jpg 00000108_005.jpg 00000122_023.jpg 00000126_005.jpg 00000132_018.jpg " \
                       "00000142_024.jpg 00000142_029.jpg 00000143_003.jpg 00000145_021.jpg 00000166_021.jpg " \
                       "00000169_021.jpg 00000186_002.jpg 00000202_022.jpg 00000208_023.jpg 00000210_003.jpg " \
                       "00000229_005.jpg 00000236_025.jpg 00000249_016.jpg 00000254_013.jpg 00000260_019.jpg " \
                       "00000261_029.jpg 00000265_029.jpg 00000271_020.jpg 00000282_026.jpg 00000316_004.jpg " \
                       "00000352_014.jpg 00000400_026.jpg 00000406_006.jpg 00000431_024.jpg 00000443_027.jpg " \
                       "00000502_015.jpg 00000504_012.jpg 00000510_019.jpg 00000514_016.jpg 00000514_008.jpg " \
                       "00000515_021.jpg 00000519_015.jpg 00000522_016.jpg 00000523_021.jpg 00000529_005.jpg " \
                       "00000556_022.jpg 00000574_011.jpg 00000581_018.jpg 00000582_011.jpg 00000588_016.jpg " \
                       "00000588_019.jpg 00000590_006.jpg 00000592_018.jpg 00000593_027.jpg 00000617_013.jpg " \
                       "00000618_016.jpg 00000619_025.jpg 00000622_019.jpg 00000622_021.jpg 00000630_007.jpg " \
                       "00000645_016.jpg 00000656_017.jpg 00000659_000.jpg 00000660_022.jpg 00000660_029.jpg " \
                       "00000661_016.jpg 00000663_005.jpg 00000672_027.jpg 00000673_027.jpg 00000675_023.jpg " \
                       "00000692_006.jpg 00000800_017.jpg 00000805_004.jpg 00000807_020.jpg 00000823_010.jpg " \
                       "00000824_010.jpg 00000836_008.jpg 00000843_021.jpg 00000850_025.jpg 00000862_017.jpg " \
                       "00000864_007.jpg 00000865_015.jpg 00000870_007.jpg 00000877_014.jpg 00000882_013.jpg " \
                       "00000887_028.jpg 00000893_022.jpg 00000907_013.jpg 00000921_029.jpg 00000929_022.jpg " \
                       "00000934_006.jpg 00000960_021.jpg 00000976_004.jpg 00000987_000.jpg 00000993_009.jpg " \
                       "00001006_014.jpg 00001008_013.jpg 00001012_019.jpg 00001014_005.jpg 00001020_017.jpg " \
                       "00001039_008.jpg 00001039_023.jpg 00001048_029.jpg 00001057_003.jpg 00001068_005.jpg " \
                       "00001113_015.jpg 00001140_007.jpg 00001157_029.jpg 00001158_000.jpg 00001167_007.jpg " \
                       "00001184_007.jpg 00001188_019.jpg 00001204_027.jpg 00001205_022.jpg 00001219_005.jpg " \
                       "00001243_010.jpg 00001261_005.jpg 00001270_028.jpg 00001274_006.jpg 00001293_015.jpg " \
                       "00001312_021.jpg 00001365_026.jpg 00001372_006.jpg 00001379_018.jpg 00001388_024.jpg " \
                       "00001389_026.jpg 00001418_028.jpg 00001425_012.jpg 00001431_001.jpg 00001456_018.jpg " \
                       "00001458_003.jpg 00001468_019.jpg 00001475_009.jpg 00001487_020.jpg 00000004_007.jpg.cat " \
                       "00000007_002.jpg.cat 00000045_028.jpg.cat 00000050_014.jpg.cat 00000056_013.jpg.cat " \
                       "00000059_002.jpg.cat 00000108_005.jpg.cat 00000122_023.jpg.cat 00000126_005.jpg.cat " \
                       "00000132_018.jpg.cat 00000142_024.jpg.cat 00000142_029.jpg.cat 00000143_003.jpg.cat " \
                       "00000145_021.jpg.cat 00000166_021.jpg.cat 00000169_021.jpg.cat 00000186_002.jpg.cat " \
                       "00000202_022.jpg.cat 00000208_023.jpg.cat 00000210_003.jpg.cat 00000229_005.jpg.cat " \
                       "00000236_025.jpg.cat 00000249_016.jpg.cat 00000254_013.jpg.cat 00000260_019.jpg.cat " \
                       "00000261_029.jpg.cat 00000265_029.jpg.cat 00000271_020.jpg.cat 00000282_026.jpg.cat " \
                       "00000316_004.jpg.cat 00000352_014.jpg.cat 00000400_026.jpg.cat 00000406_006.jpg.cat " \
                       "00000431_024.jpg.cat 00000443_027.jpg.cat 00000502_015.jpg.cat 00000504_012.jpg.cat " \
                       "00000510_019.jpg.cat 00000514_016.jpg.cat 00000514_008.jpg.cat 00000515_021.jpg.cat " \
                       "00000519_015.jpg.cat 00000522_016.jpg.cat 00000523_021.jpg.cat 00000529_005.jpg.cat " \
                       "00000556_022.jpg.cat 00000574_011.jpg.cat 00000581_018.jpg.cat 00000582_011.jpg.cat " \
                       "00000588_016.jpg.cat 00000588_019.jpg.cat 00000590_006.jpg.cat 00000592_018.jpg.cat " \
                       "00000593_027.jpg.cat 00000617_013.jpg.cat 00000618_016.jpg.cat 00000619_025.jpg.cat " \
                       "00000622_019.jpg.cat 00000622_021.jpg.cat 00000630_007.jpg.cat 00000645_016.jpg.cat " \
                       "00000656_017.jpg.cat 00000659_000.jpg.cat 00000660_022.jpg.cat 00000660_029.jpg.cat " \
                       "00000661_016.jpg.cat 00000663_005.jpg.cat 00000672_027.jpg.cat 00000673_027.jpg.cat " \
                       "00000675_023.jpg.cat 00000692_006.jpg.cat 00000800_017.jpg.cat 00000805_004.jpg.cat " \
                       "00000807_020.jpg.cat 00000823_010.jpg.cat 00000824_010.jpg.cat 00000836_008.jpg.cat " \
                       "00000843_021.jpg.cat 00000850_025.jpg.cat 00000862_017.jpg.cat 00000864_007.jpg.cat " \
                       "00000865_015.jpg.cat 00000870_007.jpg.cat 00000877_014.jpg.cat 00000882_013.jpg.cat " \
                       "00000887_028.jpg.cat 00000893_022.jpg.cat 00000907_013.jpg.cat 00000921_029.jpg.cat " \
                       "00000929_022.jpg.cat 00000934_006.jpg.cat 00000960_021.jpg.cat 00000976_004.jpg.cat " \
                       "00000987_000.jpg.cat 00000993_009.jpg.cat 00001006_014.jpg.cat 00001008_013.jpg.cat " \
                       "00001012_019.jpg.cat 00001014_005.jpg.cat 00001020_017.jpg.cat 00001039_008.jpg.cat " \
                       "00001039_023.jpg.cat 00001048_029.jpg.cat 00001057_003.jpg.cat 00001068_005.jpg.cat " \
                       "00001113_015.jpg.cat 00001140_007.jpg.cat 00001157_029.jpg.cat 00001158_000.jpg.cat " \
                       "00001167_007.jpg.cat 00001184_007.jpg.cat 00001188_019.jpg.cat 00001204_027.jpg.cat " \
                       "00001205_022.jpg.cat 00001219_005.jpg.cat 00001243_010.jpg.cat 00001261_005.jpg.cat " \
                       "00001270_028.jpg.cat 00001274_006.jpg.cat 00001293_015.jpg.cat 00001312_021.jpg.cat " \
                       "00001365_026.jpg.cat 00001372_006.jpg.cat 00001379_018.jpg.cat 00001388_024.jpg.cat " \
                       "00001389_026.jpg.cat 00001418_028.jpg.cat 00001425_012.jpg.cat 00001431_001.jpg.cat " \
                       "00001456_018.jpg.cat 00001458_003.jpg.cat 00001468_019.jpg.cat 00001475_009.jpg.cat " \
                       "00001487_020.jpg.cat "
        corrupt_data = corrupt_data.replace('\n', '').strip()
        return corrupt_data.split(' ')


def rotateCoords(coords, center, angleRadians):
    # Positive y is down so reverse the angle, too.
    angleRadians = -angleRadians
    xs, ys = coords[::2], coords[1::2]
    newCoords = []
    n = min(len(xs), len(ys))
    i = 0
    centerX = center[0]
    centerY = center[1]
    cosAngle = math.cos(angleRadians)
    sinAngle = math.sin(angleRadians)
    while i < n:
        xOffset = xs[i] - centerX
        yOffset = ys[i] - centerY
        newX = xOffset * cosAngle - yOffset * sinAngle + centerX
        newY = xOffset * sinAngle + yOffset * cosAngle + centerY
        newCoords += [newX, newY]
        i += 1
    return newCoords


def preprocessCatFace(coords, image):
    leftEyeX, leftEyeY = coords[0], coords[1]
    rightEyeX, rightEyeY = coords[2], coords[3]
    mouthX = coords[4]
    if leftEyeX > rightEyeX and leftEyeY < rightEyeY and \
            mouthX > rightEyeX:
        # The "right eye" is in the second quadrant of the face,
        # while the "left eye" is in the fourth quadrant (from the
        # viewer's perspective.) Swap the eyes' labels in order to
        # simplify the rotation logic.
        leftEyeX, rightEyeX = rightEyeX, leftEyeX
        leftEyeY, rightEyeY = rightEyeY, leftEyeY

    eyesCenter = (0.5 * (leftEyeX + rightEyeX),
                  0.5 * (leftEyeY + rightEyeY))

    eyesDeltaX = rightEyeX - leftEyeX
    eyesDeltaY = rightEyeY - leftEyeY
    eyesAngleRadians = math.atan2(eyesDeltaY, eyesDeltaX)
    eyesAngleDegrees = eyesAngleRadians * 180.0 / math.pi

    # Straighten the image and fill in gray for blank borders.
    rotation = cv2.getRotationMatrix2D(
        eyesCenter, eyesAngleDegrees, 1.0)
    imageSize = image.shape[1::-1]
    straight = cv2.warpAffine(image, rotation, imageSize,
                              borderValue=(128, 128, 128))

    # Straighten the coordinates of the features.
    newCoords = rotateCoords(
        coords, eyesCenter, eyesAngleRadians)

    # Make the face as wide as the space between the ear bases.
    w = abs(newCoords[16] - newCoords[6])
    # Make the face square.
    h = w
    # Put the center point between the eyes at (0.5, 0.4) in
    # proportion to the entire face.
    minX = eyesCenter[0] - w / 2
    if minX < 0:
        w += minX
        minX = 0
    minY = eyesCenter[1] - h * 2 / 5
    if minY < 0:
        h += minY
        minY = 0

    # Crop the face.
    crop = straight[int(minY):int(minY + h), int(minX):int(minX + w)]
    # Return the crop.
    return crop
