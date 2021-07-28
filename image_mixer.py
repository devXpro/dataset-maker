import os
from random import randint

from image_processor import ImageProcessor


class ImageMixer:
    @staticmethod
    def mix_images():
        backgrounds = os.listdir('image/backgrounds')
        receipts = os.listdir('image/receipts')
        i = 100
        while i > 0:
            background = f'image/backgrounds/{backgrounds[randint(0, len(backgrounds) - 1)]}'
            receipt = f'image/receipts/{receipts[randint(0, len(receipts) - 1)]}'
            try:
                processor = ImageProcessor(background, receipt, str(i))
                processor.process()
            except Exception:
                pass
            i -= 1
            print(i)


x = ImageMixer()
x.mix_images()
