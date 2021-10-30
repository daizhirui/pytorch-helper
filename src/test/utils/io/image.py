from unittest import TestCase

import matplotlib.pyplot as plt


class TestImage(TestCase):

    def test_imread(self):
        from pytorch_helper.utils.io.image import imread
        # pdf
        images = imread('/home/aang/DATA-WD-01/Development/BEVNet/iccv2021/images/teaser/teaser.pdf')

        for im in images:
            plt.figure()
            plt.imshow(im)
            plt.show(block=False)
