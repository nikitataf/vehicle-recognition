import glob
import os
import numpy as np
from skimage import img_as_ubyte
from skimage.io import imread, imsave
from skimage.transform import resize


def load_data(folder):
    """
    Load all images from subdirectories of
    'folder'. The subdirectory name indicates
    the class.
    """

    X = []          # Images go here
    y = []          # Class labels go here
    classes = []    # All class names go here

    subdirectories = glob.glob(folder + "/*")

    # Loop over all folders
    for d in subdirectories:

        # Find all files from this folder
        files = glob.glob(d + os.sep + "*.jpg")

        # Load all files
        for name in files:

            # Load image and parse class name
            img = imread(name)
            class_name = name.split(os.sep)[-2]

            # Convert class names to integer indices:
            if class_name not in classes:
                classes.append(class_name)

            class_idx = classes.index(class_name)

            X.append(img)
            y.append(class_idx)

    # Convert python lists to contiguous numpy arrays
    X = np.array(X)
    y = np.array(y)
    classes = np.array(classes)

    return X, y, classes


def resize_test_data():
    root = './test/testset/'
    new_path = './test/scaled_test/'
    os.mkdir(new_path)
    i = 0
    for file in sorted(os.listdir(root)):
        img = imread(root + file)
        res = resize(img, (200, 200))
        imsave(new_path + os.sep + file, img_as_ubyte(res))
        i = i + 1
        print(str(i) + ' images out of ' + str(len(os.listdir(root))) + ' processed')

    print('Successfully resized')


def resize_train_data():
    root = './train/train/'
    new_path = './train/scaled_train/'
    os.mkdir(new_path)
    i = 0
    # Rescale all files in each subdirectory
    for category in sorted(os.listdir(root)):
        os.mkdir(new_path + category)
        for file in sorted(os.listdir(os.path.join(root, category))):
            img = imread(root + category + os.sep + file)
            res = resize(img, (200, 200))
            imsave(new_path + os.sep + category + os.sep + file, img_as_ubyte(res))
            i = i + 1
            print(str(i) + ' images processed')

    print('Successfully resized')


if __name__ == "__main__":

    # Rescale. One way
    # INTER_NEAREST - a nearest-neighbor interpolation
    # INTER_LINEAR - a bilinear interpolation (used by default)
    # INTER_AREA - resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire’-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method.
    # INTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhood
    # INTER_LANCZOS4 - a Lanczos interpolation over 8x8 pixel neighborhood
    # img = cv2.imread('000009.jpg')
    # res = cv2.resize(img, dsize=(200, 200), interpolation=cv2.INTER_CUBIC)
    # imageio.imwrite('1.jpg', img_as_ubyte(res))

    resize_test_data()
    resize_train_data()
