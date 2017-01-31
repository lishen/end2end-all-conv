import cv2
import argparse
from dm_preprocess import DMImagePreprocessor



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='input image path')
    parser.add_argument('output', type=str, help='output image path')
    parser.add_argument('--remove-pectoral', dest='pect', action='store_true', 
                        help='whether to remove the pectoral muscle region or \
                              not')
    args = parser.parse_args()
    # Preprocess the input image and write to an output image.
    preprocessor = DMImagePreprocessor()
    img_in = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    img_out, _ = preprocessor.process(img_in, pect_removal=args.pect)
    cv2.imwrite(args.output, img_out)

























