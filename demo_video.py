import matplotlib
matplotlib.use('Agg')

import os
import sys
import random
import math
import numpy as np
import skimage.io

import matplotlib.pyplot as plt
import cv2

import glob
import json
import subprocess

def get_framerate(video_path):
    cap = cv2.VideoCapture(video_path)
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver) < 3:
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        print("Frames per second for OpenCV v2: {0}".format(fps))
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("Frames per second for OpenCV v2: {0}".format(fps))
    cap.release()
    fps = round(fps)
    return fps

def get_framerate_bash(video_path):
    command = "ffprobe " + video_path + " -v 0 -select_streams v -print_format flat -show_entries stream=r_frame_rate"
    (stdout, stderr) = subprocess.Popen(command.split(), stdout=subprocess.PIPE).communicate()
    fps = stdout.strip().replace('streams.stream.0.r_frame_rate="', '').replace('"', '')
    nominator, denominator = [int(x) for x in fps.split('/')]
    framerate = int(nominator/denominator)
    return framerate

def create_video(video_name, output_dir):
    video_path = os.path.join(output_dir, video_name + '_out.avi')
    images = glob.glob(os.path.join(output_dir, '*_detrack.jpg'))
    indices = [x.split('__')[-1].replace('_detrack.jpg', '') for x in images]
    indices_int = sorted([int(y) for y in indices])

    image_path = os.path.join(output_dir, '{}__{:05d}_detrack.jpg'.format(video_name, indices_int[0]))
    im = cv2.imread(image_path)

    height, width = im.shape[0:2]

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.cv.CV_FOURCC('M', 'P', '4', 'V')
    out = cv2.VideoWriter(video_path, fourcc, 5.0, (width, height), isColor=True)
    total = len(indices_int)
    for i, index in enumerate(indices_int):
        image_path = os.path.join(output_dir, '{}__{:05d}_detrack.jpg'.format(video_name, index))
        im = cv2.imread(image_path)
        # print image_path, im.shape
        print('{}/{}'.format(i+1, total))
        if im is None:
            print('image not found')
        # im_jpg = im[:, :, 0:3]
        # print im_jpg.shape
        out.write(im)
        # cv2.imshow('frame', im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    cv2.destroyAllWindows()
    print('video saved to: {}'.format(video_path))


def demo_video(video_path, output_dir, framerate=5, max_dim=400):
    colors = np.random.rand(32, 3)
    original_framerate = get_framerate(video_path)
    interval = int(original_framerate / framerate)
    print((original_framerate, framerate, interval))
    if interval < 1:
        interval = 1
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    count = 0
    success = True
    while success:
        # Capture frame-by-frame
        success, frame = cap.read()
        if success and (count % interval == 0):
            image_name = '{}__{:05d}.jpg'.format(video_name, count + 1)
            # print(image_name)
            # Display the resulting frame
            # cv2.imshow('frame', frame)
            # cv2.imwrite(os.path.join(frames_folder, image_name), frame)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = image.shape
            resize_ratio = max(h, w)/(max_dim*1.0)
            image = cv2.resize(image, None, fx=resize_ratio, fy=resize_ratio)
            # print(image.shape)
            # Run detection
            results = model.detect([image], verbose=1)

            # Visualize results
            r = results[0]
            out_image_name = os.path.join(output_dir, os.path.splitext(image_name)[0] + '_detrack.jpg')
            print('processing: {}'.format(image_name))

            visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                        class_names, scores=r['scores'], auto_show=False,
                                        output_name=out_image_name)
            print('output: {}'.format(out_image_name))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        count = count + 1

    # When everything done, release the capture
    cap.release()
    # cv2.destroyAllWindows()
    create_video(video_name, output_dir)

if __name__ == '__main__':
    # Root directory of the project
    ROOT_DIR = os.path.abspath(".")

    # Import Mask RCNN
    sys.path.append(ROOT_DIR)  # To find local version of the library
    from mrcnn import utils
    import mrcnn.model as modellib
    from mrcnn import visualize
    # Import COCO config
    sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
    import coco

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "samples/coco/mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)


    class InferenceConfig(coco.CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']

    video_path = sys.argv[1]
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_dir = os.path.dirname(video_path)

    framerate = 10
    output_dir = os.path.join(ROOT_DIR, 'results', video_name)

    os.makedirs(output_dir, exist_ok=True)
    demo_video(video_path, output_dir, framerate, max_dim=400)

    # visualize.save_image(image, out_image_name, r['rois'], r['masks'],
    #     r['class_ids'], r['scores'], class_names, scores_thresh=0.9, mode=0)
