#!/usr/bin/env python3

# IMPORTS
import rospy
import roslib
import numpy as np
import cv2
import time
import math
from cv_bridge import CvBridge
from sensor_msgs.msg import Image as SensImage
from geometry_msgs.msg import Pose 

from PIL import Image
from sensor_msgs.msg import Image as ROSImage

import argparse
import os
import sys

from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import torch
import detectron2
print(f"Detectron2 version is {detectron2.__version__}")
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import Metadata
from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog

from PIL import Image 

#GLOBAL VARIABLES
cv_image = None

# FUNCTIONS
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='custom', help='Model path(s)')
    parser.add_argument('--source', nargs='+', type=str, default='ros', help='Path to an image, default works with ros')
    parser.add_argument('--device', nargs='+', type=str, default=['cpu'], help='Device to use: cpu or cuda')
    parser.add_argument('--confidence', nargs='+', type=float, default=[0.7], help='Min confidence to do mask')
    parser.add_argument('--show_segmentation', nargs='+', type=int, default=[0], help='1 or 0(True or False), publish the segmentation or not, without segmentation is faster')
    parser.add_argument('--save_frames', nargs='+', type=int, default=[0], help='put anything put 0 to save segmentation, show_segmentation must be True')
    parser.add_argument('--resize', nargs='+', type=float, default=[1], help='To change resolution of the image between [0, 1]')
    parser.add_argument('--crop', nargs='+', type=int, default=[1], help='1 to crop a square image, 0 to keep the original without crop')
    opt = parser.parse_args()
    return opt

def showSegmentation(visualizer,out,plot=True,pub=None):
    # visualizer: An instance of the Detectron2 Visualizer class.
    # out: Output from a Detectron2 instance segmentation model.
    # plot: Boolean, whether to display the result using OpenCV.
    # pub: ROS publisher for publishing the segmentation result.
    # Draw instance predictions on the visualizer.
    out_vis = visualizer.draw_instance_predictions(out["instances"].to("cpu"))
    # If plot is True, display the segmentation result using OpenCV.
    if plot:
        cv2.imshow("Detection",out_vis.get_image()[:, :, ::-1])
    # If pub is not None, publish the segmentation result as a ROS message.    
    if pub is not None:
        cv_bridge=CvBridge()
        #converts an OpenCV image (NumPy array) to a ROS sensor_msgs/Image message.
        pub.publish(cv_bridge.cv2_to_imgmsg(out_vis.get_image()[:, :, ::-1], 'bgr8'))
    #cv2.waitKey(0)
    return out_vis



def getMask(out):
    # Extract the instance masks from the output
    masks = np.asarray(out["instances"].pred_masks.to("cpu"))
    # Check if any masks were detected
    if len(masks) > 0:
        # Initialize an empty mask of the same size as the input masks
        total_mask = np.zeros([masks.shape[1], masks.shape[2], 1], dtype=np.uint8)
        # Iterate through the detected masks
        for i in range(len(masks)):
            # Pick an item to mask
            item_mask = masks[i]

            # Create a PIL image out of the mask
            mask = Image.fromarray((item_mask * 255).astype('uint8'))
            mask_imageBlackWhite = np.array(mask)
            # Convert the black and white mask to an RGB format
            mask_image = cv2.cvtColor(mask_imageBlackWhite, cv2.COLOR_GRAY2RGB)
            # Add the current mask to the total mask
            total_mask = np.add(total_mask, mask_image)
            # Print mask information
            print(f"Mask {i} processed: shape {mask_image.shape}")
    else:
        total_mask = None
        print("No masks detected.")
    return total_mask

def getOrientedBoxes(mask, plot, pub=None):
    if mask is None:
        print("No mask provided.")
        return None, None, None
    
    # Convert image to grayscale
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    im_height, im_width = gray.shape

    # Convert image to binary
    _, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
 
    # Find all the contours in the thresholded image
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    area = 0
    box = None
    center = None
    angle = None
    altitude = None

    for i, c in enumerate(contours): 
        if cv2.contourArea(c) > area:
            area = cv2.contourArea(c)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            center = (int(rect[0][0]), int(rect[0][1])) 
            width = int(rect[1][0])
            height = int(rect[1][1])
            angle = int(rect[2])

            if width < height:
                angle = 90 - angle
            else:
                angle = 180 - angle

            if angle > 180:
                angle = angle - 180

            angle = angle - 90

            if width < height:
                norm_width = width / im_width
            else:
                norm_width = height / im_width
        
            altitude = 0.75 / (norm_width + 0.01)
    
    if box is not None:
        label = f"{angle} deg "
        cv2.drawContours(mask, [box], 0, (0, 0, 255), 2)
        cv2.putText(mask, label, (center[0], center[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 1, cv2.LINE_AA) 
        mask = cv2.circle(mask, (center[0], center[1]), radius=10, color=(0, 0, 255), thickness=3)
        center = [int(1000 * (center[0] - im_width / 2) / im_width), int(1000 * (center[1] - im_height / 2) / im_height)]
        
        # Print bounding box information
        print(f"Bounding Box Center: {center}, Angle: {angle} degrees, Altitude: {altitude}")
    else:
        print("No valid contours found.")
        center, angle, altitude = None, None, None

    if plot:
        cv2.imshow("Mask with boxes", mask)

    if pub is not None:
        cv_bridge = CvBridge()
        pub.publish(cv_bridge.cv2_to_imgmsg(mask, 'bgr8'))

    return center, angle, altitude

# Example usage:
# out = {"instances": instances} # Assuming 'instances' is a valid object with pred_masks
# mask = getMask(out)
# getOrientedBoxes(mask, plot=True)



# SUBSCRIBERs CALLBACK
def callback(startImg):
    global cv_image
    cv_bridge=CvBridge()
    cv_image = cv_bridge.imgmsg_to_cv2(startImg, 'bgr8')
    #print("image receveid")

def main():
    rospy.init_node('detectron2', anonymous=False)
    args= parse_opt()
    save_frames=args.save_frames[0]
    resize=args.resize[0]
    crop=args.crop[0]
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence[0]  # set threshold for this model
    cfg.MODEL.DEVICE= args.device[0] # Run on cpu/cuda
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    if(args.weights=="default"):
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
        my_metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        #print(my_metadata)

    else:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES=1
        cfg.MODEL.WEIGHTS = os.path.join(ROOT,"/home/alaaeldin/follow_railway/src/railway_follow/insp_rail_pkg/model/Detectron2_Models/detectron2_101_sim/model_final.pth",args.weights[0])
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        my_metadata = Metadata()
        my_metadata.set(thing_classes = ['rail'])
        #print(my_metadata)

    predictor = DefaultPredictor(cfg)
    
    show_segmentation=args.show_segmentation[0]

    # For test and debug
    if(args.source!='ros'):
        if(os.path.isfile(args.source[0])):
            img = cv2.imread(args.source[0])
            # resize image
            im_height,im_width,_=img.shape
            if(crop==1):
                if(im_width>im_height):
                    img=img[0:int(im_height),int((im_width-im_height)/2):int(im_width-(im_width-im_height)/2)]
                else:
                    img=img[int((im_height-im_width)/2):int(im_height-(im_height-im_width)/2),0:int(im_width)]
                im_height,im_width,_=img.shape
    
            dsize = (int(im_width*resize), int(im_height*resize))
            img = cv2.resize(img, dsize, interpolation = cv2.INTER_AREA)
            #cv2.imshow("my image",img)
            #cv2.waitKey()
            now = time.time()   
            outputs = predictor(img)
        
            v = Visualizer(img[:, :, ::-1],
               metadata=my_metadata, 
               scale=1, 
               instance_mode=ColorMode.IMAGE_BW    # remove the colors of unsegmented pixels. This option is only available for segmentation models
               )

            mask=getMask(outputs)
            getOrientedBoxes(mask,True)
            segmentation=showSegmentation(v,outputs)
            if(save_frames>0):
                    image_name="image"+str(save_frames)+".jpg"
                    print(cv2.imwrite(os.path.join(ROOT,image_name), segmentation.get_image()[:, :, ::-1]))
            print("img dimension: ",img.shape)
            print("total time: ", time.time()-now)
            cv2.waitKey(0)

        if(os.path.isdir(args.source[0])):
            pub_boxes = rospy.Publisher('boxes_and_mask', SensImage, queue_size=1)
            pub_segm = rospy.Publisher('segmentation', SensImage, queue_size=1)
            pub_loc = rospy.Publisher('localization', Pose, queue_size=1)
            
            files = [f for f in os.listdir(args.source[0]) if os.path.isfile(os.path.join(args.source[0], f))]
            #for sorting the file names properly
            files.sort()
    
            for i in range(len(files)):
                filename=args.source[0] + files[i]
                # reading each files
                img = cv2.imread(filename)
                # resize image
                im_height,im_width,_=img.shape
                if(crop==1):
                    if(im_width>im_height):
                        img=img[0:int(im_height),int((im_width-im_height)/2):int(im_width-(im_width-im_height)/2)]
                    else:
                        img=img[int((im_height-im_width)/2):int(im_height-(im_height-im_width)/2),0:int(im_width)]
                    im_height,im_width,_=img.shape
                    
                dsize = (int(im_width*resize), int(im_height*resize))
                img = cv2.resize(img, dsize, interpolation = cv2.INTER_AREA)
                now = time.time()   
                outputs = predictor(img)
        
                v = Visualizer(img[:, :, ::-1],
                    metadata=my_metadata, 
                    scale=1, 
                    instance_mode=ColorMode.IMAGE_BW    # remove the colors of unsegmented pixels. This option is only available for segmentation models
                    )

                mask=getMask(outputs)
                [center,angle,altitude]=getOrientedBoxes(mask,True,pub_boxes)
                segmentation=showSegmentation(v,outputs,True,pub_segm)
                
                
                if(center is not None and angle is not None):
                    loc=Pose()
                    loc.position.x=center[0]
                    loc.position.y=center[1]
                    loc.position.z=altitude
                    loc.orientation.x=im_width
                    loc.orientation.y=im_height
                    loc.orientation.z=angle
                    pub_loc.publish(loc)

                else:
                    loc=Pose()
                    loc.orientation.w = 42 # Just a way to say that no rail was detected
                    loc.orientation.x=im_width
                    loc.orientation.y=im_height
                    pub_loc.publish(loc)

                print(files[i])
                print("img dimension: ",img.shape)
                print("total time: ", time.time()-now)
                cv2.waitKey(0)
                
            print("Done all test images")
                

    else:
        pub_boxes = rospy.Publisher('boxes_and_mask', SensImage, queue_size=1)
        pub_segm = rospy.Publisher('segmentation', SensImage, queue_size=1)
        pub_loc = rospy.Publisher('localization', Pose, queue_size=1)
        
        rospy.Subscriber("output/image_raw", SensImage, callback,queue_size=1)
        time.sleep(1)

        while not rospy.is_shutdown():
            now = time.time()
            print("cv_image:", cv_image)
 

            img=cv_image
            im_height,im_width,_=img.shape

            if(crop==1):
                if(im_width>im_height):
                    img=img[0:int(im_height),int((im_width-im_height)/2):int(im_width-(im_width-im_height)/2)]
                else:
                    img=img[int((im_height-im_width)/2):int(im_height-(im_height-im_width)/2),0:int(im_width)]
                im_height,im_width,_=img.shape

            dsize = (int(im_width*resize), int(im_height*resize))
            img = cv2.resize(img, dsize, interpolation = cv2.INTER_AREA)

            outputs = predictor(img)
            mask=getMask(outputs)
            [center,angle,altitude]=getOrientedBoxes(mask,False,pub_boxes)
            #[center,angle,altitude]=getOrientedBoxes(mask,False)

            if(show_segmentation==1):
                v = Visualizer(img[:, :, ::-1],
                 metadata=my_metadata, 
                 scale=1, 
                 instance_mode=ColorMode.IMAGE_BW    # remove the colors of unsegmented pixels. This option is only available for segmentation models
                 )
                segmentation=showSegmentation(v,outputs,False,pub_segm)

                if(save_frames>0):
                    frame_name="frame"+str(save_frames)+".jpg"
                    cv2.imwrite(os.path.join(ROOT,"frames",frame_name), segmentation.get_image()[:, :, ::-1])
                    save_frames=save_frames+1

            if(center is not None and angle is not None):
                loc=Pose()
                loc.position.x=center[0]
                loc.position.y=center[1]
                loc.position.z=altitude
                loc.orientation.x=im_width
                loc.orientation.y=im_height
                loc.orientation.z=angle
                pub_loc.publish(loc)

            else:
                loc=Pose()
                loc.orientation.w = 42 # Just a way to say that no rail was detected
                loc.orientation.x=im_width
                loc.orientation.y=im_height
                pub_loc.publish(loc)

            print("img dimension: ",img.shape)
            print("img seg time: ", time.time()-now)

    #rospy.spin()


if __name__ == "__main__":
    main()
