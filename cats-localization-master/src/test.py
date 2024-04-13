import os
import cv2 

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

classes = ["Blacky", "Niche"]

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TEST = ("cats_val")
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

predictor = DefaultPredictor(cfg)

# Predict image
def test_image(image_path):
    

    image = cv2.imread(image_path)

    outputs = predictor(image)

    v = Visualizer(image[:, :, ::-1],
        metadata = MetadataCatalog.get("cats_val").set(
            thing_classes=classes,
            thing_colors=[(177, 205, 223), (223, 205, 177)]),
        scale = 0.8,
        instance_mode = ColorMode.IMAGE_BW
        )


    pred_classes = (outputs['instances'].pred_classes).detach()
    pred_scores = (outputs['instances'].scores).detach()

    print(f"File: {image_path}")
    for c, s in zip(pred_classes, pred_scores):
        print(f"--> Class: {classes[c]}, {s * 100:.2f}%")

    v = v.draw_instance_predictions(outputs['instances'].to("cpu"))
    cv2.imwrite("sample_pred.jpg", v.get_image()[:, :, ::-1])

def test_video(video_path):
    # Open the video
    video_cap = cv2.VideoCapture(video_path)

    # Configure the video writer

    # The size of frame must be the same that the output image predicted
    frame_width = 1536 #int(video_cap.get(3))
    frame_height = 864 #int(video_cap.get(4))
    video_writer = cv2.VideoWriter('output.avi', 
                                    cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 
                                    video_cap.get(cv2.CAP_PROP_FPS), 
                                    (frame_width, frame_height))

    if video_cap.isOpened() == False:
        print(f"Error opening the video: {video_path}")
        return

    # Read all video frames
    while(video_cap.isOpened()):
        success, image = video_cap.read()
        #print(f"Original shape: {image.shape}")
        
        if success:
            # Change the color map to RGB
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Make predictions
            outputs = predictor(frame)

            # Create the visualization
            v = Visualizer(image[:, :, ::-1],
                metadata = MetadataCatalog.get("cats_val").set(
                    thing_classes=classes,
                    thing_colors=[(177, 205, 223), (223, 205, 177)]),
                scale = 0.8,
                instance_mode = ColorMode.IMAGE_BW
                )

            print((outputs['instances'].pred_classes).detach())
            print((outputs['instances'].scores).detach())

            v = v.draw_instance_predictions(outputs['instances'].to("cpu"))

            # Write the frame
            img = v.get_image()[:, :, ::-1]
            #print(f"Output shape: {img.shape}")
            video_writer.write(img)
        else:
            break

    # Release the video capture and writer
    video_cap.release()
    video_writer.release()


# Test one image
# image_path = os.path.join("input", "sample.jpg")
# test_image(image_path)

# Test one video
video_path = os.path.join("input", "video.mp4")
test_video(video_path)