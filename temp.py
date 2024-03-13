import multiprocessing 
import time 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import tensorflow as tf
import cv2
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def expand_box_with_buffer(box, buffer_percentage, image_shape):
        y_min, x_min, y_max, x_max = box
        y_min = int(y_min * image_shape.shape[0])
        x_min = int(x_min * image_shape.shape[1])
        y_max = int(y_max * image_shape.shape[0])
        x_max = int(x_max * image_shape.shape[1])
        height_buffer = int((y_max - y_min) * buffer_percentage)
        width_buffer = int((x_max - x_min) * buffer_percentage)

        print(f"Original Box: {y_min}, {x_min}, {y_max}, {x_max}")
        print(f"Height Buffer: {height_buffer}, Width Buffer: {width_buffer}")

        y_min_expanded = max(0, y_min - height_buffer)
        y_max_expanded = min(image_shape.shape[0], y_max + height_buffer)
        x_min_expanded = max(0, x_min - width_buffer)
        x_max_expanded = min(image_shape.shape[1], x_max + width_buffer)

        print(f"Expanded Box: {y_min_expanded}, {x_min_expanded}, {y_max_expanded}, {x_max_expanded}")

        return y_min_expanded, x_min_expanded, y_max_expanded, x_max_expanded




# Function to process each frame and save annotated video
def person_detection(input_path, output_path,frame_queue,top_ppe,bottom_ppe):
    PATH_TO_MODEL_DIR = 'model/V9.4/saved_model'
    PATH_TO_LABELS = 'model/V9.4/safety_label_map.pbtxt'
    MIN_CONF_THRESH = 0.3
    print('Loading model...', end='')
    start_time = time.time()
    detect_fn = tf.saved_model.load(PATH_TO_MODEL_DIR)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))

    # LOAD LABEL MAP DATA FOR PLOTTING
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    cap = cv2.VideoCapture(input_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, image_np = cap.read()
        if not ret:
            break

        # Convert to RGB (if needed) and expand dimensions to create a batch of size 1
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_expanded = np.expand_dims(image_rgb, axis=0)

        # Run object detection
        detections = detect_fn(image_expanded)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        frame_queue.put(image_np)
        # Draw bounding boxes and labels on the image
        for i in range(num_detections):
            box = detections['detection_boxes'][i]
            class_id = detections['detection_classes'][i]
            score = detections['detection_scores'][i]

            if score >= MIN_CONF_THRESH and category_index[class_id]["name"]=="Person":
                y_min, x_min, y_max, x_max = box
                y_min = int(y_min * frame_height)
                x_min = int(x_min * frame_width)
                y_max = int(y_max * frame_height)
                x_max = int(x_max * frame_width)
                roi_person = image_np[y_min:y_max, x_min:x_max]
                buffer_percentage = .3  # You can adjust the buffer percentage as needed
                expanded_box = expand_box_with_buffer(box, buffer_percentage, image_np)
                
                # Extract the expanded region from the image
                y_min, x_min, y_max, x_max = expanded_box
                roi_person = image_np[y_min:y_max, x_min:x_max]
                top_half = roi_person[roi_person.shape[0]//6:roi_person.shape[0]//2, :]
                bottom_half = roi_person[roi_person.shape[0]//2:, :]
                height, width, _ = bottom_half.shape
                second_quarter = bottom_half[height//2:, :]
                top_ppe.put(top_half)
                bottom_ppe.put(second_quarter)
        
        cv2.imshow('top half',image_np)
        out.write(image_np)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()




def process_top_half(top_ppe):
    time.sleep(.5)
    PATH_TO_MODEL_DIR = 'model/V9.4/saved_model'
    PATH_TO_LABELS = 'model/V9.4/safety_label_map.pbtxt'
    MIN_CONF_THRESH = 0.3
    print('Loading model...', end='')
    start_time = time.time()
    detect_fn = tf.saved_model.load(PATH_TO_MODEL_DIR)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))

    # LOAD LABEL MAP DATA FOR PLOTTING
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    while True:
        print('top working')
        # print(top_ppe.get())
        image_np = top_ppe.get()
        image_expanded = np.expand_dims(image_np, axis=0)
        detections = detect_fn(image_expanded)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        for i in range(num_detections):
            box = detections['detection_boxes'][i]
            class_id = detections['detection_classes'][i]
            score = detections['detection_scores'][i]

            # if score >= .4 and self.category_index[class_id]["name"] != 'Person':
            if (score >= .5 and category_index[class_id]["name"] in ['Hardhat','Helmet','Safety Vest','Vest']) or (category_index[class_id]["name"]=='Goggle' and score >= .45):
                y_min, x_min, y_max, x_max = box
                y_min = int(y_min * image_np.shape[0])
                x_min = int(x_min * image_np.shape[1])
                y_max = int(y_max * image_np.shape[0])
                x_max = int(x_max * image_np.shape[1])
                
                # Draw bounding box and label on the image
                label = f'{category_index[class_id]["name"]} {int(score * 100)}%'
                if category_index[class_id]["name"] == 'Vest':
                    rgb = (245, 69, 66) #Vest
                elif category_index[class_id]["name"] == 'Helmet':
                    rgb = (66, 245, 105) #Helmet
                elif category_index[class_id]["name"] == 'Goggle':
                    rgb = (75, 245, 66) #Goggle

                    
                
                cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), rgb, 2)
                cv2.putText(image_np, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, rgb, 2)
        cv2.imshow('tt half',image_np)
            
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def process_bottom_half(bottom_ppe):
    time.sleep(.5)
    PATH_TO_MODEL_DIR = 'model/V9.4/saved_model'
    PATH_TO_LABELS = 'model/V9.4/safety_label_map.pbtxt'
    MIN_CONF_THRESH = 0.3
    print('Loading model...', end='')
    start_time = time.time()
    detect_fn = tf.saved_model.load(PATH_TO_MODEL_DIR)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))

    # LOAD LABEL MAP DATA FOR PLOTTING
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    while True:
        print('top working')
        # print(top_ppe.get())
        image_np = bottom_ppe.get()
        image_expanded = np.expand_dims(image_np, axis=0)
        detections = detect_fn(image_expanded)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        for i in range(num_detections):
            box = detections['detection_boxes'][i]
            class_id = detections['detection_classes'][i]
            score = detections['detection_scores'][i]

            # if score >= .4 and self.category_index[class_id]["name"] != 'Person':
            if (score >= .5 and category_index[class_id]["name"] in ['Hardhat','Helmet','Safety Vest','Vest']) or (category_index[class_id]["name"]=='Goggle' and score >= .45):
                y_min, x_min, y_max, x_max = box
                y_min = int(y_min * image_np.shape[0])
                x_min = int(x_min * image_np.shape[1])
                y_max = int(y_max * image_np.shape[0])
                x_max = int(x_max * image_np.shape[1])
                
                # Draw bounding box and label on the image
                label = f'{category_index[class_id]["name"]} {int(score * 100)}%'
                if category_index[class_id]["name"] == 'Vest':
                    rgb = (245, 69, 66) #Vest
                elif category_index[class_id]["name"] == 'Helmet':
                    rgb = (66, 245, 105) #Helmet
                elif category_index[class_id]["name"] == 'Goggle':
                    rgb = (75, 245, 66) #Goggle

                    
                
                cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), rgb, 2)
                cv2.putText(image_np, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, rgb, 2)
        cv2.imshow('bottom half',image_np)
            
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__=='__main__':
    # PROVIDE PATH TO VIDEO FILE
    VIDEO_PATH = 'video/v2.avi'
    frame_queue = multiprocessing.Queue()
    top_ppe = multiprocessing.Queue()
    bottom_ppe = multiprocessing.Queue()
    output_video_path = 'annotated_video.mp4'
    p1 = multiprocessing.Process(target=person_detection,args=(VIDEO_PATH, output_video_path,frame_queue,top_ppe,bottom_ppe))
    p2 = multiprocessing.Process(target=process_top_half,args=(top_ppe,))
    p3 = multiprocessing.Process(target=process_bottom_half,args=(bottom_ppe,))

    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    p3.join()
    # person_detection(VIDEO_PATH, output_video_path)


