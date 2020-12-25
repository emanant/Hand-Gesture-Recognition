import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import imutils

# global
bg = None


def resizeImage(imageName):
    basewidth = 100
    img = Image.open(imageName)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    img.save(imageName)


def run_avg(image, aWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return

    cv2.accumulateWeighted(image, bg, aWeight)


def segment(image, threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # get the foreground
    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]
    # cv2.THRESH_TOZERO)[1]

    (im2, cnts, _) = cv2.findContours(thresholded.copy(),
                                      cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)


def main():
    aWeight = 0.5

    camera = cv2.VideoCapture(0)

    # top, right, bottom, left = 210, 350, 425, 590
    # top, right, bottom, left = 200, 300, 450, 550 # 1
    top, right, bottom, left = 100, 450, 350, 700

    # initialize num of frames
    num_frames = 0
    start_recording = False

    while(True):
        (grabbed, frame) = camera.read()

        frame = imutils.resize(frame, width=700)

        frame = cv2.flip(frame, 1)

        clone = frame.copy()

        (height, width) = frame.shape[:2]

        roi = frame[top:bottom, right:left]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if num_frames < 30:
            run_avg(gray, aWeight)
            print(num_frames)
        else:
            hand = segment(gray)

            if hand is not None:
                (thresholded, segmented) = hand

                cv2.drawContours(
                    clone, [segmented + (right, top)], -1, (0, 0, 255))
                if start_recording:
                    cv2.imwrite('Temp.png', thresholded)
                    resizeImage('Temp.png')
                    predictedClass, confidence = getPredictedClass()
                    showStatistics(predictedClass, confidence)
                cv2.imshow("Thesholded", thresholded)

        # drawing the segmented
        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

        num_frames += 1

        cv2.imshow("Video Feed", clone)

        keypress = cv2.waitKey(1) & 0xFF

        if keypress == ord("q"):
            break

        if keypress == ord("s"):
            start_recording = True
    camera.release()
    cv2.destroyAllWindows()


def getPredictedClass():
    # Predict
    image = cv2.imread('Temp.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, (100, 100))
    prediction = model.predict([gray_image.reshape(1, 100, 100, 1)])
    print(prediction)
    return np.argmax(prediction), (np.amax(prediction) / (prediction[0][0] + prediction[0][1] + prediction[0][2] + prediction[0][3] + prediction[0][4] + prediction[0][5] + prediction[0][6] + prediction[0][7] + prediction[0][8] + prediction[0][9] + prediction[0][10] + prediction[0][11]))
    # return np.argmax(prediction), (np.amax(prediction) / (prediction[0][0] + prediction[0][1] + prediction[0][2] + prediction[0][3] + prediction[0][4] + prediction[0][5] + prediction[0][6] + prediction[0][7] + prediction[0][8] + prediction[0][9] + prediction[0][10] + prediction[0][11]))


def showStatistics(predictedClass, confidence):

    textImage = np.zeros((300, 512, 3), np.uint8)
    className = ""

    # dataset
    if predictedClass == 0:
        className = "Fist"
    elif predictedClass == 1:
        className = "Five"
    elif predictedClass == 2:
        className = "Four"
    elif predictedClass == 3:
        className = "Ok"
    elif predictedClass == 4:
        className = "One"
    elif predictedClass == 5:
        className = "Rock"
    elif predictedClass == 6:
        className = "Spock"
    elif predictedClass == 7:
        className = "Swing"
    elif predictedClass == 8:
        className = "Three"
    elif predictedClass == 9:
        className = "Thumbsdown"
    elif predictedClass == 10:
        className = "ThumbsUp"
    elif predictedClass == 11:
        className = "Two"

    # dataset_1
    # if predictedClass == 0:
    #     className = "Fist"
    # elif predictedClass == 1:
    #     className = "Five"
    # elif predictedClass == 2:
    #     className = "Four"
    # elif predictedClass == 3:
    #     className = "Ok"
    # elif predictedClass == 4:
    #     className = "One"
    # elif predictedClass == 5:
    #     className = "Palm"
    # elif predictedClass == 6:
    #     className = "Rock"
    # elif predictedClass == 7:
    #     className = "Spock"
    # elif predictedClass == 8:
    #     className = "Swing"
    # elif predictedClass == 9:
    #     className = "Three"
    # elif predictedClass == 10:
    #     className = "ThumbsUp"
    # elif predictedClass == 11:
    #     className = "Two"

    cv2.putText(textImage, "Pedicted Class : " + className,
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2)

    cv2.putText(textImage, "Confidence : " + str(confidence * 100) + '%',
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2)
    cv2.imshow("Statistics", textImage)


# Model defined
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                           input_shape=(100, 100, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(12, activation='softmax')
])
model.summary()

# Load Saved Model
model.load_weights("TrainedModel/GestureRecogModel.h5")

main()
