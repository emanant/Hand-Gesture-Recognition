import cv2
import imutils
import numpy as np

bg = None


def run_avg(image, aWeight):

    global bg

    if bg is None:
        bg = image.copy().astype("float")
        return

    cv2.accumulateWeighted(image, bg, aWeight)


def segment(image, threshold=25):
    global bg
    # absolute difference b/w background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

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

    top, right, bottom, left = 100, 450, 350, 700

    num_frames = 0
    image_num = 0

    start_recording = False

    ind = 0
    labels = ["fist", "one", "two", "three", "four", "five",
              "thumbsup", "thumbsdown", "ok", "spock", "swing", "rock"]
    while(True):
        (grabbed, frame) = camera.read()
        if (grabbed == True):

            frame = imutils.resize(frame, width=700)

            frame = cv2.flip(frame, 1)

            clone = frame.copy()

            (height, width) = frame.shape[:2]

            roi = frame[top:bottom, right:left]

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)

            label = labels[ind]

            if num_frames < 30:
                run_avg(gray, aWeight)
                print(num_frames)
            elif image_num <= 120:
                hand = segment(gray)

                if hand is not None:
                    # segmented region
                    (thresholded, segmented) = hand

                    # draw the segmented region and display the frame
                    cv2.drawContours(
                        clone, [segmented + (right, top)], -1, (0, 0, 255))
                    if start_recording:

                        print(image_num, label)
                        # the directory
                        path = "dataset/train/" + \
                            str(label) + "/" + str(label) + \
                            "_" + str(image_num) + '.png'
                        if image_num > 100:
                            path = "dataset/test/" + \
                                str(label) + "/" + str(label) + \
                                "_" + str(image_num) + '.png'
                        cv2.imwrite(path, thresholded)
                        image_num += 1
                    cv2.imshow("Thesholded", thresholded)

            # drawing the segmented image
            cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

            num_frames += 1

            cv2.imshow("Video Feed", clone)

            keypress = cv2.waitKey(1) & 0xFF

            if keypress == ord("q"):
                break

            if keypress == ord("s"):
                start_recording = True

            if keypress == ord("n"):
                ind += 1
                image_num = 0

        else:
            print("[Warning!] Error input, Please check your(camra Or video)")
            break
    camera.release()
    cv2.destroyAllWindows()


main()
