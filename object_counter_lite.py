from library.centroidtracker import CentroidTracker
from library.trackableobject import TrackableObject

import numpy as np
import cv2
import dlib


cap = cv2.VideoCapture("videos/TEST.mov")
frames_count, fps, width, height = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS), cap.get(
    cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

width = int(width)
height = int(height)
print(frames_count, fps, width, height)

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalObject = 0

sub = cv2.createBackgroundSubtractorMOG2()  # create background subtractor
ret, frame = cap.read()  # import image
ratio = 1.0
rgx = slice(0, width)
rgy = slice(0, height)
nrFrameStab = 150

i = 0
cumulativeAffine = np.eye(3)
while True:
    ret, frame = cap.read()  # import image
    if not ret:  # if there is a frame continue with code
        break

    image = cv2.resize(frame, (0, 0), None, ratio, ratio)  # resize image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(image, (15, 15), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    _, th_hue = cv2.threshold(h, 100, 255, cv2.THRESH_BINARY_INV)
    _, th_sat = cv2.threshold(s, 40, 255, cv2.THRESH_BINARY)
    final = cv2.bitwise_and(th_hue, th_sat)

    join = np.concatenate((final, h, s, v), axis=1)


    # if the frame dimensions are empty, set them
    if width is None or height is None:
        (height, width) = frame.shape[:2]

    # initialize the current status along with our list of bounding
    # box rectangles returned by either (1) our object detector or
    # (2) the correlation trackers
    status = "Waiting"
    rects = []

    contours, hierarchy = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    minarea = 10000
    maxarea = 25000
    cxx = np.zeros(len(contours))
    cyy = np.zeros(len(contours))
    for i in range(len(contours)):  # cycles through all contours in current frame
        
        # set the status and initialize our new set of object trackers
        status = "Detecting"
        trackers = []


        if hierarchy[0, i, 3] == -1:  # using hierarchy to only count parent contours (contours not within others)
            area = cv2.contourArea(contours[i])  # area of contour
            cnt = contours[i]
            x, y, w, h = cv2.boundingRect(cnt)

            if minarea > area or area > maxarea:  # area threshold for contour
                continue
                # calculating centroids of contours

            if w/h < 0.5 or w/h > 2:
                continue
            if y < 100:
                continue

            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            # gets bounding points of contour to create rectangle
            # x,y is top left corner and w,h is width and height

            # creates a rectangle around contour
            #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Prints centroid text in order to double check later on
            #cv2.putText(image, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, .3,
                        #(0, 0, 255), 1)
            #cv2.drawMarker(image, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, markerSize=8, thickness=3,
                           #line_type=cv2.LINE_8)

            # compute the (x, y)-coordinates of the bounding box
            # for the object
            #box = cnt[0, 0, i, 3:7] * np.array([w, h, w, h])
            #(startX, startY, endX, endY) = cnt.astype("int")

            # construct a dlib rectangle object from the bounding
            # box coordinates and then start the dlib correlation
            # tracker
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(x, y, x + w, y + h)
            tracker.start_track(gray, rect)
            # add the tracker to our list of trackers so we can
            # utilize it during skip frames
            #trackers.append(tracker)


            # set the status of our system to be 'tracking' rather
            # than 'waiting' or 'detecting'
            status = "Tracking"

            # update the tracker and grab the updated position
            tracker.update(gray)
            pos = tracker.get_position()

            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # add the bounding box coordinates to the rectangles list
            rects.append((startX, startY, endX, endY))





    # use the centroid tracker to associate the (1) old object
    # centroids with (2) the newly computed object centroids
    objects = ct.update(rects)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current
        # object ID
        to = trackableObjects.get(objectID, None)

        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)

        # otherwise, there is a trackable object so we can utilize it
        # to determine direction
        else:
            # the difference between the y-coordinate of the *current*
            # centroid and the mean of *previous* centroids will tell
            # us in which direction the object is moving (negative for
            # 'up' and positive for 'down')
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)

            # check to see if the object has been counted or not
            if not to.counted:

                #GOALS 3 count object 
                # if the direction is positive (indicating the object
                # is moving down) AND the centroid is below the
                # center line, count the object
                if direction > 0 and centroid[1] > h // 2:
                    totalObject += 1
                    to.counted = True

        # store the trackable object in our dictionary
        trackableObjects[objectID] = to

        # draw both the ID of the object and the centroid of the
        # object on the output frame
        #text = "ID {}".format(objectID)
        #cv2.putText(image, text, (centroid[0] - 10, centroid[1] - 10),
            #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(image, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # construct a tuple of information we will be displaying on the
    # frame
    info = [
        ("Confezioni", totalObject),
        #("Stato", status),
    ]

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(image, text, (10, height - ((i * 20) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


    # draw a horizontal line in the center of the frame -- once an
    # object crosses this line we will determine whether they were
    # moving 'up' or 'down'
    cv2.line(image, (0, height // 2), (width, height // 2), (0, 255, 255), 2)


    # show the output frame
    #cv2.imshow("countours", join)
    cv2.imshow("original", image)


    # increment the total number of frames processed thus far and
    # then update the FPS counter
    totalFrames += 1

    #Stop
    key = cv2.waitKey(20)
    if key == 27:
       break
    i = i + 1

cap.release()
cv2.destroyAllWindows()