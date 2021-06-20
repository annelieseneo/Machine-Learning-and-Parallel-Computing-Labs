# import library for computer vision
import cv2

# READ, WRITE, AND DISPLAY AN IMAGE

# read an image
image = cv2.imread(r'C:\Users\annel\Downloads\ITS66604 Machine Learning and Parallel Computing\labs\image_flower.jpg')
# storing image reading into a variable to use it multiple times without rereading
# less expensive method

# showing the image
cv2.imshow('The title of the window for the image of flower, 9th June 2021', image)

# put execution on hold, value 0 allows forever until window is closed
cv2.waitKey(0)

# close the window to show next image
cv2.destroyAllWindows()

# write the image
cv2.imwrite(r'C:\Users\annel\Downloads\ITS66604 Machine Learning and Parallel Computing\labs\cv outputs\image_flower.png', 
            image)
# output is "True" for successful write as .png file at the location



# COLOR SPACE CONVERSION

# read an image
image2 = cv2.imread(r'C:\Users\annel\Downloads\ITS66604 Machine Learning and Parallel Computing\labs\Penguins.jpg')

# show the image
cv2.imshow(r'C:\Users\annel\Downloads\ITS66604 Machine Learning and Parallel Computing\labs\BGR_Penguins', 
           image2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# convert the BGR image into grayscale
image2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)

# show the converted image
cv2.imshow(r'C:\Users\annel\Downloads\ITS66604 Machine Learning and Parallel Computing\labs\Gray_Penguins', 
           image2)
cv2.waitKey(0)
cv2.destroyAllWindows()



# EDGE DETECTION

# read an image
image3 = cv2.imread(r'C:\Users\annel\Downloads\ITS66604 Machine Learning and Parallel Computing\labs\Penguins.jpg')

# detect the edges using Canny()
cv2.imwrite(r'C:\Users\annel\Downloads\ITS66604 Machine Learning and Parallel Computing\labs\cv outputs\edges_Penguins.jpg',
            cv2.Canny(image3,200,300))

# show the image edges
cv2.imshow(r'C:\Users\annel\Downloads\ITS66604 Machine Learning and Parallel Computing\labs\edges_Penguins', 
           image3)
cv2.waitKey(0)
cv2.destroyAllWindows()



# FACE DETECTION

# import haarcascade_frontalface_default.xml classifier
face_detection = cv2.CascadeClassifier(r'C:\Users\annel\Downloads\ITS66604 Machine Learning and Parallel Computing\labs\haarcascades\haarcascades\haarcascade_frontalface_default.xml')

# read an image
image4 = cv2.imread(r'C:\Users\annel\Downloads\ITS66604 Machine Learning and Parallel Computing\labs\AB.jpg')

# convert the BGR image to only acceptable grayscale format
gray = cv2.cvtColor(image4, cv2.COLOR_BGR2GRAY)

# perform face detection
faces = face_detection.detectMultiScale(gray, 1.3, 5)

# draw a rectangle to indicate the face
for (x,y,w,h) in faces:
    image4 = cv2.rectangle(image4,(x,y),(x+w, y+h),(0,255,0),3)
    # (0,255,0) indicates the color of the rectangle
    # 3 indicates the thickness of the border lines

# write the image
cv2.imwrite(r'C:\Users\annel\Downloads\ITS66604 Machine Learning and Parallel Computing\labs\cv outputs\Face_AB.jpg',
            image4)    



# EYE DETECTION

# import the haarcascade_eye.xml classifier
eye_cascade = cv2.CascadeClassifier(r'C:\Users\annel\Downloads\ITS66604 Machine Learning and Parallel Computing\labs\haarcascades\haarcascades\haarcascade_eye.xml')

# read an image
image5 = cv2.imread(r'C:\Users\annel\Downloads\ITS66604 Machine Learning and Parallel Computing\labs\AB_Eye.jpg')

# convert the BGR image to only acceptable grayscale format
gray2 = cv2.cvtColor(image5, cv2.COLOR_BGR2GRAY)

# perform eye detection
eyes = eye_cascade.detectMultiScale(gray2, 1.03, 5)

# draw a rectangle to indicate the eyes
for (ex,ey,ew,eh) in eyes:
    image5 = cv2.rectangle(image5,(ex,ey),(ex+ew, ey+eh),(0,0,255),2)

# write  the image    
cv2.imwrite(r'C:\Users\annel\Downloads\ITS66604 Machine Learning and Parallel Computing\labs\cv outputs\Eye_AB.jpg',
            image5)