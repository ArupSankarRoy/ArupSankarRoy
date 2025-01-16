import cv2
import numpy as np

def grayscale(img:np.ndarray):
    return cv2.cvtColor(img , cv2.COLOR_RGB2GRAY)

def canny(img:np.ndarray,low_threshold:int,high_threshold:int):
    return cv2.Canny(img,low_threshold,high_threshold)

# Gaussian blur effectively smooths out noise and sharp edges in an image by averaging pixel values using a weighted Gaussian distribution.
def gaussian_blur(img:np.ndarray,kernel_size:int):
    return cv2.GaussianBlur(img,(kernel_size,kernel_size),0) # 0 = Standard deviation

def roi(img:np.ndarray,vertics:np.ndarray):
    
    """Applies a mask to focus on the region of interest."""
    mask = np.zeros_like(img)
    ignore_mask_color = 255 if len(img.shape) == 2 else (255,)*img.shape[2]
    cv2.fillPoly(mask,vertics,ignore_mask_color)
    return cv2.bitwise_and(img , mask)

def draw_lines(img,lines,color=[255,255,0],thickness=6):
    left_lines , right_lines = [] , []

    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                slope = (y2-y1)/(x2-x1+1e-6)

                if 0.3<slope<2:
                    right_lines.append((x1,y1,x2,y2))
                elif -2<slope<-0.3:
                    left_lines.append((x1,y1,x2,y2))

    def average_line(lines):
        if not lines:
            return None

        x_coords,y_coords = [],[]
        for x1,y1,x2,y2 in lines:
            x_coords.extend([x1,x2])
            y_coords.extend([y1,y2])

        ploy = np.polyfit(x_coords,y_coords,1)
        return ploy
    
    left_ploy = average_line(left_lines)
    right_lines = average_line(right_lines)

    if left_ploy is not None:
        slope , intercept = left_ploy
        y1,y2 = img.shape[0] , 260
        x1,x2 = int((y1-intercept)/slope),int((y2-intercept)/slope)
        cv2.line(img,(x1,y1),(x2,y2),color,thickness)

    if right_lines is not None:
        slope,intercept = right_lines
        y1,y2 = img.shape[0],260    
        x1,x2 = int((y1-intercept)/slope),int((y2-intercept)/slope)
        cv2.line(img,(x1,y1),(x2,y2),color,thickness)


def hough_lines(img,rho,theta,threshold,min_line_lane,max_line_gap):

    """
     rho = The resolution of the accumulator in pixels.
     theta = The angular resolution of the accumulator in radians.
     min_line_lane = The minimum length of a line that will be accepted.
     max_line_gap = The minimum gap between line segments that will be connected into a single line.
    
    """

    lines = cv2.HoughLinesP(img,rho,theta,threshold,np.array([]),min_line_lane,max_line_gap)
    line_img = np.zeros((*img.shape,3),dtype=np.uint8) # Create a blank image (MASK) with the same shape as the input image
    draw_lines(line_img,lines)
    return line_img

def weighted_image(initial_img:np.ndarray,img:np.ndarray,alpha:float=0.8,beta:float=1.0,gamma:float=0.0):
    """ Combines the original image with the line image."""
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)

def process_image(image:np.ndarray,width,height):
    img = cv2.resize(image,(width,height))
    blurred = gaussian_blur(img,7)
    edges = canny(blurred,50,150)
    
    vertics = np.array([[(0, height), (width, height), (int(width * 0.6), int(height * 0.5)), (int(width * 0.4), int(height * 0.5))]]) 
    # vertics = np.array([[(216, 559), (276, 522), (333, 480), (373, 445), (973, 520), (889, 500), (804, 475), (742, 455)]],dtype=np.int32)   
    masked_edges = roi(edges, vertics)
    line_img = hough_lines(masked_edges, 1, np.pi / 180, 35, 15, 100)
    rgb_frame = weighted_image(img, line_img)
    return rgb_frame



