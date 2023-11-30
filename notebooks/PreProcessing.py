import cv2
import numpy as np
from PIL import Image
import extcolors
from sklearn.metrics.pairwise import cosine_distances
from IPython.core.display import display,HTML

class PreProcessing:

    def __init__(self) -> None:
        calculate_padding = self.calculate_padding
        cropNcolor = self.cropNcolor
        color = self.color
        displayImages = self.displayImages
        remove_duplicate = self.remove_duplicate
        resize_and_pad = self.resize_and_pad
        selection = self.selection
        sort = self.sort

    def remove_duplicate(lst):
        """
        Remove duplicate entries from a list based on the maximum value of a specific key.

        Parameters:
        - lst (list): List of items with associated key, value, and another value.

        Returns:
        - dict: A dictionary containing unique items based on the maximum value of the specified key.
        """
        max_values = {}
        for item in lst:
            key, value,c = item
            if key not in max_values or value > max_values[key][0]:
                max_values[key] = [value,c]
        return max_values


    def sort(obj):
        """
        Sort a list of objects based on certain criteria.

        Parameters:
        - obj: An object containing information about boxes and their characteristics.

        Returns:
        - dict: A dictionary containing sorted values based on specified criteria.
        """
        cls = obj[0].boxes.cls.tolist()
        conf =obj[0].boxes.conf.tolist()
        coord = obj[0].boxes.xyxy.to('cpu').numpy().astype(int)
        if cls!=[] and cls !=[0]:
            lst = list(zip(cls,conf,coord))
            d = PreProcessing.remove_duplicate(lst)
            orted_dict = dict(sorted(d.items(), key=lambda item: item[1],reverse=True))
        else:
            orted_dict={}
        return orted_dict


    def selection(obj):
        """
        Select specific items based on certain conditions.

        Parameters:
        - obj: An object containing information about sorted boxes.

        Returns:
        - tuple: Two lists containing selected keys and values based on specified conditions.
        """
        dict_ = PreProcessing.sort(obj)
        if len(dict_)==1:   #2=full
            return list(dict_.keys()),list(dict_.values())
        if len(dict_)==2 and  3 not in list(dict_.keys()):
            return list(dict_.keys()),list(dict_.values())
        if len(dict_)==2 and  3 in list(dict_.keys()):
            return [list(dict_.keys())[0]],[list(dict_.values()[0])]
        if  all([True if i in list(dict_.keys()) else False for i in [1,2,3] ]):
            if np.average([dict_[1][0],dict_[2][0]])>dict_[3][0]:
                return [1,2],[dict_[1][1],dict_[2][1]]

            if np.average([dict_[1][0],dict_[2][0]])<dict_[3]:
                return [3],dict_[3][1]
        else:
            return [],[]


    def calculate_padding(frame, new_width, new_height):
        """
        Calculate padding based on the aspect ratio of the current and new dimensions.

        Parameters:
        - frame: The input image frame.
        - new_width: The desired width.
        - new_height: The desired height.

        Returns:
        - tuple: Padding values for width and height.
        """
        curr_height, curr_width = frame.shape[:2]
        curr_aspect_ratio = curr_height / curr_width
        new_aspect_ratio = new_height / new_width
        pad_height = 0
        pad_width = 0
        height_to_keep_aspect_ratio = int(curr_height * (new_width / curr_width))
        width_to_keep_aspect_ratio = int(curr_width * (new_height / curr_height))
        if new_aspect_ratio > curr_aspect_ratio:  # pad in height axis. for example, 16:9 -> 4:3
            pad_height = new_height - height_to_keep_aspect_ratio
        else:  # pad in width axis. for example, 1:1 -> 16:9
            pad_width = new_width - width_to_keep_aspect_ratio
        return pad_width, pad_height


    def resize_and_pad(frame, new_width, new_height, detection_pad_value, downsample_algorithm=cv2.INTER_AREA):
        """
        Resize and pad an image to the desired dimensions.

        Parameters:
        - frame: The input image frame.
        - new_width: The desired width.
        - new_height: The desired height.
        - detection_pad_value: Padding value.
        - downsample_algorithm: Algorithm for downsampling.

        Returns:
        - tuple: Resized and padded image, along with padding values.
        """
        curr_height, curr_width = frame.shape[:2]
        pad_width, pad_height = PreProcessing.calculate_padding(frame , new_width, new_height)

        if pad_height != 0:  # pad in height axis. for example, 16:9 -> 4:3
            frame_downsampled = cv2.resize(frame, (new_width, int(curr_height * (new_width / curr_width))),
                                            interpolation=downsample_algorithm)
        else:
            frame_downsampled = cv2.resize(frame, (int(curr_width * (new_height / curr_height)), new_height),
                                            interpolation=downsample_algorithm)

        frame_downsampled = cv2.copyMakeBorder(frame_downsampled, 0, int(pad_height), 0, int(pad_width),
                                                cv2.BORDER_CONSTANT,
                                                value=detection_pad_value)
        return frame_downsampled, pad_width, pad_height


    def color(d1,clr):
        """
        Match a color with the closest color from a dictionary.

        Parameters:
        - d1: The dictionary of colors.
        - clr: The target color.

        Returns:
        - str: The name of the closest matching color.
        """
        a=np.array(list(clr)).reshape(1,3)
        #print(a.shape)
        l = []
        for i in d1.values():
            b=np.array(list(i)).reshape(1,3)
            l.append(cosine_distances(a,b))
        v = l.index(min(l))
        #print(min(l))
        return list(d1.keys())[v]


    def cropNcolor(img,cord,dict1):
        """
        Crop an image and identify the color.

        Parameters:
        - img: The input image.
        - cord: Coordinates for cropping.
        - dict1: Dictionary of colors.

        Returns:
        - tuple: Color name, cropped image, and RGB values.
        """
        crop_img=img[round(cord[1]):round(cord[3]),round(cord[0]):round(cord[2])]
        #Color Mapping
        im2=cv2.cvtColor(crop_img,cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(im2)
        asd=extcolors.extract_from_image(pil_im,tolerance = 10, limit =10)
        print(asd)
        if asd[0][0][0] == (0,0,0):clr=asd[0][1][0]
        else:clr=asd[0][0][0]
        print("\n\nProduct Color", clr)
        color_name=PreProcessing.color(dict1,clr)
        return color_name,crop_img,clr


    def displayImages(images):

        if(len(images)==0):return "Colour Not Matrch"

        imgs='<div style="display:flex;flex-direction:row;flex-wrap: wrap;justify-content: space-evenly;width:100%">'

        for i in images:

            imgs = imgs+'<img src="'+i+'" width=200 style="box-shadow:0px 0px 5px gray;">'

        imgs=imgs+"</div>"

        display(HTML(imgs))
