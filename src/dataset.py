'''
convert image to bird_eyes_images
project pcl to segmented images
convert pcl to bird_eyes_images

'''

Class Dataset(object):
    def __init__(self,image,pointcloud,camera,lidar):
        self.image= image
        self.pcl=pointcloud
        self.camera =camera ##param of camera
        self.lidar =lidar  ##param of lidar


    def convert_image_to_bird_eyes(self):
        '''

        :param self:
        :return:
        '''
