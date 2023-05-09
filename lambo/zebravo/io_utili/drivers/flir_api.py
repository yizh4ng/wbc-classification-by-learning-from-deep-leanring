'''
Created on Mar 26, 2018
@author: Hao Wu
'''

import numpy as np
import pyspin.PySpin as pyspin

'''
FLIRCamDev is the FoundryScope Driver for Point-Grey cameras. It is calling the 
FLIR Spinnaker Python binding pyspin. The newest version of pyspin can be
obtained from the FLIR official website
'''


class FLIRCamDev(object):

    def __init__(self):
        '''
        camera id is 0,1,2,..., the maximum is the number of point-grey camera
        connected to the computer
        '''
        #self.camera_sn = camera_sn
        self.open()

    '''
    Camera operations
    '''

    def open(self):
        '''
        open up the connection to the camera
        '''
        try:
            # find the list of camera and choose the right camera
            self.system = pyspin.System.GetInstance()
            self.cam_list = self.system.GetCameras()

            # get the camera by id
            #self.cam = self.cam_list.GetBySerial(self.camera_sn)
            self.cam = self.cam_list[0]

            # read camera device information
            self.nodemap_tldevice = self.cam.GetTLDeviceNodeMap()
            self.nodemap_tlstream = self.cam.GetTLStreamNodeMap()
            # initialize camera
            self.cam.Init()

            # read camera control information
            self.nodemap = self.cam.GetNodeMap()

            # enable auto exposure
            self.set_auto_exposure(False)
            self.set_exp(500)

            # get height and width of the field of view
            self.height = self.get_height()
            self.width = self.get_width()

        except pyspin.SpinnakerException as ex:
            print("Error: %s" % ex)
            return None

    def start(self):
        '''
        Start the continuous acquisition mode
        '''
        try:
            # get handle for acquisition mode
            node_acquisition_mode = pyspin.CEnumerationPtr(self.nodemap.GetNode("AcquisitionMode"))
            if not pyspin.IsAvailable(node_acquisition_mode) or not pyspin.IsWritable(node_acquisition_mode):
                print("Unable to set acquisition mode to continuous (enum retrieval). Aborting...")
                return False

            # Retrieve entry node from enumeration node
            node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName("Continuous")
            if not pyspin.IsAvailable(node_acquisition_mode_continuous) or not pyspin.IsReadable(
                    node_acquisition_mode_continuous):
                print("Unable to set acquisition mode to continuous (entry retrieval). Aborting...")
                return False

            # Retrieve integer value from entry node
            acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

            # Set integer value from entry node as new value of enumeration node
            node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

            # Begin Acquisition
            self.cam.BeginAcquisition()

        except pyspin.SpinnakerException as ex:
            print("Error: %s" % ex)

    def stop(self):
        '''
        stop the continuous acquisition mode
        '''
        try:
            self.cam.EndAcquisition()
        except pyspin.SpinnakerException as ex:
            print("Error: %s" % ex)

    def close(self):
        '''
        close the camera instance and delete itself
        '''
        try:
            # release the devices properly
            self.cam.DeInit()
            num_cam = self.cam_list.GetSize()
            num_init = 0
            for i in range(num_cam):
                if self.cam_list.GetByIndex(i).IsInitialized():
                    num_init += 1

            if num_init > 0:
                print('Camera system still in use, removing camera')
                del self.cam
            else:
                print('Camera system not in use, removing camera and shutting down system')
                del self.cam
                self.cam_list.Clear()
                self.system.ReleaseInstance()
                del self.cam_list
                del self.system
        except pyspin.SpinnakerException as ex:
            print("Error: %s" % ex)

    '''
    Data operations
    '''

    def read(self):
        '''
        read and return the next frame from the camera
        '''
        image = self.cam.GetNextImage()
        if image.IsIncomplete():
            print('Image incomplete with image status %d ...' % image.GetImageStatus())
            #return False
        else:
            image_converted = image.Convert(pyspin.PixelFormat_Mono8, pyspin.HQ_LINEAR)
            image.Release()
            return image_converted

    def to_numpy(self, image):
        '''
        Convert an image object to data
        There is a internal bug with the pyspin driver,
        The program have a chance to crash when the camera drops frame
        Use with caution

        2018/04/04 - I am contacting FLIR for this bug

        image: image object to get data from
        return: numpy array containing image data if collection is successful
        otherwise return an array of 1s
        '''
        status = image.GetImageStatus()
        if not status == 0:
            print('corrupted image %i' % status)
            return np.ones((self.height, self.width), dtype=np.uint8)
        buffer_size = image.GetBufferSize()
        if buffer_size == 0:
            print('corrupted image %i' % buffer_size)
            return np.ones((self.height, self.width), dtype=np.uint8)
        if image.IsIncomplete():
            print('incomplete iamge, returning ones')
            return np.ones((self.height, self.width), dtype=np.uint8)
        try:
            data = image.GetData()
            if type(data) == np.ndarray:
                new_data = np.copy(data)
                if new_data.size == (self.height * self.width):
                    output_data = new_data.reshape((self.height, self.width))
                    return output_data
                else:
                    print(status)
                    print('Error: Data size %i is not the right size, returning ones' % new_data.size)
                    return np.ones((self.height, self.width), dtype=np.uint8)
            else:
                print(status)
                print('Error: data is %s, returning ones' % type(data))
                return np.ones((self.height, self.width), dtype=np.uint8)
        except pyspin.SpinnakerException as ex:
            print("Error: %s, returning ones" % ex)
            return np.ones((self.height, self.width), dtype=np.uint8)
        except Exception as ex:
            print("Error: %s, returning ones, exception" % ex)
            return np.ones((self.height, self.width), dtype=np.uint8)

    def save_image(self, image):
        '''
        Save current image to a JPEG file
        image: image object
        '''
        image.Save('C:/Users/lambc/Desktop/save.jpeg')

    '''
    Setting Functions
    '''

    def get_model(self):
        """
        This function get the model name
        """
        try:
            node_device_information = pyspin.CCategoryPtr(self.nodemap_tldevice.GetNode("DeviceInformation"))

            if pyspin.IsAvailable(node_device_information) and pyspin.IsReadable(node_device_information):
                features = node_device_information.GetFeatures()
                for feature in features:
                    node_feature = pyspin.CValuePtr(feature)
                    if node_feature.GetName() == 'DeviceModelName':
                        return node_feature.ToString()

            else:
                return 'N/A'

        except pyspin.SpinnakerException as ex:
            print("Error: %s" % ex)
            return 'N/A'

    def get_width(self):
        try:
            node_width = pyspin.CIntegerPtr(self.nodemap.GetNode("Width"))
            if pyspin.IsAvailable(node_width):
                return node_width.GetValue()
        except pyspin.SpinnakerException as ex:
            print("Error: %s" % ex)

    def get_height(self):
        try:
            node_height = pyspin.CIntegerPtr(self.nodemap.GetNode("Height"))
            if pyspin.IsAvailable(node_height):
                return node_height.GetValue()
        except pyspin.SpinnakerException as ex:
            print("Error: %s" % ex)

    def set_frame_rate(self, fr):
        '''
        set frame rate in fps

        fr: framerate in fps
        '''
        try:
            return self.cam.AcquisitionFrameRate.SetValue(fr)

        except pyspin.SpinnakerException as ex:
            print("Error: %s" % ex)


    def get_frame_rate(self):
        '''
        get frame rate in fps
        '''
        try:
            return self.cam.AcquisitionFrameRate.GetValue()

        except pyspin.SpinnakerException as ex:
            print("Error: %s" % ex)





    def get_exp_min(self):
        '''
        get min exposure time in microseconds
        '''
        try:
            return self.cam.ExposureTime.GetMin()

        except pyspin.SpinnakerException as ex:
            print("Error: %s" % ex)

    def get_exp_max(self):
        '''
        get max exposure time in microseconds
        '''
        try:
            return self.cam.ExposureTime.GetMax()

        except pyspin.SpinnakerException as ex:
            print("Error: %s" % ex)

    def get_exp(self):
        '''
        get exposure time in microseconds
        '''
        try:
            return self.cam.ExposureTime.GetValue()

        except pyspin.SpinnakerException as ex:
            print("Error: %s" % ex)

    def set_exp(self, exp_time):
        '''
        set exposure time in microseconds

        exp_time: exposure time in microseconds
        '''
        try:
            if self.cam.ExposureTime.GetAccessMode() != pyspin.RW:
                print("Unable to set exposure time. Aborting...")
                return None

            exp_time = min(exp_time, self.get_exp_max())
            exp_time = max(exp_time, self.get_exp_min())
            self.cam.ExposureTime.SetValue(exp_time)


        except pyspin.SpinnakerException as ex:
            print("Error: %s" % ex)




    def get_auto_exposure(self):
        '''
        get the status of auto exposure, either on or off
        '''
        try:
            val = self.cam.ExposureAuto.GetValue()

            if val == 2:
                return True
            elif val == 0:
                return False
            else:
                print('Unable to get auto exposure setting')

        except pyspin.SpinnakerException as ex:
            print("Error: %s" % ex)



    def set_auto_exposure(self, mode):
        '''
        set the status of auto exposure, either on or off

        mode: boolean value of True(on) or False(off)
        '''
        try:
            if self.cam.ExposureAuto.GetAccessMode() != pyspin.RW:
                print("Unable to enable automatic gain (node retrieval). Non-fatal error...")
                return None

            if mode:
                self.cam.ExposureAuto.SetValue(pyspin.ExposureAuto_Continuous)
                #print('Enable auto exposure')
            else:
                self.cam.ExposureAuto.SetValue(pyspin.ExposureAuto_Off)
                #print('Disable auto exposure')


        except pyspin.SpinnakerException as ex:
            print("Error: %s" % ex)

    def get_auto_gain(self):
        '''
        get the status of auto exposure, either on or off
        '''
        try:
            val = self.cam.GainAuto.GetValue()

            if val == 2:
                return True
            elif val == 0:
                return False
            else:
                print('Unable to get auto gain setting')

        except pyspin.SpinnakerException as ex:
            print("Error: %s" % ex)


    def set_auto_gain(self, mode):
        '''
        set the status of auto exposure, either on or off

        mode: boolean value of True(on) or False(off)
        '''
        try:
            if self.cam.GainAuto.GetAccessMode() != pyspin.RW:
                print("Unable to enable automatic gain (node retrieval). Non-fatal error...")
                return None

            if mode:
                self.cam.GainAuto.SetValue(pyspin.GainAuto_Continuous)
                #print('Enable auto gain')
            else:
                self.cam.GainAuto.SetValue(pyspin.GainAuto_Off)
                #print('Disable auto gain')

        except pyspin.SpinnakerException as ex:
            print("Error: %s" % ex)

    def get_gain(self):
        '''
        get gain
        '''
        try:
            return self.cam.Gain.GetValue()

        except pyspin.SpinnakerException as ex:
            print("Error: %s" % ex)

    def get_gain_min(self):
        '''
        get min gain time in microseconds
        '''
        try:
            return self.cam.Gain.GetMin()

        except pyspin.SpinnakerException as ex:
            print("Error: %s" % ex)

    def get_gain_max(self):
        '''
        get max gain time in microseconds
        '''
        try:
            return self.cam.Gain.GetMax()

        except pyspin.SpinnakerException as ex:
            print("Error: %s" % ex)

    def set_gain(self, gain):
        '''
        set gain

        '''
        try:
            if self.cam.Gain.GetAccessMode() != pyspin.RW:
                print("Unable to set gain. Aborting...")
                return None

            gain = min(gain, self.get_gain_max())
            gain = max(gain, self.get_gain_min())
            self.cam.Gain.SetValue(gain)



        except pyspin.SpinnakerException as ex:
            print("Error: %s" % ex)

    def get_video_mode(self):
        '''
        get the video mode of the camera
        '''
        try:
            node_video_mode = pyspin.CEnumerationPtr(self.nodemap.GetNode("VideoMode"))
            return int(node_video_mode.GetCurrentEntry().GetSymbolic()[4])
        except pyspin.SpinnakerException as ex:
            print("Error: %s" % ex)

    def set_video_mode(self, mode_number):
        '''
        set the video mode of the camera, depends on the model, certain mode
        might not exist

        mode_number: integer number of the video mode
        '''
        try:
            node_video_mode = pyspin.CEnumerationPtr(self.nodemap.GetNode("VideoMode"))
            Mode0 = node_video_mode.GetEntryByName("Mode0")
            Mode1 = node_video_mode.GetEntryByName("Mode1")
            Mode2 = node_video_mode.GetEntryByName("Mode2")
            mode_list = [Mode0, Mode1, Mode2]

            node_video_mode.SetIntValue(mode_list[mode_number].GetValue())
        except pyspin.SpinnakerException as ex:
            print("Error: %s" % ex)

    '''
    Streaming information
    '''

    def get_feature(self, nodemap, node_name, feature_name):
        '''
        method to get any stt from a camera

        nodemap: the node map of a collection of camera properties,
                e.g. TLDEVICE

        node_name: Name of the specific node, such as DeviceInformation

        feature_name: Name of the specific feature, such as ModelNumber
        '''
        try:
            node = pyspin.CCategoryPtr(nodemap.GetNode(node_name))
            if pyspin.IsAvailable(node) and pyspin.IsReadable(node):
                features = node.GetFeatures()
                for feature in features:
                    node_feature = pyspin.CValuePtr(feature)
                    if node_feature.GetName() == feature_name:
                        return node_feature.ToString()
            else:
                print('No feature named %s found' % feature_name)
                return None
        except pyspin.SpinnakerException as ex:
            print("Error: %s" % ex)

    def set_feature(self, nodemap, node_name, feature_name, value):
        '''
        method to get any stt from a camera

        nodemap: the node map of a collection of camera properties,
                e.g. TLDEVICE

        node_name: Name of the specific node, such as DeviceInformation

        feature_name: Name of the specific feature, such as ModelNumber
        '''
        try:
            node = pyspin.CCategoryPtr(nodemap.GetNode(node_name))
            if pyspin.IsAvailable(node) and pyspin.IsReadable(node):
                features = node.GetFeatures()
                for feature in features:
                    node_feature = pyspin.CValuePtr(feature)
                    if node_feature.GetName() == feature_name:
                        node_feature.FromString(value)

            else:
                print('No feature named %s found' % feature_name)
                return None
        except pyspin.SpinnakerException as ex:
            print("Error: %s" % ex)

    def get_buffer_count(self):
        """
        This function get the buffer count of the stream
        """
        return int(self.get_feature(self.nodemap_tlstream,
                                    'BufferHandlingControl',
                                    'StreamDefaultBufferCount'))

    def set_buffer_count(self, value):
        """
        This function set the buffer count of the stream
        """
        '''return self.set_feature(self.nodemap_tlstream,
                                'BufferHandlingControl',
                                'StreamDefaultBufferCount',
                                str(value))'''
        s_node_map=self.nodemap_tlstream
        handling_mode = pyspin.CEnumerationPtr(s_node_map.GetNode('StreamBufferHandlingMode'))
        if not pyspin.IsAvailable(handling_mode) or not pyspin.IsWritable(handling_mode):
            print('Unable to set Buffer Handling mode (node retrieval). Aborting...\n')
            return False
        handling_mode_entry = handling_mode.GetEntryByName('NewestOnly')
        handling_mode.SetIntValue(handling_mode_entry.GetValue())
        print('\n\nBuffer Handling Mode has been set to %s' % handling_mode_entry.GetDisplayName())

if __name__=='__main__':
    cam=FLIRCamDev()
    cam.start()
    #img=cam.read()
    #img=cam.to_numpy(img)
    #print(img)
    cam.set_buffer_count(1)
    #cam.set_gain(0)
    cam.stop()
    cam.close()