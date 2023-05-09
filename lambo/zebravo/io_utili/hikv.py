from lambo.zebravo.io_utili.fetch import Fetcher
import cv2
import time
import msvcrt
from lambo.zebravo.io_utili.drivers.hik.MvCameraControl_class import *
from ctypes import *
import numpy as np

class HIkVFetcher(Fetcher):

  def __init__(self, max_len=20):
    super(HIkVFetcher, self).__init__(max_len)
    self.camera = MvCamera()
    # self.camera.set_frame_rate(60)


  def _init(self):

    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

    # ch:枚举设备 | en:Enum device
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0:
      print("enum devices fail! ret[0x%x]" % ret)
      sys.exit()

    if deviceList.nDeviceNum == 0:
      print("find no device!")
      sys.exit()

    print("Find %d devices!" % deviceList.nDeviceNum)

    for i in range(0, deviceList.nDeviceNum):
      mvcc_dev_info = cast(deviceList.pDeviceInfo[i],
                           POINTER(MV_CC_DEVICE_INFO)).contents
      if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
        print("\ngige device: [%d]" % i)
        strModeName = ""
        for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
          strModeName = strModeName + chr(per)
        print("device model name: %s" % strModeName)

        nip1 = ((
                    mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
        nip2 = ((
                    mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
        nip3 = ((
                    mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
        nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
        print("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
      elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
        print("\nu3v device: [%d]" % i)
        strModeName = ""
        for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
          if per == 0:
            break
          strModeName = strModeName + chr(per)
        print("device model name: %s" % strModeName)

        strSerialNumber = ""
        for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
          if per == 0:
            break
          strSerialNumber = strSerialNumber + chr(per)
        print("user serial number: %s" % strSerialNumber)

    nConnectionNum = 0

    if int(nConnectionNum) >= deviceList.nDeviceNum:
      print("intput error!")
      sys.exit()

    # ch:创建相机实例 | en:Creat Camera Object
    cam = MvCamera()

    # ch:选择设备并创建句柄 | en:Select device and create handle
    stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)],
                        POINTER(MV_CC_DEVICE_INFO)).contents

    ret = cam.MV_CC_CreateHandle(stDeviceList)
    if ret != 0:
      print("create handle fail! ret[0x%x]" % ret)
      sys.exit()

    # ch:打开设备 | en:Open device
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
      print("open device fail! ret[0x%x]" % ret)
      sys.exit()

    # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
    if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
      nPacketSize = cam.MV_CC_GetOptimalPacketSize()
      if int(nPacketSize) > 0:
        ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
        if ret != 0:
          print("Warning: Set Packet Size fail! ret[0x%x]" % ret)
      else:
        print("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

    stBool = c_bool(False)
    ret = cam.MV_CC_GetBoolValue("AcquisitionFrameRateEnable", stBool)
    if ret != 0:
      print("get AcquisitionFrameRateEnable fail! ret[0x%x]" % ret)
      sys.exit()

    # ch:设置触发模式为off | en:Set trigger mode as off
    ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    if ret != 0:
      print("set trigger mode fail! ret[0x%x]" % ret)
      sys.exit()
    self.camera = cam
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
      print ("start grabbing fail! ret[0x%x]" % ret)
      sys.exit()
    print('Hikvision Camera Initialized')

  def _loop(self):
    stOutFrame = MV_FRAME_OUT()
    memset(byref(stOutFrame), 0, sizeof(stOutFrame))
    ret = self.camera.MV_CC_GetImageBuffer(stOutFrame, 1000)
    pData = (
          c_ubyte * stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight)()
    cdll.msvcrt.memcpy(byref(pData), stOutFrame.pBufAddr,
                       stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight)
    data = np.frombuffer(pData, count=int(
      stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight),
                         dtype=np.uint8)
    image = data.reshape((stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth))
    self.append_to_buffer(image)
    nRet = self.camera.MV_CC_FreeImageBuffer(stOutFrame)
    time.sleep(0)


  def _finalize(self):
    pass