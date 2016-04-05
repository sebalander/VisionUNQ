# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 12:53:05 2016

@author: Lily
"""

from FishEyeCamera import FishEyeCamera
from PTZCamera import PTZCamera
import logging


class PTZCamera(IPCamera):

    def __init__(self, host, port, user, passwd):
        IPCamera.__init__(self, host, port, user, passwd)
        self.ptzService = self.create_ptz_service()


    def getStreamUri(self):
#        return self.mediaService.GetStreamUri()[0]
        return 'rtsp://10.2.1.49:554/Streaming/Channels/1?transportmode=unicast&profile=Profile_1'
        
        
    def getStatus(self):
        media_profile = self.mediaService.GetProfiles()[0]
        
        request = self.ptzService.create_type('GetStatus')
        request.ProfileToken = media_profile._token
        
        ptzStatus = self.ptzService.GetStatus(request)
        pan = ptzStatus.Position.PanTilt._x
        tilt = ptzStatus.Position.PanTilt._y
        zoom = ptzStatus.Position.Zoom._x

        return (pan, tilt, zoom)
        
        
    def moveRight(self):
        status = self.getStatus()
        print "Movimiento hacia derecha desde " + str(status)
        actualPan = status[0]
        actualTilt = status[1]
        
        media_profile = self.mediaService.GetProfiles()[0]
        
        request = self.ptzService.create_type('AbsoluteMove')
        request.ProfileToken = media_profile._token
        pan = actualPan - float(2)/360
        if pan <= -1:
            pan = 1

        request.Position.PanTilt._x = pan
        request.Position.PanTilt._y = actualTilt
        absoluteMoveResponse = self.ptzService.AbsoluteMove(request)
        
    def moveLeft(self):
        status = self.getStatus()
        print "Movimiento hacia izquierda desde " + str(status)
        actualPan = status[0]
        actualTilt = status[1]
        
        media_profile = self.mediaService.GetProfiles()[0]
        
        request = self.ptzService.create_type('AbsoluteMove')
        request.ProfileToken = media_profile._token
        pan =  round(actualPan + float(2)/360 , 6)
        if pan >= 1:
            pan = -1
        print pan
        request.Position.PanTilt._x = pan
        request.Position.PanTilt._y = actualTilt
        absoluteMoveResponse = self.ptzService.AbsoluteMove(request)
        
        
    def moveUp(self):
        status = self.getStatus()
        print "Movimiento hacia arriba desde " + str(status)
        actualPan = status[0]
        actualTilt = status[1]
        
        media_profile = self.mediaService.GetProfiles()[0]
        
        request = self.ptzService.create_type('AbsoluteMove')
        request.ProfileToken = media_profile._token
        tilt =  round(actualTilt - float(2)/90, 6)
        pan = actualPan
        if tilt <= -1:
            tilt = -1
            pan = actualPan
        elif tilt >= 1:
                tilt = 1
                pan = actualPan + 180*float(2)/360
                
        request.Position.PanTilt._x = pan
        request.Position.PanTilt._y = tilt
        absoluteMoveResponse = self.ptzService.AbsoluteMove(request)
        
        
        
    def moveDown(self):
        status = self.getStatus()
        print "Movimiento hacia abajo desde " + str(status)
        actualPan = status[0]
        actualTilt = status[1]
        
        media_profile = self.mediaService.GetProfiles()[0]
        
        request = self.ptzService.create_type('AbsoluteMove')
        request.ProfileToken = media_profile._token
        tilt = round(actualTilt + float(2)/90, 6)
        pan = actualPan
        if tilt <= -1:
            tilt = -1
            pan = actualPan
        elif tilt >= 1:
                tilt = 1
                pan = actualPan + 180*float(2)/360

        request.Position.PanTilt._x = pan
        request.Position.PanTilt._y = tilt
        absoluteMoveResponse = self.ptzService.AbsoluteMove(request)
        
        
    def zoomIn(self):
        status = self.getStatus()
        print "Zoom in desde " + str(status)
        media_profile = self.mediaService.GetProfiles()[0]
        
        request = self.ptzService.create_type('AbsoluteMove')
        request.ProfileToken = media_profile._token
        
        if status[2] < 0.05:
            paso = 0.07
        else:
            paso = 0.035
            
        pZoom = status[2] + paso
        if pZoom > 1:
            pZoom = 1
        
        request.Position.Zoom._x = pZoom
        absoluteMoveResponse = self.ptzService.AbsoluteMove(request)
        
    def zoomOut(self):
        status = self.getStatus()
        print "Zoom out desde " + str(status)
        media_profile = self.mediaService.GetProfiles()[0]
        
        request = self.ptzService.create_type('AbsoluteMove')
        request.ProfileToken = media_profile._token
        
        pZoom = status[2] - 0.01    # Con este paso anda bien
        if pZoom < 0:
            pZoom = 0

        request.Position.Zoom._x = pZoom
        absoluteMoveResponse = self.ptzService.AbsoluteMove(request)
        
    def moveAbsolute(self, pan, tilt, zoom = 0):
        media_profile = self.mediaService.GetProfiles()[0]
        
        request = self.ptzService.create_type('AbsoluteMove')
        request.ProfileToken = media_profile._token
        
        pPan = round(1 - float(pan)/180, 6)
        pTilt = round(1 - float(tilt)/45, 6)
        pZoom = round(float(zoom/100), 6)
       
        request.Position.PanTilt._x = pPan
        request.Position.PanTilt._y = pTilt
        request.Position.Zoom._x = pZoom
        absoluteMoveResponse = self.ptzService.AbsoluteMove(request)
        
        
    def setHomePosition(self):
        media_profile = self.mediaService.GetProfiles()[0]
        
        request = self.ptzService.create_type('SetHomePosition')
        request.ProfileToken = media_profile._token
        self.ptzService.SetHomePosition(request)
        
    def gotoHomePosition(self):
        media_profile = self.mediaService.GetProfiles()[0]
        
        request = self.ptzService.create_type('GotoHomePosition')
        request.ProfileToken = media_profile._token
        self.ptzService.GotoHomePosition(request)
        
    def getSnapshotUri(self):
        media_profile = self.mediaService.GetProfiles()[0]
        
        request = self.mediaService.create_type('GetSnapshotUri')
        request.ProfileToken = media_profile._token
        response = self.mediaService.GetSnapshotUri(request)
        
        print response.Uri
        urllib.urlretrieve("http://10.2.1.49/onvif-http/snapshot", "local-filename.jpeg")
                    
   


class IPCamera(ONVIFCamera):
    
    def __init__(self, host, port ,user, passwd):
        ONVIFCamera.__init__(self, host, port, user, passwd)
        self.mediaService = self.create_media_service()
        self.deviceService = self.create_devicemgmt_service()
        
        self.capture = None
        
    def getTimestamp(self):
        request = self.devicemgmt.create_type('GetSystemDateAndTime')
        response = self.devicemgmt.GetSystemDateAndTime(request)

        year = response.LocalDateTime.Date.Year
        month = response.LocalDateTime.Date.Month
        day = response.LocalDateTime.Date.Day
        
        hour = response.LocalDateTime.Time.Hour
        minute = response.LocalDateTime.Time.Minute
        second = response.LocalDateTime.Time.Second
        
        print str(year) + "/" + self.formatTimePart(month) + "/" + self.formatTimePart(day)
        print self.formatTimePart(hour) + ":" + self.formatTimePart(minute) + ":" + self.formatTimePart(second)
        
    def setTimestamp(self):
        request = self.devicemgmt.create_type('SetSystemDateAndTime')
#        request.DateTimeType = 'Manual'
#        request.UTCDateTime.Time.Hour = 11
        
        response = self.devicemgmt.SetSystemDateAndTime(request)
        
        print response

      
        
        
    def formatTimePart(self, param):
        sParam = str(param)
        if len(sParam) == 1:
            return "0" + sParam
        else:
            return sParam



class FishEyeCamera(IPCamera):
    
     def __init__(self, host, port ,user, passwd):
        IPCamera.__init__(self, host, port, user, passwd)
        

        
     def getStreamUri(self):
#        media_service = self.create_media_service()
#        return media_service.GetStreamUri()
        return 'rtsp://10.2.1.48/live.sdp'
        
  
        
#     def test(self):
         
#        media_service = self.create_media_service()
#        print media_service.GetStreamUri()
#         profiles = self.devicemgmt.GetProfiles()
#         print str(profiles)
         
#         
#         resp1 = self.devicemgmt.GetCapabilities()
#         print 'My camera`s hostname: ' + str(resp1)
##         print 'My camera`s hostname: ' + str(resp1.Media.XAddr)
#         
#         resp3 = self.devicemgmt.GetServices({'IncludeCapability': True})
#         print 'My camera`s hostname: ' + str(resp3)
         
#         
#         resp2 = self.devicemgmt.GetNetworkProtocols()
#         print 'My camera`s hostname: ' + str(resp2)
#         resp = self.devicemgmt.GetHostname()
#         print 'My camera`s hostname: ' + str(resp.Name)
#         
#         dt = self.devicemgmt.GetSystemDateAndTime()
#         tz = dt.TimeZone
#         year = dt.UTCDateTime.Date.Year
#         hour = dt.UTCDateTime.Time.Hour
#         print str(tz) + ' ' + str(year) + ' ' + str(hour)
#
#         di = self.devicemgmt.GetDeviceInformation()
#         print 'Device Information: ' + str(di)
#         
#         su = self.devicemgmt.GetWsdlUrl()
#         print 'Uris: ' + str(su)
         


class CameraManager():
    
    def __init__(self, fe=True, ptz=True):
        self.feCam = None
        self.ptzCam = None
        if fe:
            self.feCam = self.createFishEyeCamera()
        if ptz:
            self.ptzCam = self.createPTZCamera()
        
        self.capture = None
    
    def createFishEyeCamera(self):
        logging.info('Se crea camara fe con parametros por defecto')
        #feHost = '10.9.6.48' #labo 127
        feHost = '10.2.1.48' #labo 126
        fePort = 80
        feUser = 'admin'
        fePwd = '12345'
        try:
            feCam = FishEyeCamera(feHost, fePort, feUser, fePwd)
            logging.info('Creacion de camara fe: OK')
        except:
            feCam = None
            logging.error('No se encontro una camara con estos parametros')
        return feCam
        
        
    def createPTZCamera(self):
        logging.info('Se crea camara ptz con parametros por defecto')
        ##ptzHost = "10.9.6.49" #labo 127
        ptzHost = '10.2.1.49' #labo 126
        ptzPort = 80
        ptzUser = 'admin'
        ptzPwd = '12345'
        try:
            ptzCam = PTZCamera(ptzHost, ptzPort, ptzUser, ptzPwd)
            logging.info('Creacion de camara ptz: OK')
        except:
            ptzCam = None
            logging.error('No se encontro una camara con estos parametros')
        return ptzCam

