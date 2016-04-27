# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 19:29:26 2014

funciones
- gui_openFile
- gui_select

clases
- ImageWindow
- PanelFrame
- ParameterSlider
- PanelButtons
- ButtonsBFS
- ButtonsROI
- Buttons4pt
- ButtonsOcl
- ButtonsRun
- PanelSelect
- PanelMain
- ImageEvent
- ImageIn
- videoThread
- runVideoThread

@author: jew
"""

import wx
import numpy
import threading
import time
import mainYabo
import os
import cv2

EVT_NEW_IMAGE = wx.PyEventBinder(wx.NewEventType(), 0)

no_resize = wx.DEFAULT_FRAME_STYLE & ~ (wx.RESIZE_BORDER |
                                        wx.MAXIMIZE_BOX)


# ----------------------------------------------------------------------
def gui_openFile(parent, message):
    # Parte gráfica
    dlg = wx.FileDialog(parent, message, "../resources/", "", "*.*")
    if dlg.ShowModal() == wx.ID_OK:
        path = dlg.GetPath()
        mypath = os.path.basename(path)

    dlg.Destroy()
    return path


def gui_select(parent, message, listElements):
    # Parte gráfica
    dlg = wx.SingleChoiceDialog(parent, message, '',
                                listElements, wx.CHOICEDLG_STYLE)
    if dlg.ShowModal() == wx.ID_OK:
        index = listElements.index(dlg.GetStringSelection())
        dlg.Destroy()

    return index

# ----------------------------------------------------------------------

class ImageWindow(wx.Frame):
    def __init__(self, vg, parent, id=-1, style=no_resize):
        self.title = vg.appName
        self.vg = vg

        wx.Frame.__init__(self, parent, id, title=self.title,
                          size=(243, 220), style=style)

        self.panel = []

        # Frame contiene a todos los demas
        self.panel.append(PanelFrame(self))             # 0
        self.panel.append(PanelSelect(self.panel[0]))   # 1
        self.panel.append(PanelMain(self.panel[0]))     # 2
        self.panel.append(PanelButtons(self.panel[0]))  # 3

        self.panel[0].Show()
        self.panel[1].Show()
        self.panel[2].Hide()
        self.panel[3].Hide()

        self.Bind(wx.EVT_CLOSE, self.OnClose)

        self.Show()
        self.Center()

    def OnClose(self, event):
        dlg = wx.MessageDialog(self,
            'Desea salir de %s' % self.title,
            '', wx.OK|wx.CANCEL|wx.ICON_QUESTION)
        result = dlg.ShowModal()
        dlg.Destroy()
        if result == wx.ID_OK:
            self.vg.saveStuff()
            self.vg.mustExec = False
            self.Destroy()

    def goToMain(self):
        self.vg.setSpeed(0.8)
        
        self.panel[1].Hide()
        self.panel[2].Show()
        self.panel[3].Show()

        sx = numpy.max([self.vg.captura.sx,self.vg.windowMinWidth])
        sy = numpy.max([self.vg.captura.sy,self.vg.windowMinHeight])

        self.setFrameSize(sx, sy, self.vg.intBH)

        self.stopBusyBox()

    def startBusyBox(self, message):
        self.busyBox = wx.BusyInfo(message, self)
        self.Hide()

    def stopBusyBox(self):
        del self.busyBox
        self.Show()

    def setFrameSize(self, sx, sy, intBH):
        # Tiene varios sleep para no tener problemas con
        # el multithreading de la ventana
        self.SetSize((sx,sy + intBH))
        self.panel[0].SetSize((sx,sy + intBH))
        self.panel[2].SetSize((sx,sy))
        self.panel[3].SetSize((sx,intBH))
        self.panel[3].SetPosition((0,sy))
        self.Center()

    def showButton(self, index):
        self.panel[3].showButton(index)

#----------------------------------------------------------------------

class PanelFrame(wx.Panel):
    def __init__(self,parent):
        wx.Panel.__init__(self, parent)
        self.parent = parent
        self.vg = parent.vg

#----------------------------------------------------------------------

class ParameterSlider(wx.Slider):
    def __init__(self, parent, parameter, minV, maxV, scale):
        wx.Slider.__init__(self,
                           parent,
                           size=wx.DefaultSize,
                           style=wx.SL_LABELS)

        self.vg = parent.vg
        self.parameter = parameter
        self.scale = scale

        value = getattr(self.vg.pv, parameter)

        self.SetValue(value*scale)
        self.SetMin(minV*scale)
        self.SetMax(maxV*scale)

    def OnScroll(self, e):
        value = self.GetValue()
        if self.scale != 1: value = value/self.scale
        setattr(self.vg.pv, self.parameter, value)
        if self.parameter == 'ksize1':
            self.vg.pv.createKernel()
        elif self.parameter[:3] == 'mog':
            self.vg.pv.createBgFilter()

#----------------------------------------------------------------------

class PanelButtons(wx.Panel):
    def __init__(self,parent):
        self.vg = parent.vg
        self.parent = parent
        wx.Panel.__init__(self,
                          parent,
                          size=(self.vg.intBH,self.vg.intBH))

        self.panel = []
        self.panel.append(ButtonsROI(self))
        self.panel.append(Buttons4pt(self))
        self.panel.append(ButtonsBFS(self))
        self.panel.append(ButtonsOcl(self))
        self.panel.append(ButtonsRun(self))

        self.showButton(0)

        self.BackgroundColour = wx.BLACK

    def showButton(self, index):
        for i in range(0,len(self.panel)):
            if i==index:
                self.panel[i].Show()
            else:
                self.panel[i].Hide()

class ButtonsBFS(wx.Panel):
    def __init__(self,parent):
        self.parent = parent
        self.vg = parent.vg
        wx.Panel.__init__(self, parent, size=(800,self.vg.intBH))
        self.BackgroundColour = wx.BLACK
        self.ForegroundColour = wx.WHITE

        self.p = []
        self.p.append(ParameterSlider(self, \
        'mog_learningRate', 0.01, 0.99, 100.))

        self.p.append(ParameterSlider(self, \
        'mog_history', 50, 99, 1))

        self.p.append(ParameterSlider(self, \
        'mog_nmixtures', 1, 50, 1))

        self.p.append(ParameterSlider(self, \
        'ksize1', 5, 25, 1))

        self.p.append(ParameterSlider(self, \
        'playSpeed', 0.01, 1., 100.))

        for i in range(0,len(self.p)):
            self.p[i].Bind(wx.EVT_SCROLL, self.p[i].OnScroll)

        b = wx.Button(self, -1, "Ok", (30,50))

        self.sizer = wx.GridSizer(3,8,0,0)
        self.sizer.Add(wx.StaticText(self, \
        label='  Ajuste los parámetros para una detección óptima. '+\
        'Cuando finalice haga click en Ok.'), 0,\
        wx.ALIGN_CENTRE_VERTICAL|wx.ALIGN_LEFT)
        for i in range(0,7): self.sizer.Add((0,0), 0)
        self.sizer.Add(wx.StaticText(self, label='learningRate'), 0, wx.ALIGN_CENTER)
        self.sizer.Add(self.p[0], 0, wx.EXPAND)
        self.sizer.Add(wx.StaticText(self, label='history'), 0, wx.ALIGN_CENTER)
        self.sizer.Add(self.p[1], 0, wx.EXPAND)
        self.sizer.Add(wx.StaticText(self, label='nmixtures'), 0, wx.ALIGN_CENTER)
        self.sizer.Add(self.p[2], 0, wx.EXPAND)
        self.sizer.Add((0,0), 0)
        self.sizer.Add(b, 0, wx.ALIGN_CENTER)
        self.sizer.Add(wx.StaticText(self, label='k size'), 0, wx.ALIGN_CENTER)
        self.sizer.Add(self.p[3], 0, wx.EXPAND)
        self.sizer.Add(wx.StaticText(self, label='velocidad'), 0, wx.ALIGN_CENTER)
        self.sizer.Add(self.p[4], 0, wx.EXPAND)
        self.SetSizer(self.sizer)

        self.Bind(wx.EVT_BUTTON, self.OnOk, b)

    def OnOk(self, event):
        self.vg.switchToMode(5)

class ButtonsROI(wx.Panel):
    def __init__(self, parent):
        self.parent = parent
        self.vg = parent.vg
        wx.Panel.__init__(self, parent, size=(800,self.vg.intBH))
        self.BackgroundColour = wx.BLACK
        self.ForegroundColour = wx.WHITE

        b1 = wx.Button(self, -1, "Ok", (30,50))
        b2 = wx.Button(self, -1, "Limpiar", (30,50))

        self.sizer = wx.GridSizer(2,2,2,2)
        self.sizer.Add(wx.StaticText(self, 0, \
        '  Por favor, seleccione una región de interés '+ \
        '(ROI) que desee analizar.'), 0, \
        wx.ALIGN_CENTRE_VERTICAL | wx.ALIGN_LEFT)
        self.sizer.Add(b1, 0, wx.ALIGN_CENTER)
        self.sizer.Add((0,0), 0)
        self.sizer.Add(b2, 0, wx.ALIGN_CENTER)

        self.SetSizer(self.sizer)

        self.Bind(wx.EVT_BUTTON, self.OnOk, b1)
        self.Bind(wx.EVT_BUTTON, self.OnClean, b2)

    def OnOk(self, event):
        if self.vg.mouseRoi[2] != ():
            self.vg.switchToMode(3)

    def OnClean(self, event):
        self.vg.mouseRoi = [(),(),()]

class Buttons4pt(wx.Panel):
    def __init__(self, parent):
        self.parent = parent
        self.vg = parent.vg
        wx.Panel.__init__(self, parent, size=(800,self.vg.intBH))
        self.BackgroundColour = wx.BLACK
        self.ForegroundColour = wx.WHITE

        b1 = wx.Button(self, -1, "Ok", (30,50))
        b2 = wx.Button(self, -1, "Limpiar", (30,50))

        self.sizer = wx.GridSizer(2,2,2,2)
        (self.sizer.Add(wx.StaticText(self, 0, 
        '  Por favor, elija 4 puntos que delimiten '+ 
        'una mayor cantidad de autopista.'), 0, 
        wx.ALIGN_CENTRE_VERTICAL | wx.ALIGN_LEFT))
        self.sizer.Add(b1, 0, wx.ALIGN_CENTER)
        self.sizer.Add((0,0), 0)
        self.sizer.Add(b2, 0, wx.ALIGN_CENTER)

        self.SetSizer(self.sizer)

        self.Bind(wx.EVT_BUTTON, self.OnOk, b1)
        self.Bind(wx.EVT_BUTTON, self.OnClean, b2)

    def OnOk(self, event):
        if self.vg.mouse4pt[3] != ():
            self.vg.switchToMode(4)

    def OnClean(self, event):
        self.vg.mouse4pt = [(),(),(),()]

class ButtonsOcl(wx.Panel):
    def __init__(self, parent):
        self.parent = parent
        self.vg = parent.vg
        wx.Panel.__init__(self, parent, size=(800,self.vg.intBH))
        self.BackgroundColour = wx.BLACK
        self.ForegroundColour = wx.WHITE

        b1 = wx.Button(self, -1, "<-", (30,50))
        b2 = wx.Button(self, -1, "->", (30,50))
        b3 = wx.Button(self, -1, "Ok", (30,50))

        self.sizer = wx.GridSizer(2,3,2,2)
        self.sizer.Add(wx.StaticText(self, 0, \
        '  Por favor, haga click sobre las oclusiones ocurridas '+ \
        'en la imágen.'), 0, \
        wx.ALIGN_CENTRE_VERTICAL|wx.ALIGN_LEFT)
        for i in range(0,2): self.sizer.Add((0,0), 0)
        self.sizer.Add(b1, 0, wx.ALIGN_CENTER)
        self.sizer.Add(b2, 0, wx.ALIGN_CENTER)
        self.sizer.Add(b3, 0, wx.ALIGN_CENTER)

        self.SetSizer(self.sizer)

        self.Bind(wx.EVT_BUTTON, self.OnPrev, b1)
        self.Bind(wx.EVT_BUTTON, self.OnNext, b2)
        self.Bind(wx.EVT_BUTTON, self.OnOk, b3)

    def OnPrev(self, event):
        self.vg.revK()

    def OnNext(self, event):
        #self.vg.saveLabelInfo()
        #self.vg.mouseLabel = []
        self.vg.addK()

    def OnOk(self, event):
        self.vg.switchToMode(6)

class ButtonsRun(wx.Panel):
    def __init__(self, parent):
        self.parent = parent
        self.vg = parent.vg
        wx.Panel.__init__(self, parent, size=(800,self.vg.intBH))
        self.BackgroundColour = wx.BLACK
        self.ForegroundColour = wx.WHITE

        b1 = wx.Button(self, -1, "a", (30,50))
        b2 = wx.Button(self, -1, "b", (30,50))
        b3 = wx.Button(self, -1, "c", (30,50))
        
        self.sizer = wx.GridSizer(2,3,2,2)
        self.sizer.Add(wx.StaticText(self, 0, ''), 0, \
        wx.ALIGN_CENTRE_VERTICAL|wx.ALIGN_LEFT)
        for i in range(0,2): self.sizer.Add((0,0), 0)
        self.sizer.Add(b1, 0, wx.ALIGN_CENTER)
        self.sizer.Add(b2, 0, wx.ALIGN_CENTER)
        self.sizer.Add(b3, 0, wx.ALIGN_CENTER)
        
        self.SetSizer(self.sizer)        
        
        self.Bind(wx.EVT_BUTTON, self.OnA, b1)
        self.Bind(wx.EVT_BUTTON, self.OnB, b2)
        self.Bind(wx.EVT_BUTTON, self.OnC, b3)
        
    def OnA(self, event):
        self.vg.revK()

    def OnB(self, event):
        self.vg.addK()

    def OnC(self, event):
        self.vg.switchToMode(7)

#----------------------------------------------------------------------

class PanelSelect(wx.Panel):
    def __init__(self,parent):
        wx.Panel.__init__(self, parent, size=(243,220))
        self.parent = parent
        self.vg = parent.vg
        
        #png = wx.Image('logo_unqui.png', wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        #wx.StaticBitmap(self, -1, png, (10, 10), (png.GetWidth(), png.GetHeight()))
        
        b1 = wx.Button(self, -1, "Capturar archivo", (46,114), (150,30))
        b2 = wx.Button(self, -1, "Capturar cámara web", (46,154), (150,30))
        
        self.Bind(wx.EVT_BUTTON, self.OnButtonFile, b1)
        self.Bind(wx.EVT_BUTTON, self.OnButtonWeb, b2)
    
    def OnButtonFile(self, evt):
        strFile = gui_openFile(self, "Seleccione el video")
        
        self.vg.setCaptura(mainYabo.capture.FileCapture(strFile, self.vg.zoom))
        self.vg.fileName = (strFile.rpartition('/')[-1].rpartition('.')[0] + '_') #.encode('ascii')
        
        self.vg.switchToMode(1)
        self.WainUntilLoad()
        
    def OnButtonWeb(self, evt):
        listNames = ['Avenida General Paz y Avenida Roca',\
                     'Avenida General Paz y Acceso Oeste',\
                     'Acceso Norte y Ruta 202',\
                     'Panamericana y Debenedetti',\
                     'Caca']
        
        listSites = ['http://190.220.3.108:1010/mjpg/video.mjpg',\
                     'http://190.220.3.108:1014/mjpg/video.mjpg',\
                     'http://190.220.3.108:1058/mjpg/video.mjpg',\
                     'http://190.220.3.108:1055/mjpg/video.mjpg',\
                     'http://207.251.86.238/cctv636.jpg']

        index = gui_select(self, 'Seleccione una cámara', listNames)
        
        self.vg.setCaptura(mainYabo.capture.WebCapture(listSites[index], 1.))
        
        self.vg.switchToMode(1)
        self.WainUntilLoad()

    def WainUntilLoad(self):
        while len(self.vg.vidAcq) < self.vg.intAcqFrames:
            time.sleep(0.5)

        self.vg.switchToMode(2)

#----------------------------------------------------------------------

class PanelMain(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.vg = parent.vg

        self.img = wx.EmptyImage(2,2)
        self.bmp = self.img.ConvertToBitmap()
        
        self.Bind(EVT_NEW_IMAGE, self.OnNewImage)
        self.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)
        self.Bind(wx.EVT_LEFT_UP, self.OnLeftUp)
        self.Bind(wx.EVT_MOTION, self.OnMouseMove)

        self.eventLock = None
        self.pause = False

        self.InitBuffer()

    def OnLeftDown(self, event):
        x,y = event.GetPosition()

        if self.vg.process_mode == 2:
            # Guardo las posiciones del mouse con los
            # siguientes criterios:
            # Inicial, Actual, Final
            self.vg.mouseRoi = [(),(),()]
            self.vg.mouseRoi[0] = (x,y)
        elif self.vg.process_mode == 3:
            # Guardo 4 puntos
            for i in range(0,4):
                if self.vg.mouse4pt[i] == ():
                   self.vg.mouse4pt[i] = (x,y)
                   break
#        elif self.vg.process_mode == 5:
#            # Marco oclusiones
#            rx = self.vg.mouseRoi[0][0]
#            ry = self.vg.mouseRoi[0][1]
#            if self.vg.labels[0][y-ry,x-rx] != 0:
#                self.vg.mouseLabel[self.vg.labels[0][y-ry,x-rx]] += 1
#            
#                i = self.vg.labels[0][y-ry,x-rx]
#                tmp_label = (self.vg.labels[0]==i)
#                w, h = mainYabo.video_process.getLabelSize(tmp_label)
#                ny = self.vg.centers[i-1][0]
#                numpy.disp(str((w,h,ny)))
        
    def OnLeftUp(self, event):
        x,y = event.GetPosition()
        
        if self.vg.process_mode == 2:
            self.vg.mouseRoi[2] = (x,y)
            
    def OnMouseMove(self, event):
        x,y = event.GetPosition()
        
        if self.vg.process_mode == 2:
            try: self.vg.mouseRoi[1] = (x,y)
            except: pass

    def InitBuffer(self):
        self.bmp = self.img.ConvertToBitmap()
        dc = wx.ClientDC(self)
        self.Draw(dc)

    def Draw(self,dc):
        dc.DrawBitmap(self.bmp,0,0)

    def UpdateImage(self, img):
        self.img = img
        self.InitBuffer()

    def OnNewImage(self, event):
        self.eventLock = event.eventLock

        if not self.pause:
            self.UpdateImage(event.img)
            self.ReleaseEventLock()
        if event.oldImageLock:
            if event.oldImageLock.locked():
                event.oldImageLock.release()

    def ReleaseEventLock(self):
        if self.eventLock:
            if self.eventLock.locked():
                self.eventLock.release()

    def OnPause(self):
        self.pause = not self.pause
        
        if not self.pause:
            self.ReleaseEventLock()

#----------------------------------------------------------------------

class ImageEvent(wx.PyCommandEvent):
    def __init__(self, eventType=EVT_NEW_IMAGE.evtType[0], id=0):
        wx.PyCommandEvent.__init__(self, eventType, id)
        self.img = None
        self.oldImageLock = None
        self.eventLock = None

#----------------------------------------------------------------------

class ImageIn:
    def __init__(self, parent):
        self.parent = parent
        self.eventLock = threading.Lock()

    def SetData(self, arr):
        h,w = arr.shape[0], arr.shape[1]

        # Paso todo a RGB
        try:
            numpy.shape(arr)[2]
            arr = cv2.cvtColor(arr,cv2.COLOR_BGR2RGB)
        except:
            #Format numpy array data for use with wx Image in RGB
            b = arr.copy()
            b.shape = h, w, 1
            bRGB = numpy.concatenate((b,b,b), axis=2)
            arr = bRGB.copy()

        img = wx.ImageFromBuffer(width=w, height=h, dataBuffer=arr.tostring())
        
        event = ImageEvent()
        event.img = img
        event.eventLock = self.eventLock

        event.eventLock.acquire()
        self.parent.AddPendingEvent(event)

class videoThread(threading.Thread):
    def __init__(self, vg, autoStart=True):
        threading.Thread.__init__(self)
        self.vg = vg
        #self.setDaemon(1)
        self.start_orig = self.start
        self.start = self.start_local
        self.frame = None
        self.lock = threading.Lock()
        self.lock.acquire()
        if autoStart:
            self.start()
    def run(self):
        app = wx.App()
        frame = ImageWindow(self.vg, None)

        self.frame = frame
        self.lock.release()

        app.MainLoop()

    def start_local(self):
        self.start_orig()
        self.lock.acquire()

def runVideoThread(vg):
    vt = videoThread(vg)
    frame = vt.frame
    myImageIn = ImageIn(frame.panel[2])
    return myImageIn, frame
