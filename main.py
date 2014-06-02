
from formlayout import fedit
import scipy.io as sio
import numpy as np

from PyQt4 import QtCore, QtGui
from simconf import *
from matplotlib.figure import Figure

import UI
import sys
import visual
import fvisual
import threading
import thread

about_msg = '''
<h2>Simulation platform based on FDTD method </h2>
<p>Developed by Wang Wen, 2014, All Rights Reserved </p>
<p>Email: wanygen@gmail.com</p>
<p>Computational Electromagnetics Lab, UESTC</p>
'''

def visualBrick(brick, opacity = 0.7, offset = (0.0, 0.0, 0.0), material = None, color = visual.color.white):
    center_x = (brick.max_x + brick.min_x) / 2 + offset[0]
    center_y = (brick.max_y + brick.min_y) / 2 + offset[1]
    center_z = (brick.max_z + brick.min_z) / 2 + offset[2]
    size_x = brick.max_x - brick.min_x
    size_y = brick.max_y - brick.min_y
    size_z = brick.max_z - brick.min_z
    visual.box(pos = (center_x, center_y, center_z),
               length = size_x, height = size_y, width = size_z,
               color = color, opacity = opacity,
               material = material
        )

def dependencies_for_myprogram():
    from scipy.sparse.csgraph import _validation
    import formlayout
    import matplotlib.backends.backend_tkagg
    import mpl_toolkits.mplot3d
    from mpl_toolkits.mplot3d import Axes3D

def readFloat(line_edit):
    if(line_edit.text() != ""):
        return float(line_edit.text())
    else:
        raise Exception("Invalid numerical value...")  

def lineEditText(edit, value):
    edit.setText(str(value))
    
class UpdateThread(threading.Thread):
    
    def __init__(self, parent):
        super(UpdateThread, self).__init__() 
        self.parent = parent
        self.stop = False
        
    def run(self):
        self.parent.ui.pushButton_Start.setEnabled(False)
        for p in self.parent.simul.perform():
            if(self.stop): return
            self.parent.progress.emit(p)
        self.parent.ui.pushButton_Start.setEnabled(True)
        
    def terminate(self):
        #super(UpdateThread, self).__stop()
        self.parent.ui.pushButton_Start.setEnabled(True)
        self.stop = True
        return

class MainDialog(QtGui.QWidget):

    progress = QtCore.pyqtSignal(int, name = 'progress')

    def __init__(self):
        super(MainDialog, self).__init__()
        self.ui = UI.Ui_MainWidget()
        self.ui.setupUi(self)
        self.configUI()
        self.simul = Simulation()
        
        self.progress.connect(self.ui.progressBar.setValue)
        self.update_thread = None
        
        X = np.linspace(0, 10, 1000)
        self.ui.graphicWidget.axes.plot(X, np.sin(X))
        self.ui.graphicWidget.draw()
        
        self.f_saved = True
        self.f_perf = False
        
    def configUI(self):
        self.ui.radioButton_GS_PlaneE.setChecked(True)
        self.ui.radioButton_GSC_Ex.setChecked(True)
        self.receive_model = QtGui.QStandardItemModel()
        self.ui.treeView_Receive.setModel(self.receive_model)
        self.receive_model.setHorizontalHeaderLabels(["X", "Y", "Z"])
        self.ui.treeView_Receive.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        
        self.scatterer_model = QtGui.QStandardItemModel()
        self.ui.treeView_Scatterer.setModel(self.scatterer_model)
        self.scatterer_model.setHorizontalHeaderLabels(["Shape", "Min", "Max", "Eps", "Mu", "Sigma"])
        self.ui.treeView_Scatterer.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        
        self.ui.progressBar.setRange(0, 100)
        self.ui.progressBar.setValue(0)
        self.ui.tabWidget.setCurrentIndex(0)
        
    def clearUI(self):
        self.ui.lineEdit_CS_GridSize.clear()
        self.ui.lineEdit_CS_X.clear()
        self.ui.lineEdit_CS_Y.clear()
        self.ui.lineEdit_CS_Z.clear()
        self.ui.lineEdit_GS_Position.clear()
        self.ui.lineEdit_RVP_X.clear()
        self.ui.lineEdit_RVP_Y.clear()
        self.ui.lineEdit_RVP_Z.clear()
        self.ui.lineEdit_SBK_Xmax.clear()
        self.ui.lineEdit_SBK_Xmin.clear()
        self.ui.lineEdit_SBK_Ymax.clear()
        self.ui.lineEdit_SBK_Ymin.clear()
        self.ui.lineEdit_SBK_Zmax.clear()
        self.ui.lineEdit_SBK_Zmin.clear()
        self.ui.lineEdit_SC_BandWidth.clear()
        self.ui.lineEdit_SC_CentralFreq.clear()
        self.ui.lineEdit_SCP_X.clear()
        self.ui.lineEdit_SCP_Y.clear()
        self.ui.lineEdit_SCP_Z.clear()
        self.ui.lineEdit_SMD_Eps.clear()
        self.ui.lineEdit_SMD_Mu.clear()
        self.ui.lineEdit_SMD_Sigma.clear()
        self.ui.lineEdit_TS_Number.clear()
        self.ui.lineEdit_TS_Width.clear()
        
        self.receive_model.clear()
        self.scatterer_model.clear()
        
        self.ui.radioButton_GS_PlaneE.setChecked(True)
        self.ui.radioButton_GSC_Ex.setChecked(True)

    def about(self):
        QtGui.QMessageBox.question(self, 'About',
                about_msg, QtGui.QMessageBox.Yes, QtGui.QMessageBox.Yes)
        
    def addScatterer(self):
        X_min = readFloat(self.ui.lineEdit_SBK_Xmin)
        X_max = readFloat(self.ui.lineEdit_SBK_Xmax)
        Y_min = readFloat(self.ui.lineEdit_SBK_Ymin)
        Y_max = readFloat(self.ui.lineEdit_SBK_Ymax)
        Z_min = readFloat(self.ui.lineEdit_SBK_Zmin)
        Z_max = readFloat(self.ui.lineEdit_SBK_Zmax)
        
        med = Medium()
        med.eps = readFloat(self.ui.lineEdit_SMD_Eps)
        med.mu = readFloat(self.ui.lineEdit_SMD_Mu)
        med.sig = readFloat(self.ui.lineEdit_SMD_Sigma)
        
        i_shape = QtGui.QStandardItem("Brick")
        i_min = QtGui.QStandardItem("%.2f, %.2f, %.2f" % (X_min, Y_min, Z_min))
        i_max = QtGui.QStandardItem("%.2f, %.2f, %.2f" % (X_max, Y_max, Z_max))
        i_med_eps = QtGui.QStandardItem(str(med.eps))
        i_med_mu = QtGui.QStandardItem(str(med.mu))
        i_med_sig = QtGui.QStandardItem(str(med.sig))
        self.scatterer_model.appendRow([i_shape, i_min, i_max, i_med_eps, i_med_mu, i_med_sig])
        self.simul.addScatterer(med, Brick(vector_3(X_min, Y_min, Z_min), vector_3(X_max, Y_max, Z_max)))
        print len(self.simul.scatterers)
    
    def setWall(self):   
        datalist = [('Epsilon', self.simul.wall_medium.eps),
                    ('Mu', self.simul.wall_medium.mu),
                    ('Sigma', self.simul.wall_medium.sig),
                    ('Thickness', self.simul.wall_thick),
            ]

        data = fedit(datalist, title="Configure the Wall",
                       comment="Set the parameters about the wall:")
        if data != None:
            self.simul.setWall(data[3], Medium(data[0], data[1], data[2]))
    
    def scattererDetail(self):
        sel = self.ui.treeView_Scatterer.selectedIndexes()
        if(len(sel) != 0):
            row = sel[0].row()
            scatterer = self.simul.scatterers[row]
            
        datalist = [('X_min', scatterer.shape.min_x),
                    ('Y_min', scatterer.shape.min_y),
                    ('Z_min', scatterer.shape.min_z),
                    ('X_max', scatterer.shape.max_x),
                    ('Y_max', scatterer.shape.max_y),
                    ('Z_max', scatterer.shape.max_z),
                    ('Epsilon', scatterer.medium.eps),
                    ('Mu', scatterer.medium.mu),
                    ('Sigma', scatterer.medium.sig)
            ]
        data = fedit(datalist, title="Scatterer Detail",
                       comment="Set the parameters about the Scatterer:")
        
        if data != None:
            scatterer.shape = Brick(vector_3(data[0], data[1], data[2]), vector_3(data[3], data[4], data[5]))
            scatterer.medium = Medium(data[6], data[7], data[8])
            self.scatterer_model.setItem(row, 1, QtGui.QStandardItem("%.2f, %.2f, %.2f" % (data[0], data[1], data[2])))
            self.scatterer_model.setItem(row, 2, QtGui.QStandardItem("%.2f, %.2f, %.2f" % (data[3], data[4], data[5])))
            self.scatterer_model.setItem(row, 3, QtGui.QStandardItem(str(data[6])))
            self.scatterer_model.setItem(row, 4, QtGui.QStandardItem(str(data[7])))
            self.scatterer_model.setItem(row, 5, QtGui.QStandardItem(str(data[8])))
            
    def exportReceiveSignal(self):
        select = self.ui.treeView_Receive.selectedIndexes()
        if len(select) == 0:
            return
        row = select[0].row()
        signal = self.simul.task.getReceiveSignal(row)
        fname = QtGui.QFileDialog.getSaveFileName(self, 'Save Signal', 'untitled', '*.csv')
        if(fname == '' or fname == None):
            return
        signal.save(str(fname))
        return
        
        
    def exportAllReceive(self):
        fname = QtGui.QFileDialog.getSaveFileName(self, 'Save Signal', 'untitled', '*.mat')
        if(fname == '' or fname == None):
            return
        n = len(self.simul.receives)
        time = np.arange(0.0, self.simul.dt * self.simul.N, self.simul.dt)
        d = {"Time": time}
        pos = []
        for i in range(n):
            signal = self.simul.task.getReceiveSignal(i)
            key = "Signal" + str(i)
            d[key] = np.array(map(lambda n: signal[n], range(signal.length)))
            vc = self.simul.receives[i].position
            pos.append((vc.x, vc.y, vc.z))
        d["Position"] = pos
        sio.savemat(str(fname), d)
            
    def deleteScatterer(self):
        sel = self.ui.treeView_Scatterer.selectedIndexes()
        if(len(sel) != 0):
            row = sel[0].row()
            self.scatterer_model.removeRow(row)
            del self.simul.scatterers[row]
        print len(self.simul.scatterers)
        
    def addReceivePoint(self):
        X = readFloat(self.ui.lineEdit_RVP_X)
        Y = readFloat(self.ui.lineEdit_RVP_Y)
        Z = readFloat(self.ui.lineEdit_RVP_Z)
        self.receive_model.appendRow([QtGui.QStandardItem(str(X)),
                        QtGui.QStandardItem(str(Y)),
                        QtGui.QStandardItem(str(Z))]
                    )
        self.simul.addReceivePoint(vector_3(X, Y, Z))

    def startSimulation(self):
        self.readBasicConfig()
        self.update_thread = UpdateThread(self)
        self.update_thread.start()
 
        self.f_perf = True
        
    def stopSimulation(self):
        self.update_thread.terminate()
        
    def deleteReceivePoint(self):
        select = self.ui.treeView_Receive.selectedIndexes()
        if len(select) != 0:
            row = select[0].row()
            self.receive_model.removeRow(row)
            del self.simul.receives[row]
        #print len(self.simul.receives)
        
    def readBasicConfig(self):
        self.simul.dt = readFloat(self.ui.lineEdit_TS_Width)
        self.simul.N = int(readFloat(self.ui.lineEdit_TS_Number))
        cs_x = readFloat(self.ui.lineEdit_CS_X)
        cs_y = readFloat(self.ui.lineEdit_CS_Y)
        cs_z = readFloat(self.ui.lineEdit_CS_Z)
        cs_resol = readFloat(self.ui.lineEdit_CS_GridSize)
        self.simul.coord_sys = Cartesian(vector_3(cs_x, cs_y, cs_z), cs_resol)
        sor_x = readFloat(self.ui.lineEdit_SCP_X)
        sor_y = readFloat(self.ui.lineEdit_SCP_Y)
        sor_z = readFloat(self.ui.lineEdit_SCP_Z)
        sor_freq = readFloat(self.ui.lineEdit_SC_CentralFreq)
        sor_band = readFloat(self.ui.lineEdit_SC_BandWidth)
        self.simul.setSource(vector_3(sor_x, sor_y, sor_z), sor_freq, sor_band)
        
        
    def saveSimulation(self):
        self.readBasicConfig()
        fname = QtGui.QFileDialog.getSaveFileName(self, 'Save file', 'untitled', '*.fmc')
        if fname != '' and fname != None:
            f = open(fname, 'w')
            f.write(self.simul.json())
            f.close()
            
    def plotReceiveSignal(self):
        print 'plotting received signal...'
        sel = self.ui.treeView_Receive.selectedIndexes()
        if(len(sel) == 0 or (not self.f_perf)):
            return
        row = sel[0].row()
        sig = self.simul.task.getReceiveSignal(row)
        fvisual.plot_signal(sig)
        
    def showEnvirModel(self):
        scene = visual.display(title='Channel Environment Visualization', exit = False)
        scene.objects
        max_x = readFloat(self.ui.lineEdit_CS_X)
        max_y = readFloat(self.ui.lineEdit_CS_Y)
        max_z = readFloat(self.ui.lineEdit_CS_Z)
        offset = (-max_x / 2, -max_y / 2, -max_z / 2)
        th = 0.02
        wl = self.simul.wall_thick
        visualBrick(Brick(vector_3(wl-th, 0, 0), vector_3(wl, max_y, max_z)), opacity = 0.2, offset = offset)
        visualBrick(Brick(vector_3(0, wl-th, 0), vector_3(max_x, wl, max_z)), opacity = 0.2, offset = offset)
        visualBrick(Brick(vector_3(0, 0, wl-th), vector_3(max_x, max_y, wl)), opacity = 0.2, offset = offset)
        visualBrick(Brick(vector_3(max_x - wl, 0, 0), vector_3(max_x + th - wl, max_y, max_z)), opacity = 0.2, offset = offset)
        visualBrick(Brick(vector_3(0, max_y - wl, 0), vector_3(max_x, max_y + th - wl, max_z)), opacity = 0.2, offset = offset)
        visualBrick(Brick(vector_3(0, 0, max_z - wl), vector_3(max_x, max_y, max_z + th - wl)), opacity = 0.2, offset = offset)
        #visualBrick(Brick(vector_3(0, 0, 0), vector_3(max_x, max_y, max_z)), opacity = 0.3, offset = offset)
        for scatt in self.simul.scatterers:
            visualBrick(scatt.shape, offset = offset, opacity = 0.9, color = (0.69, 0.09, 0.122), material = visual.materials.plastic)
        
    def showFieldDistribution(self):
        pos = readFloat(self.ui.lineEdit_GS_Position)
        component = None
        if self.ui.radioButton_GSC_Ex.isChecked():
            component = lambda yee: yee.E.x
        elif self.ui.radioButton_GSC_Ey.isChecked():
            component = lambda yee: yee.E.y
        elif self.ui.radioButton_GSC_Ez.isChecked():
            component = lambda yee: yee.E.z
        elif self.ui.radioButton_GSC_Hx.isChecked():
            component = lambda yee: yee.H.x
        elif self.ui.radioButton_GSC_Hy.isChecked():
            component = lambda yee: yee.H.y
        elif self.ui.radioButton_GSC_Hz.isChecked():
            component = lambda yee: yee.H.z
        
        self.ui.graphicWidget.figure.subplots_adjust(left=0.0, bottom=0.0, top=1.0, right=1.0)
        if self.ui.radioButton_GS_PlaneE.isChecked():
            fvisual.surf_E_plane(self.simul.task, pos, component, self.ui.graphicWidget.figure)
        else:
            fvisual.surf_H_plane(self.simul.task, pos, component, self.ui.graphicWidget.figure)
            
        self.ui.graphicWidget.draw()
        #self.ui.graphicWidget.figure.show()
    '''
    def showFieldDistribution(self):
        component = lambda yee: yee.E.z
        fvisual.surf_E_plane(self.simul.task, 2.5, component, self.ui.graphicWidget.figure)
        self.ui.graphicWidget.draw()
        print 'Done'
    '''
    def updateUI(self):
        lineEditText(self.ui.lineEdit_TS_Number, self.simul.N)
        lineEditText(self.ui.lineEdit_TS_Width, self.simul.dt)
        lineEditText(self.ui.lineEdit_CS_X, self.simul.coord_sys.max_x)
        lineEditText(self.ui.lineEdit_CS_Y, self.simul.coord_sys.max_y)
        lineEditText(self.ui.lineEdit_CS_Z, self.simul.coord_sys.max_z)
        lineEditText(self.ui.lineEdit_CS_GridSize, self.simul.coord_sys.resol)
        lineEditText(self.ui.lineEdit_SC_BandWidth, self.simul.source.band_width)
        lineEditText(self.ui.lineEdit_SC_CentralFreq, self.simul.source.central_freq)
        lineEditText(self.ui.lineEdit_SCP_X, self.simul.source.position.x)
        lineEditText(self.ui.lineEdit_SCP_Y, self.simul.source.position.y)
        lineEditText(self.ui.lineEdit_SCP_Z, self.simul.source.position.z)
        
        self.scatterer_model.setHorizontalHeaderLabels(["Shape", "Min", "Max", "Eps", "Mu", "Sigma"])
        self.receive_model.setHorizontalHeaderLabels(["X", "Y", "Z"])

        for recv in self.simul.receives:
            X = recv.position.x
            Y = recv.position.y
            Z = recv.position.z
            self.receive_model.appendRow([QtGui.QStandardItem(str(X)),
                            QtGui.QStandardItem(str(Y)),
                            QtGui.QStandardItem(str(Z))]
                    )
        for scatt in self.simul.scatterers:
            X_min = scatt.shape.min_x
            X_max = scatt.shape.max_x
            Y_min = scatt.shape.min_y
            Y_max = scatt.shape.max_y
            Z_min = scatt.shape.min_z
            Z_max = scatt.shape.max_z
        
            med = scatt.medium
            i_shape = QtGui.QStandardItem("Brick")
            i_min = QtGui.QStandardItem("%.2f, %.2f, %.2f" % (X_min, Y_min, Z_min))
            i_max = QtGui.QStandardItem("%.2f, %.2f, %.2f" % (X_max, Y_max, Z_max))
            i_med_eps = QtGui.QStandardItem(str(med.eps))
            i_med_mu = QtGui.QStandardItem(str(med.mu))
            i_med_sig = QtGui.QStandardItem(str(med.sig))
            self.scatterer_model.appendRow([i_shape, i_min, i_max, i_med_eps, i_med_mu, i_med_sig])
            
    def openSimulation(self):
        if self.newSimulation() == QtGui.QMessageBox.Cancel:
            return False
        fname = QtGui.QFileDialog.getOpenFileName(self, 'Open file', '\home', filter = '*.fmc')
        self.simul.load(fname)
        self.updateUI()
        
    def newSimulation(self):
        if self.f_saved == False:
            reply = QtGui.QMessageBox.question(self, 'Message',
                        "Do you want to save your former configurations?", QtGui.QMessageBox.Yes | 
                        QtGui.QMessageBox.No | QtGui.QMessageBox.Cancel, QtGui.QMessageBox.Yes)

            if reply == QtGui.QMessageBox.Yes:
                self.saveSimulation()
                self.simul.clear()
                self.clearUI()
                return reply
            elif reply == QtGui.QMessageBox.Cancel:
                return reply
            else:
                self.simul.clear()
                self.clearUI()
                return reply
        else:
            self.simul.clear()
            self.clearUI()
        
        
if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    widget = MainDialog()
    widget.show()
    sys.exit(app.exec_())
 