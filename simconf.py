
from fdtd_W import *
from math import *
import json

vector_3.dict = lambda self: {"X": self.x, "Y": self.y, "Z": self.z }

point_3 = vector_3
coord_3 = vector_3
v = vector_3

Medium.eps0 = 8.854187817e-12
Medium.mu0 = 4e-7 * pi
Medium.sig0 = 0.0

Medium.silver = Medium(Medium.eps0, Medium.mu0, 6.17e7)
Medium.red_copper = Medium(Medium.eps0, Medium.mu0, 5.80e7)
Medium.brass = Medium(Medium.eps0, Medium.mu0, 1.57e7)
Medium.gold = Medium(Medium.eps0, Medium.mu0, 4.10e7)
Medium.concrete = Medium(8.16 * Medium.eps0, Medium.mu0, 0.001)
Medium.wood = Medium(2.1 * Medium.eps0, 1.00000043 * Medium.mu0, 0.0)

def _sphere_dict(self):
    return { "Shape": "Sphere",
             "Center_X": self.center.x,
             "Center_Y": self.center.y,
             "Center_Z": self.center.z,
             "Radius": self.radius
            }
    
def _brick_dict(self):
    return { "Shape": "Brick",
             "X_min": self.min_x,
             "Y_min": self.min_y,
             "Z_min": self.min_z,
             "X_max": self.max_x,
             "Y_max": self.max_y,
             "Z_max": self.max_z
           }
    
def _medium_dict(self):
    return { "Epsilon": self.eps,
             "Mu": self.mu,
             "Sigma": self.sig
            }
    
def _cartesian_dict(self):
    return { "X_max": self.max_x,
             "Y_max": self.max_y,
             "Z_max": self.max_z,
             "Resolution": self.resol
            }
    
def _source_dict(self):
    return { "Position": self.position.dict(),
             "Central Frequency": self.central_freq,
             "Band Width": self.band_width
            }
    
def _recommand_ds(self):
    high_freq = self.central_freq + self.band_width / 2
    lamb = 3e8 / (high_freq * 1e9)
    return lamb / 10

def _recommand_dt(self):
    ds = _recommand_ds(self)
    return ds / (3e8 * sqrt(3.0))

Source.recommand_ds = _recommand_ds
Source.recommand_dt = _recommand_dt
    
Sphere.dict = _sphere_dict
Brick.dict = _brick_dict
Medium.dict = _medium_dict
Cartesian.dict = _cartesian_dict
Source.dict = _source_dict

class Scatterer(object):
    
    def __init__(self, shape = None, medium = None):
        
        self.shape = shape
        self.medium = medium
    
    def dict(self):
        return {
                 "Geometry": self.shape.dict(),
                 "Medium": self.medium.dict()
                }
        
class Receive(object):
        
    def __init__(self, posi = v(0, 0, 0), signal = Signal()):
        self.position = posi
        self.signal = signal
    
    def dict(self):
        seq = map(lambda n: self.signal[n], range(self.signal.length))        
        return { "Position": self.position.dict(),
                "Signal": seq
                }

'''
class Source(object):
    
    def __init__(self):
        
        self.position = vector_3()
        self.central_freq = 0.0
        self.band_width = 0.0
    
    def dict(self):
        return { "Position": self.position.dict(),
                "Central Frequency": self.central_freq,
                "Band Width": self.band_width
            }
'''

class Simulation(object):

    def __init__(self):
        self.coord_sys = Cartesian()
        self.dt = 0.0
        self.N = 0
        self.source = Source()
        self.scatterers = []
        self.receives = []
        self.wall_thick = 0.0
        self.wall_medium = Medium()
        self.task = Fdtd()
        
    def clear(self):
        self.task.release()
        self.__init__()
        
    def setCoordinateSystem(self, coord_sys):
        self.coord_sys = coord_sys
        
    def setTimeTep(self, step_width, step_number):
        self.dt = step_width
        self.N = step_number
        
    def setWall(self, thick, medium):
        self.wall_thick = thick
        self.wall_medium = medium
        
    def addScatterer(self, medium, shape):
        self.scatterers.append(Scatterer(shape, medium))
    
    def addReceivePoint(self, point):
        self.receives.append(Receive(point, Signal(point, self.dt)))
    
    def setSource(self, point, central_freq, band_width):
        self.source = Source(point, central_freq, band_width)
        
    def perform(self):
        self.task.release()
        self.task.setCoordinateSystem(self.coord_sys)
        self.task.setTimePace(self.dt)
        self.task.addPointSource(self.source.position, self.source.central_freq, self.source.band_width)
        self.task.setWall(self.wall_thick, self.wall_medium)
        for recv in self.receives:
            self.task.addReceivePoint(recv.position)
        for scat in self.scatterers:
            self.task.addMedium(scat.shape, scat.medium)
        for i in range(self.N):
            self.task.update(1)
            prog = int ((float(i + 1) / self.N) * 100 )
            yield prog
        
    def release(self):
        self.task.release()
        
    def dict(self):
        dc = { "Coordinate System": self.coord_sys.dict(),
               "Time Step": { "Step Width": self.dt,
                              "Step Number": self.N
                            },
               "Source": self.source.dict(),
               "Wall": { "Thickness": self.wall_thick,
                        "Medium": self.wall_medium.dict()
                       },
               "Scatterers": [],
               "Receives": []
           }
        
        for sc in self.scatterers:
            dc["Scatterers"].append(sc.dict())
        
        for rv in self.receives:
            dc["Receives"].append(rv.dict())
            
        return dc

    def json(self):
        return json.dumps(self.dict(), sort_keys = True, indent = 4)
    
    def save(self, filename):
        fl = open(filename, "w")
        fl.write(self.json())
        fl.close()
        return True
    
    def load(self, filename):
        jsons = json.loads(open(filename, "r").read())
        json_cs = jsons["Coordinate System"]
        self.coord_sys = Cartesian(
                coord_3(
                    json_cs["X_max"],
                    json_cs["Y_max"],
                    json_cs["Z_max"]
                ),
                json_cs["Resolution"]
        )
        
        json_ts = jsons["Time Step"]
        self.dt = json_ts["Step Width"]
        self.N = json_ts["Step Number"]
        
        json_sc = jsons["Source"]
        pos = v(json_sc["Position"]["X"], json_sc["Position"]["Y"], json_sc["Position"]["Z"])
        central_freq = json_sc["Central Frequency"]
        band_width = json_sc["Band Width"]
        self.source = Source(pos, central_freq, band_width)
        
        json_wl = jsons["Wall"]
        self.wall_thick = json_wl["Thickness"]
        self.wall_medium = Medium(json_wl["Medium"]["Epsilon"],
                    json_wl["Medium"]["Mu"],
                    json_wl["Medium"]["Sigma"])
        
        self.scatterers = []
        json_st = jsons["Scatterers"]
        for scat in json_st:
            meds = scat["Medium"]
            med = Medium(meds["Epsilon"], meds["Mu"], meds["Sigma"])
            geos = scat["Geometry"]
            if(geos["Shape"] == "Brick"):
                brick = Brick(v(geos["X_min"], geos["Y_min"], geos["Z_min"]), v(geos["X_max"], geos["Y_max"], geos["Z_max"]))
                self.scatterers.append(Scatterer(shape = brick, medium = med))
            if(geos["Shape"] == "Sphere"):
                sphere = Sphere(v(geos["Center_X"], geos["Center_Y"], geos["Center_Z"]), geos["Radius"])
                self.scatterers.append(Scatterer(shape = sphere, medium = med))
        
        self.receives = []
        for receiv in jsons["Receives"]:
            poss = receiv["Position"]
            pos = v(poss["X"], poss["Y"], poss["Z"])
            signal = Signal(pos, self.dt)
            for amp in receiv["Signal"]:
                signal.push_value(amp)
            self.receives.append(Receive(pos, signal))
            
        return True


def save_test():
    fc = Simulation()
    fc.scatterers.append(Scatterer(Sphere(), Medium()))
    fc.scatterers.append(Scatterer(Brick(), Medium()))
    fc.receives.append(Receive())
    fc.save("filename")
    
def load_test():
    fc = Simulation()
    fc.load("filename")
    fc.save("another")

if __name__ == '__main__':
    src = Source(v(0, 0, 0), 0.25, 0.2)
    print src.recommand_ds()
    print src.recommand_dt()