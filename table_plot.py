import numpy as np
import matplotlib.path as mpltPath
from mpl_toolkits.basemap import Basemap
import sys
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
def gen_state_polys():
        tree = ET.parse('states.xml')
        root = tree.getroot()
        polys={}
        for child in root:
                if child.tag!='state':
                        continue
                plist=[]
                for p in child:
                        if p.tag!='point':
                                continue
                        plist.append([float(p.attrib['lng']), float(p.attrib['lat'])])
                polys[child.attrib['name'].lower().strip()]=np.array(plist)
        return polys
def state_has_p(state_bounds, lnglat):
        path = mpltPath.Path(state_bounds)
        return path.contains_points(lnglat)
def longlat2state(all_state_bounds,lnglat):
        for s in all_state_bounds:
                if state_has_p(all_state_bounds[s],np.array([lnglat])):
                        return s
        return None
if __name__=="__main__":
        ax=plt.subplot(1,1,1)
        map = Basemap(width=12000000,height=9000000,projection='lcc',
            resolution='c',lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.)        
        P=gen_state_polys()
        patches=[]
        map.drawmapboundary(fill_color='gray')
        map.drawcoastlines()
        map.drawstates()
        sf=open('table_states.txt')
        c=[]
        for line in sf:
                s=' '.join(line.split()[0:-2])
                fear=float(line.split()[-2])
                reg=float(line.split()[-1])
                print(s,fear,reg)
                shape=P[s]
                patches.append( Polygon(np.array(shape)[:], True) )
                c.append([fear,reg,reg,1.0])
                ax.add_collection(PatchCollection(patches, facecolor='m', edgecolor='k'))
                patches=[]
                c=[]
        plt.show()

