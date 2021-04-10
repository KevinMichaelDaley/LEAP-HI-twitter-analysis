import numpy as np, sys
import scipy.stats
from scipy.stats import kendalltau as ktau
from scipy.stats import pearsonr, linregress

#import matplotlib.cm as cm
import xml.etree.ElementTree as ET
usps_abbrev=[]
for line in open('/home/kmd/tweat/src/states_abbrev.txt'):
    if line.strip()=='':
        continue
    fields=line.split()
    usps_abbrev.append(fields[-1])
usps_abbrev.remove('DC')
usps_abbrev.append('DC')
def ix_to_usps(i, DC=True):
    Br=usps_abbrev
    if not DC:
        return sorted(Br[:-1]).index(usps_abbrev[i])
    return sorted(Br).index(usps_abbrev[i])
rega=[]
feara=[]
sts=[]
statenames=[]
sn=[]
FIPS=[x.split()[-1] for x in open('/home/kmd/tweat/src/fips.txt') if x.strip()!='']
pops=np.loadtxt('/home/kmd/tweat/src/pops2.txt', delimiter=' ', usecols=[2]).reshape([50,-1])

laws=np.loadtxt('/home/kmd/tweat/src/laws.txt')
ixx=0
for line in open('/home/kmd/tweat/src/bystate_ht2.txt'):
        s=' '.join(line.split()[:-2]).lower()
        fields=float(line.split()[-1]), float(line.split()[-2])
        rega.append(fields[0])
        feara.append(fields[1])
        sts.append(s)
        statenames.append(s.upper())
        ixx+=1



bgs2=np.loadtxt('/home/kmd/tweat/src/gallup_preprocesses.txt')
bgs2[:,0]=np.array([FIPS.index(x.split()[0].zfill(5)[:2]) for x in open('/home/kmd/tweat/src/gallup_preprocesses.txt')])
bgs3=[]
for i in range(50):
    bgs3.append(np.sum(np.logical_and(bgs2[:,0]==i, bgs2[:,1]==1))/np.sum(np.logical_and(bgs2[:,0]==i, bgs2[:,1]<4)))
    
bgs3=np.nan_to_num(np.array(bgs3))
bg=np.loadtxt('/home/kmd/tweat/src/bgchex_raw2.csv', skiprows=1, usecols=list(range(1,51)))
#print("populations:", pops[ix_to_usps(19,False)][-1])

slopes=[]
slopes2=[]
#bgs3=[]

for i in range(50):
    Xx=bg[-6:-3,i].sum()/pops[ix_to_usps(i,False), -1]
    slopes.append(Xx)

for i in range(50):
    Xx=bg[:,i].sum()/pops[ix_to_usps(i,False), -1]
    slopes2.append(Xx)
slopes=np.array(slopes)
slopes2=np.array(slopes2)
def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    # Finally get corr coeff
    return (A.dot(B)) / (np.linalg.norm(A)*np.linalg.norm(B))


def gen_state_polys():
        tree = ET.parse('/home/kmd/tweat/src/states.xml')
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
    
def TXTBOX(x,y,z, txt, scale=0.1, rot=[],  fontsize=2, fontfamily=None):
    import bpy, bmesh
    myFontCurve = bpy.data.curves.new(type="FONT",name="myFontCurve")
    myFontOb = bpy.data.objects.new(txt,myFontCurve)
    myFontOb.data.body = txt
    if fontfamily is not None and fontfamily !='':
        fnt = bpy.data.fonts.load(fontfamily)
        myFontOb.data.font = fnt
        myFontOb.data.size = fontsize
    myFontOb.scale=[scale,scale,scale]
    myFontOb.location=[x,y,z]
    if len(rot)==3:
        myFontOb.rotation_euler=(np.array(rot)*np.pi/180.0).tolist()
    bpy.context.collection.objects.link(myFontOb)
    return myFontOb
def corr(x,y):
        return pearsonr(x,y)[0]

def corr3(x,y,z):
        R=np.array([[corr(x,x), corr(x,y)],[corr(y,x), corr(y,y)]])
        c=np.array([corr(x,z), corr(y,z)])
        r= np.sqrt(c.T @ np.linalg.inv(R) @ c)
        F=(r*r/2)/((1-r*r)/(len(x)-3))
        df1=2
        df2=len(x)-3
        return [r,1.0-scipy.stats.f.cdf(F, df1, df2)]
        
def arrow(body_length=1, body_width=0.03, head_length=0.05, head_width=0.04, height=0.02):
        import bpy, bmesh

        #Half sizes along Z
        half_body_width = body_width / 2.0
        half_head_width = head_width / 2.0
        V=[]
        V2=[]
        #Define the wanted coordinates
        coords = [ [body_length, 0.0, half_body_width],       #1
                    [body_length, 0.0, half_head_width],      #2
                    [body_length + head_length, 0.0, 0.0],    #3
                    [body_length, 0.0, -half_head_width],     #4
                    [body_length, 0.0, -half_body_width],     #5
                    [0, 0.0, -half_body_width],    #6
#                    [-body_length, 0.0, -half_head_width],    #7
#                    [-body_length - head_length, 0.0, 0.0],   #8
#                    [-body_length, 0.0, half_head_width],     #9
                    [0, 0.0, half_body_width] ]    #10

        #Create the mesh
        bpy.ops.object.add(type='MESH')
        #Get the object
        obj = bpy.context.object
        #Get the mesh data
        mesh = obj.data

        #Create a bmesh instance in order to add data (vertices and faces) to the mesh
        bm = bmesh.new()

        #Create the vertices
        for coord in coords:
            V.append(bm.verts.new( coord ))
            V2.append(bm.verts.new([coord[0], height, coord[2]]))
            

        #Add a face with all the vertices (the vertices order matters here)
        F=bm.faces.new(  V )
        F2=bm.faces.new(  V2 )
        
        bmesh.ops.bridge_loops(bm, edges=list(F.edges) + list(F2.edges))
        #Updates to Blender
        bm.to_mesh(mesh)
        mesh.update()
        return obj

def scplot(x,y,z,sz,xlabel,ylabel,zlabel, zoffs=0, sz0=0.02, h=0.006, xgap=[0.25,0.75], ygap=[0.25,0.75], zgap=[0.25,0.75], use_states=True):
    import bpy, bmesh
    P=gen_state_polys()
    for i in range(len(x)):
        W=max(np.std(P[sts[i]][:,0]),np.std(P[sts[i]][:,0]))
        mx=np.mean(P[sts[i]][:,0])
        my=np.mean(P[sts[i]][:,1])
        mesh = bpy.data.meshes.new(sts[i])
        basic_sphere = bpy.data.objects.new(statenames[i], mesh)
        bpy.context.collection.objects.link(basic_sphere)

        # Select the newly created object
        bpy.context.view_layer.objects.active = basic_sphere
        basic_sphere.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.new()
        outlier=True
        Xn=(x-np.min(x))/(np.max(x)-np.min(x))
        Yn=(y-np.min(y))/(np.max(y)-np.min(y))
        Zn=(z-np.min(z))/(np.max(z)-np.min(z))
        if np.abs(Xn[i]-Xn.mean())>2.0*np.std(Xn):
                outlier=True
        if np.abs(Yn[i]-Yn.mean())>2.0*np.std(Yn):
                outlier=True
        if np.abs(Zn[i]-Zn.mean())>2.0*np.std(Zn):
                outlier=True
        if not outlier or use_states==False:
                bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, diameter=sz0)
        else:
                V=[]
                V2=[]
                for a in P[sts[i]]:
                   V.append(bm.verts.new([(a[0]-mx)*sz/W,(a[1]-my)*sz/W,0]))
                   V2.append(bm.verts.new([(a[0]-mx)*sz/W,(a[1]-my)*sz/W,h]))
                F=bm.faces.new(V)
                F2=bm.faces.new(V2)
                bmesh.ops.bridge_loops(bm, edges=list(F.edges) + list(F2.edges))
        bm.normal_update()
        color_layer = bm.loops.layers.color.new("Col")
        for face in bm.faces:
            for loop in face.loops:
                #loop[color_layer]=cm.bwr(1.0-(laws[i]-min(laws))/(max(laws)-min(laws)))
                pass
        bpy.ops.object.mode_set(mode='OBJECT')
        bm.to_mesh(mesh)
        bm.free()
        ob=basic_sphere
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
        basic_sphere.location=[Xn[i]*(xgap[1]-xgap[0])+xgap[0],Yn[i]*(ygap[1]-ygap[0])+ygap[0],Zn[i]*(zgap[1]-zgap[0])+zoffs+zgap[0]]
        basic_sphere.rotation_euler=[np.pi/6.0, -np.pi/4, 3.0*np.pi/4]
        # Get material
        mat = bpy.data.materials.get("Material")
        if mat is None:
            # create material
            mat = bpy.data.materials.new(name="Material")

            # assign to 1st material slot
            ob.data.materials[0] = mat
        else:
            # no slots
            ob.data.materials.append(mat)
        
    ob1=TXTBOX(0.95,0.2, 0.001+zoffs, ylabel)
    ob1.scale=[0.10,0.1,0.1]
    d90=np.pi/2.0
    ob1.rotation_euler=[0,0,d90]

    y0=TXTBOX(0.01,ygap[0], 0.001+zoffs,'%.3f'%min(y))
    y0.scale=[0.075,0.075,0.075]
    y0.rotation_euler=[d90,0,d90]

    y1=TXTBOX(0.01,ygap[1]-0.01-len('%.3f'%max(y))*0.05, 0.001+zoffs, '%.3f'%max(y))
    y1.scale=[0.075,0.075,0.075]
    y1.rotation_euler=[d90,0,d90]

    x0=TXTBOX(xgap[0]+len('%.3f'%min(y))*0.05, 0.001,0.001+zoffs, '%.3f'%min(x))
    x0.scale=[-0.075,0.075,0.075]
    x0.rotation_euler=[d90,0,0]

    x1=TXTBOX(xgap[1], 0.001, 0.001+zoffs, '%.3f'%max(x))
    x1.scale=[-0.075,0.075,0.075]
    x1.rotation_euler=[d90,0,0]
    
    
    z0=TXTBOX(0.001,0.99,zgap[0]+zoffs, '%.3f'%min(z))
    z0.scale=[0.075,0.075,0.075]
    z0.rotation_euler=[d90,0,d90]

    z1=TXTBOX(0.0,0.99,zgap[1]+zoffs, '%.3f'%max(z))
    z1.scale=[0.075,0.075,0.075]
    z1.rotation_euler=[d90,0,d90]

    xyc=TXTBOX(1.2,0.05, 0.001+zoffs, 'r=%.2f,p=%.3f'%(pearsonr(x,y)))
    xyc.rotation_euler=[0,0,d90]
    xyc.scale=[0.16,0.16,0.16]
    ob2=TXTBOX(0.9,0.05, 0.001+zoffs, xlabel)
    ob2.rotation_euler=[0,0,0]
    ob2.scale=[-0.1,-0.1,0.1]
    ob3=TXTBOX(0.1,0.01, 1.3+zoffs, zlabel)
    ob3.rotation_euler=[-d90,d90,0]
    ob3.scale=[0.125,0.125,0.125]
    xzc=TXTBOX(1.2,0.001, 1.15+zoffs, 'r=%.2f,p=%.3f'%(pearsonr(x,z)))
    xzc.scale=[0.16,0.16,0.16]
    xzc.rotation_euler=[-d90,d90,0]
    yzc=TXTBOX(0.0,1.2, 1.15+zoffs, 'r=%.2f,p=%.3f'%(pearsonr(y,z)))
    yzc.scale=[0.16,0.16,0.16]
    yzc.rotation_euler=[0,d90,0]
    data=np.concatenate((Xn*(xgap[1]-xgap[0])+xgap[0],Yn*(ygap[1]-ygap[0])+ygap[0],Zn*(zgap[1]-zgap[0])+zgap[0]+zoffs)).reshape((3,-1)).T
    M=data.mean(axis=0) 
    uu, dd, vv = np.linalg.svd(data - M)
    dir=vv[0]
    #ar=arrow(2.0*data.std())
    #ar.location=M-data.std()
    dir/=np.linalg.norm(dir)

    #ar.rotation_euler=[np.arctan(dir[1]/dir[0]) if dir[0]!=0 else d90,d90/2.0,np.arcsin(dir[2])]
    r,p=corr3(x,y,z)
    obf=TXTBOX(0,0,1.6+zoffs, '|r|=%.2f, p=%.3f'%(r, p))
    obf.rotation_euler=[d90,0,d90]
    obf.scale=[0.2,0.2,0.2]   
    
def lplot(x,y,z, sz, xlabel,ylabel,zlabel, zoffs=0, xgap=[0.25,0.75], ygap=[0.25,0.75], zgap=[0.25,0.75], h=0.3):
    import bpy, bmesh
    for i in range(len(x)):
       
        mesh = bpy.data.meshes.new(str(i))
        basic_sphere = bpy.data.objects.new(str(i), mesh)
        bpy.context.collection.objects.link(basic_sphere)

        # Select the newly created object
        bpy.context.view_layer.objects.active = basic_sphere
        basic_sphere.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.new()
        outlier=False
        Xn=(x-np.min(x))/(np.max(x)-np.min(x))
        Yn=(y-np.min(y))/(np.max(y)-np.min(y))
        Zn=(z-np.min(z))/(np.max(z)-np.min(z))
        bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, diameter=sz)
        bm.normal_update()
        color_layer = bm.loops.layers.color.new("Col")
        for face in bm.faces:
            for loop in face.loops:
                loop[color_layer]=cm.magma(i/len(x))
        bpy.ops.object.mode_set(mode='OBJECT')
        bm.to_mesh(mesh)
        bm.free()
        ob=basic_sphere
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
        basic_sphere.location=[Xn[i]*(xgap[1]-xgap[0])+xgap[0],Yn[i]*(ygap[1]-ygap[0])+ygap[0],Zn[i]*(zgap[1]-zgap[0])+zoffs+zgap[0]]
        basic_sphere.rotation_euler=[np.pi/6.0, -np.pi/4, 3.0*np.pi/4]
        # Get material
        mat = bpy.data.materials.get("Material")
        if mat is None:
            # create material
            mat = bpy.data.materials.new(name="Material")

            # assign to 1st material slot
            ob.data.materials[0] = mat
        else:
            # no slots
            ob.data.materials.append(mat)
            
    ob1=TXTBOX(0.95,0.2, 0.001+zoffs, ylabel)
    ob1.scale=[0.10,0.1,0.1]
    d90=np.pi/2.0
    ob1.rotation_euler=[0,0,d90]

    y0=TXTBOX(0.01,ygap[0], 0.001+zoffs,'%.3f'%min(y))
    y0.scale=[0.075,0.075,0.075]
    y0.rotation_euler=[d90,0,d90]

    y1=TXTBOX(0.01,ygap[1]-0.01-len('%.3f'%max(y))*0.05, 0.001+zoffs, '%.3f'%max(y))
    y1.scale=[0.075,0.075,0.075]
    y1.rotation_euler=[d90,0,d90]

    x0=TXTBOX(xgap[0]+len('%.3f'%min(y))*0.05, 0.001,0.001+zoffs, '%.3f'%min(x))
    x0.scale=[-0.075,0.075,0.075]
    x0.rotation_euler=[d90,0,0]

    x1=TXTBOX(xgap[1], 0.001, 0.001+zoffs, '%.3f'%max(x))
    x1.scale=[-0.075,0.075,0.075]
    x1.rotation_euler=[d90,0,0]
    
    
    z0=TXTBOX(0.001,0.99,zgap[0]+zoffs, '%.3f'%min(z))
    z0.scale=[0.075,0.075,0.075]
    z0.rotation_euler=[d90,0,d90]

    z1=TXTBOX(0.0,0.99,zgap[1]+zoffs, '%.3f'%max(z))
    z1.scale=[0.075,0.075,0.075]
    z1.rotation_euler=[d90,0,d90]

    xyc=TXTBOX(1.2,0.05, 0.001+zoffs, 'r=%.2f,p=%.3f'%(pearsonr(x,y)))
    xyc.rotation_euler=[0,0,d90]
    xyc.scale=[0.16,0.16,0.16]
    ob2=TXTBOX(0.9,0.05, 0.001+zoffs, xlabel)
    ob2.rotation_euler=[0,0,0]
    ob2.scale=[-0.1,-0.1,0.1]
    ob3=TXTBOX(0.1,0.01, 1.3+zoffs, zlabel)
    ob3.rotation_euler=[-d90,d90,0]
    ob3.scale=[0.125,0.125,0.125]
    xzc=TXTBOX(1.2,0.001, 1.15+zoffs, 'r=%.2f,p=%.3f'%(pearsonr(x,z)))
    xzc.scale=[0.16,0.16,0.16]
    xzc.rotation_euler=[-d90,d90,0]
    yzc=TXTBOX(0.0,1.2, 1.15+zoffs, 'r=%.2f,p=%.3f'%(pearsonr(y,z)))
    yzc.scale=[0.16,0.16,0.16]
    yzc.rotation_euler=[0,d90,0]
    data=np.concatenate((Xn*(xgap[1]-xgap[0])+xgap[0],Yn*(ygap[1]-ygap[0])+ygap[0],Zn*(zgap[1]-zgap[0])+zgap[0]+zoffs)).reshape((3,-1)).T
    M=data.mean(axis=0) 
    uu, dd, vv = np.linalg.svd(data - M)
    dir=vv[0]
    #ar=arrow(2.0*data.std())
    #ar.location=M-data.std()
    dir/=np.linalg.norm(dir)

    #ar.rotation_euler=[np.arctan(dir[1]/dir[0]) if dir[0]!=0 else d90,d90/2.0,np.arcsin(dir[2])]
    r,p=corr3(x,y,z)
    obf=TXTBOX(0,0,1.6+zoffs, '|r|=%.2f, p=%.3f'%(r, p))
    obf.rotation_euler=[d90,0,d90]
    obf.scale=[0.2,0.2,0.2]
        
     
        
def printcorr(x0,y0,z0):
    ixs=list(range(50))
    ixs.remove(6)
    ixs.remove(10)
    x=[x0[i] for i in ixs]
    y=[y0[i] for i in ixs]
    z=[z0[i] for i in ixs]
    print(('%.4f '*11)%tuple([min(x), max(x), min(y), max(y), min(z), max(z), pearsonr(x,y)[0], pearsonr(x,z)[0], pearsonr(y,z)[0]]+corr3(x,y,z)))

    #TXTbox(0,0,-0.2,xlabel,rot=[0,0,0], scale=0.2)
    #TXTbox(-0.2,0.3,-0.2,ylabel,rot=[0,0,90], scale=0.2)
    #TXTbox(-0.2,0.0,1.0,zlabel,rot=[0,90,0], scale=0.2)
    #TXTbox(0,0,-0.4,'r='+corr2_coeff(np.array(x).reshape([-1,1]),np.concatenate([y,z]).reshape([-1,2])))
    
#X2=np.array([bgs3[sind[i]] for i in range(len(rega))])
#printcorr(feara,laws,X2)
#printcorr(rega,laws,X2)



#printcorr(rega,feara,bgs3)
#printcorr(rega,laws,bgs3)
#printcorr(feara,laws,bgs3)
#printcorr(rega,laws,slopes)
#printcorr(feara,laws,slopes)
#printcorr(rega,feara,slopes)
#printcorr(feara,laws, slopes2)
#printcorr(rega,laws, slopes2)bn
#printcorr(rega,feara,slopes2)
#X2=
     
#X2=np.array([bgs3[ix_to_usps(sind[i])] for i in range(len(rega))])
#X1=np.concatenate([np.array(rega),np.array(feara),X2]).reshape([3,-1])
#C2=np.corrcoef(X1)
#print(C2)

    
#print(corr2_coeff(np.concatenate([feara, rega]).reshape([48,2]).T, bgs).reshape([1,-1])))
ixs=list(range(50))
ixs.remove(6)
ixs.remove(10)

#scplot(np.array(laws)[ixs], np.array(slopes)[ixs], np.array(rega)[ixs], 0.025, 'Legal rest.', 'NICS (2019)', 'Fear of Reg.', zoffs=-3)
scplot(np.array(laws)[ixs], np.array(slopes2)[ixs], np.array(rega)[ixs], 0.025, 'Legal rest.', 'NICS (1999-2019)', 'Fear of Reg.', zoffs=-6)
scplot(np.array(laws)[ixs],  np.array(bgs3)[ixs],np.array(rega)[ixs], 0.025, 'Legal rest.', 'Gallup', 'Fear of Reg.', zoffs=-3)
scplot(np.array(feara)[ixs], np.array(rega)[ixs], np.array(slopes)[ixs], 0.025, 'B.D.W.', 'F.O.R.', 'NICS (2019)', zoffs=-0)
#lplot(np.array(regd)[ix2], np.array(feard)[ix2], np.array(bgd)[ix2], 0.023, 'Fear of Reg.', 'B.D.W.', 'NICS \n(8/3/19-8/10/19)',zoffs=-12)

#TXTBOX('rega, feara: r=%.3f, p=%.3f'%pearsonr(rega,feara)[0]))
#TXTBOX('rega, feara: r=%.3f'%pearsonr(,feara)[0]))

#rate of change in gun bg checks data
#glmm
#maybe redo sentiment analysis
#hunger for guns
#internal meetings multiple times a week with presentations
#send update at the middle of the week

#document share
#raghu multilinear model and multiple correlation
#send email verifying gallup states
#kslote doing sent analysis
#slightly longer time horizon but should have something next week
#thursday 
#saturday presentation so need files
#wednesday: ;labrl

print('state, term freq(b.d.w.), term freq (f.o.reg.),  population (2019), bg checks (raw 8/2019), bg checks (8/2019 only--normalized),  bg checks (est. total increase in per capita guns since 1999), gallup')
for i in range(50):
        print(','.join(list(map(str,[statenames[i], feara[i], rega[i], pops[ix_to_usps(i,False),-1],  bg[-5,i], slopes2[i], slopes[i], bgs3[i]]))))

