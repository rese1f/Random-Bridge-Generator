# Tutorial
## 1. Install bpy Package
In most of the time, the Blender-Python API is used just inside of the Blender, if you use other text editor like VSCode or IDE like PyCharm, install the bpy package will help you to get code hint.
```bash
pip install fake-bpy-module-2.80
```

## 2. Blender Import Tutorial
### Preparation
Firstly, link all the python **files** you need in the blender text editor. 
### Implementation
For example, if you want to use the codes inside of the [member.py](Member/member.py) in [component.py](Component/component.py), you need to add these codes at the beginning of the [component.py](Component/component.py).
```python
import bpy

# import the member.py as a module
mb = bpy.data.texts["member.py"].as_module()

# Then you can use the classes or functions inside of the member.py in the target python file
rectangle_cfg = {
    'name': rectangle,
    'shape': {
        'Flange length': 2,
        'Web length': 1,
    }
}
rectangle = mb.Rectangle(rectangle_cfg)
```

## 3. Run Python script from the command line
The following demonstration is done on Windows.

Firstly, go to the directory where Blender is installed.
```bash
cd c:\<blender installation directory> # Replace the content in <> 
```
Use `blender [args ...] [file] [args ...]` to use Blender in command line.

For Python, to run the Python script file. Use `-P`, `--python` `<filepath>`
```bash
blender -P G:\srpp\Synthetic-Structural-Benchmark\BridgeGenerator\test_chj.py
```
## 4. Random Bridge Generator

### 4.1. Set CFG Dictionary

CFG dictionary consists of the basis parameters for future construction. It has two keys: “name” (the model that these parameters are applied to), and “shape” (all the parameters’ name together with their value). For example, the CFG dictionary for deck is called **`setDeckBasic`**, and it is constructed as
``` python
def setDeckBasic(w_deck, t_deck, h_deck):
    cfg = {
        "name": "deck_basic",
        "shape": {
            'deck width': w_deck,
            'deck thickness': t_deck,
            'deck top surface height': h_deck
        }
    }
    return cfg
```

### 3.2. Member Class

Meber class consists of three main functions. **`setMember()`** and **`setMember3d()`** are used to transfer the CFG variable into Blender vertices list and faces list. **`createObj()`** is used to transfer Blender vertices list and faces list into Blender object.

For argument of this class
| arg name | meaning |
| :------: | :------:|
|cfg| CFG dictionary|
|n| length of the object (x direction)|
|t| translation of the cross-section|
|quat| rotation in quaternion of the cross-sections|

For Member variables
| variable name | meaning |
| :------: | :------:|
|v| Blender vertices|
|f| Blender faces|
|npts| number of points in one cross-section|
|obj| Blender object|

#### 3.2.1 setMember

@rwh

#### 3.2.2 setMember3D

The main idea of this method is construct a Blender object using only two faces: start and end face. The advantage of this method is that it can not only input 2D coordinate but also 3D coodinate, and the shape of face can change as long as the number of points in faces keep the same. The disadvantage of this method is that it can't construct rotation and transformation. 

The code is as follow:
``` python
def setMember3d(self):

        self.v = []
        self.f = []

        if self.three_d == False:
            start = np.zeros([self.yz.shape[0], 3])
            start[:, 0] = -self.n/2
            start[:, 1:] = self.yz[:, :]
            
            end = np.zeros([self.yz_end.shape[0], 3])
            end[:, 0] = self.n/2
            end[:, 1:] = self.yz_end[:, :]
        
        else:
            start = self.yz
            
            end = self.yz_end        

        for i1 in range(start.shape[0]):
            self.v.append(start[i1, :])

        for i2 in range(end.shape[0]):
            self.v.append(end[i2, :])

        npts = start.shape[0]

        for j in range(npts):
            self.f = self.f + [(j, np.mod(j+1,npts),np.mod(j+1,npts) + npts, j + npts) for k in range(npts)]

        f1 = ()
        f2 = ()
        for i in range(npts):
            f1 += (i,)
            f2 += (npts + i,)
        self.f.append(f1)
        self.f.append(f2)
```

For function variable
| variable name | meaning |
| :------: | :------:|
|`yz`|coordinate of start face |
|`yz_end`| coodinate of end face|
|`three_d`| whether face coordinate is 2D or 3D (True: 3D)|

First change the input coordinate into 3D. If input coordinate is 2D, add x coordinate according to object length `n`. The coordinate of start face is transformed into `start`, end face is transformed into `end`. Code is following:
``` python
if self.three_d == False:
    start = np.zeros([self.yz.shape[0], 3])
    start[:, 0] = -self.n/2
    start[:, 1:] = self.yz[:, :]
    
    end = np.zeros([self.yz_end.shape[0], 3])
    end[:, 0] = self.n/2
    end[:, 1:] = self.yz_end[:, :]

else:
    start = self.yz
    
    end = self.yz_end   
```

Then, append all the points to `v` (Blender vertices):

``` python
for i1 in range(start.shape[0]):
    self.v.append(start[i1, :])

for i2 in range(end.shape[0]):
    self.v.append(end[i2, :])
```

Next, connect points of start and end to construct new faces:

``` python
npts = start.shape[0]

for j in range(npts):
    self.f = self.f + [(j, np.mod(j+1,npts),np.mod(j+1,npts) + npts, j + npts) for k in range(npts)]
```

Finally, cover the start and end faces:
```  python
f1 = ()
f2 = ()
for i in range(npts):
    f1 += (i,)
    f2 += (npts + i,)
self.f.append(f1)
self.f.append(f2)
```

#### 2.3 createObj
This part of code is used to create Blender object. The code is as follow:
``` python
def createObj(self, name, obj_num=1):
    vertices = self.v
    edges = []
    faces = self.f


    new_mesh = bpy.data.meshes.new("new_mesh")
    new_mesh.from_pydata(vertices, edges, faces)
    new_mesh.update()
    obj = bpy.data.objects.new(name, new_mesh)
    view_layer = bpy.context.view_layer
    view_layer.active_layer_collection.collection.objects.link(obj)

    self.obj = obj

```

## 3. Construct Cross-Section Coordinate
All these classes based on **`Member`** class create all the cross-section coordinate according to input CFG dictionary. Take **`ConcreteSolid`** class as an example:

``` python
class ConcreteSolid(Member):
    def __init__(self, cfg, cfg_end, n, three_d = False, t=None, quat=None):
        super().__init__(cfg, n, t, quat)

        self.three_d = three_d
        self.cfg_end = cfg_end
        self.shape_end = self.cfg_end['shape']

        w_deck = self.shape['deck width']
        t_deck = self.shape['deck thickness']
        h_deck_t = self.shape['deck top surface height']
        h_deck = h_deck_t - t_deck

        m = random.uniform(0, 1)
        self.yz = np.array([
            [(w_deck/2 - m), (h_deck)],
            [(w_deck/2), (h_deck + t_deck/2)],
            [(w_deck/2 - m), (h_deck + t_deck)],
            [-(w_deck/2 - m), (h_deck + t_deck)],
            [-(w_deck/2), (h_deck + t_deck/2)],
            [-(w_deck/2 - m), (h_deck)]
        ])

        self.yz_end = self.yz

        self.setMember3d()
```

For class arguments:
| arg name | meaning |
| :------: | :------:|
|cfg| CFG dictionary of start face|
|cfg_end| CFG dictionary of end face|
|n| length of the object (x direction)|
| three_d | whether face coordinate is 2D or 3D (True: 3D) | 
|t| translation of the cross-section|
|quat| rotation in quaternion of the cross-sections|

First, it reads CFG dictionary and create the coordinate. If `three_d` is True, construct 3D coordinate; if `three_d` is False, construct 2D coordinate.

``` python
self.three_d = three_d
self.cfg_end = cfg_end
self.shape_end = self.cfg_end['shape']

w_deck = self.shape['deck width']
t_deck = self.shape['deck thickness']
h_deck_t = self.shape['deck top surface height']
h_deck = h_deck_t - t_deck

m = random.uniform(0, 1)
self.yz = np.array([
    [(w_deck/2 - m), (h_deck)],
    [(w_deck/2), (h_deck + t_deck/2)],
    [(w_deck/2 - m), (h_deck + t_deck)],
    [-(w_deck/2 - m), (h_deck + t_deck)],
    [-(w_deck/2), (h_deck + t_deck/2)],
    [-(w_deck/2 - m), (h_deck)]
])

self.yz_end = self.yz
```
`m` here is used to create a little randomization.

Next, create `v` and `f` using `setMember3d()`.

Here is also an example of 3D coordinate.
``` python
class Triangle2(Member):
    def __init__(self, cfg, cfg_end, n, three_d=True, t=None, quat=None):
        super().__init__(cfg, n, t, quat)

        self.cfg_end = cfg_end
        self.three_d = three_d
        self.shape_end = self.cfg_end['shape']          

        a1 = self.shape['x1']
        a2 = self.shape['x2']
        a3 = self.shape['x3']
        b1 = self.shape['z1']
        b2 = self.shape['z2']
        b3 = self.shape['z3']
        c_start = self.shape['y']

        self.yz = np.array([
        [a1, c_start, b1],
        [a2, c_start, b2],
        [a3, c_start, b3]
        ])

        a1 = self.shape_end['x1']
        a2 = self.shape_end['x2']
        a3 = self.shape_end['x3']
        b1 = self.shape_end['z1']
        b2 = self.shape_end['z2']
        b3 = self.shape_end['z3']
        c_end = self.shape_end['y']

        self.yz_end = np.array([
        [a1, c_end, b1],
        [a2, c_end, b2],
        [a3, c_end, b3]
        ])        

        self.setMember3d()

```

## 4. Construct Blender Object

Blender objects are constructed according to different classifications: **`SuperStructure`**, **`SubStructure`**, **`Deck`**, **`Bearing`**. Each classification is one parent class, and specific structure is children class. Take cable-stayed bridge as an example, **`Cable`** is the children class of **`SuperStructure`**, **`Column`** is the children class of **`SuperStructure`**, **`Slab`** is the children class of **`Deck`**, **`CableBase`** and **`CableTop`** is the children class of **`Bearing`**.  Each parent class return the Blender object constructed by its children class. For example, **`SuperStructure`** return `column`, which is Blender column object.

First, it is necessary to introduce one auxiliary function, **`Hollow()`**. The code is as follow:

``` python
def Hollow(big_obj, small_obj):
    hollow = big_obj.modifiers.new("MyModifier", "BOOLEAN")
    hollow.object = small_obj

    # blender need first activate the object, then apply modifier
    bpy.context.view_layer.objects.active = big_obj
    bpy.ops.object.modifier_apply(modifier="MyModifier")

    # blender need first choose the object, then delete
    small_obj.select_set(True)
    bpy.ops.object.delete()
```

The arguments for this function is **big_obj** and **small_obj**, which are the origin Blender object and the object used to hollow the origin object. This function creates a hollowed origin Blender object.

#### 4.1 SuperStructure

##### 4.1.1 Cable
This class create the cable object by connecting the start circle face and end circle face. Class arguments are as follow:

| arg name | meaning |
| :------: | :------:|
|cable_start|coordinate of circle center for start face|
|cable_end|coordinate of circle center for end face|
|name|the name of the object|
|cable_radius|radius of cable|
|num|number of points used to approximate the circle| 

#### 4.2 SubStructure

##### 4.2.1 Column
This class create different shape of column. Current column shape include **`A1`**, **`double`**, **`door`**, **`tower`**, each column shape is a class function. Class arguments are as follow:

| arg name | meaning |
| :------: | :------:|
|w_column| width of column |
|h_column| height of column|
|t_column| thickness of column (in yz plane)|
|l_column| length of column (in x direction)|
|h_deck_t| height of deck top surface|
|t_deck| thickness of deck |
|name| name of Blender object|

**`Column`** class has one variable: `cable_function`. It is used to locate the cable contact point on the column and help to calculate the proper width of deck. This variable is a list that consists of six elements: [`z_start`, `z_end`, `k`, `b`, `b_in`, `y_cable`]. The contact point on the column can be connected as a strict line in yz plane. `z_start` is the z coordinate of the start position for the cable contact position; `z_end` is the z coordinate for the end position for the cable contact position; `k` and `b` is the slope and intercept for this cable contact straight line (only consider positive `k`). The position that the side of deck in contact with column also has a track of straight line, and it is parallel to cable contact straight line in current situation. `k` and `b_in` is the slope and intercept for this straight line track. However, cable contact straight line is parallel to z axis, `k` no longer exist, so `k`, `b` are directly set as 0, and the function of cable contact straight line is y =  `y_cable`.

This class use **`Hollow`** function a lot. For example, **`A1`** column is constructed by original `A1` object hollowed by a triangle object.

#### 4.3 Deck
##### 4.3.1 Slab
This class create different shape of slab. Current column shape include **`concrete_solid`**, **`concrete_PK`**, **`concrete_box`**, **`concrete_costalia`**, **`steel_box`**, **`steel_sidebox`**, **`truss`** each column shape is a class function. Class arguments are as follow:

| arg name | meaning |
| :------: | :------:|
|t_deck| thickness of deck|
|h_deck|height of deck top surface|
|l_deck|length of deck (x direction)|
|w_column|width of column|
|t_column| thickness of column (in yz plane)|
|name| name of Blender object|
|w_deck|width of deck|

For class variable
| class variable name | meaning |
| :------: | :------:|
|T_truss| thickness of truss|

Class function **`truss`** has little different with other class funtion in this class. Basically, for first step, a bigger cuboid is **`Hollow`** by a smaller cuboid in x direction. For second step, the hollowed shape is **`Hollow`** by several other triangular columns in y direction. Through this way, a truss is constructed.  

For function variable,
| function variable name | meaning |
| :------: | :------:|
|v_width|thickness of wall in yz plane z direction|
|h_width|thickness of wall in yz plane y direction|
|thick_bar|thickness of truss bar in xz plane|
|width_bar|width of one truss unit in x direction|

For first step, the code is as follow:
``` python
v_width = 0.25
h_width = 0.25

cfg_start = setRectangle(self.W_deck, self.T_truss, h)  
cfg_end = cfg_start

cfg_hollow_start = setRectangle(self.W_deck - 2*h_width, self.T_truss - 2*v_width, h - v_width) 
cfg_hollow_end = cfg_hollow_start

orig = Rectangle(cfg_start, cfg_end, self.L_deck)
orig.createObj(self.name)
hollow = Rectangle(cfg_hollow_start, cfg_hollow_end, self.L_deck + 5)
hollow.createObj('hollow') 

Hollow(orig.obj, hollow.obj)
```

For second step, the code is as follow:
``` python
thick_bar = v_width * 1.5
l = self.L_deck - thick_bar
width_bar = 4
height_bar = T_truss

a11 = -l/2 + thick_bar/2
b11 = thick_bar + self.H_deck - self.T_deck
a12 = -l/2 + width_bar - thick_bar/2
b12 = thick_bar+ self.H_deck - self.T_deck
a13 = -l/2 + thick_bar/2
b13 = height_bar - thick_bar -  height_bar/width_bar * thick_bar + self.H_deck - self.T_deck

a23 = -l/2 + thick_bar/2
b23 = height_bar - thick_bar + self.H_deck - self.T_deck
a22 = -l/2 + width_bar - thick_bar/2
b22 = height_bar - thick_bar + self.H_deck - self.T_deck
a21 = -l/2 + width_bar - thick_bar/2 
b21 = thick_bar + height_bar/width_bar * thick_bar + self.H_deck - self.T_deck    

for i in range(int(l/width_bar)):
# for i in range(1):    
    name = 'down_hollow' + str(i)
    cfg1_start = setTriangle2(a11 + width_bar*i, b11, a12+ width_bar*i, b12, a13+ width_bar*i, b13, 10) 
    cfg1_end = setTriangle2(a11+ width_bar*i, b11, a12+ width_bar*i, b12, a13+ width_bar*i, b13, -10)
    tria1 = Triangle2(cfg1_start, cfg1_end, 1)
    tria1.createObj(name)
    Hollow(orig.obj, tria1.obj)

for i in range(int(l/width_bar)):
# for i in range(1):
    name = 'up_hollow' + str(i)
    cfg2_start = setTriangle2(a21 + width_bar*i, b21, a22+ width_bar*i, b22, a23+ width_bar*i, b23, 10) 
    cfg2_end = setTriangle2(a21+ width_bar*i, b21, a22+ width_bar*i, b22, a23+ width_bar*i, b23, -10)
    tria2 = Triangle2(cfg2_start, cfg2_end, 1)
    tria2.createObj(name)
    Hollow(orig.obj, tria2.obj)             

self.slab = orig.obj
```

#### 4.4 Bearing
##### 4.4.1 CableBase
**`CableBase`** is the connection for cable and slab. For class arguments,
| function variable name | meaning |
| :------: | :------:|
|t1 |currently not used|
|t2|the length of cylinder burried inside|
|t3|the length of cylinder exposed outside|
|r2| radius of cylinder|
|name|name of Blender object|
|turn|make sure CableBase point to correct direction|

##### 4.4.2 CableTop
**`CableTop`** is the connection for cable and column. For class arguments,
| function variable name | meaning |
| :------: | :------:|
|t1 |currently not used|
|t2|the length of cylinder burried inside|
|t3|the length of cylinder exposed outside|
|r1| radius of cylinder|
|name|name of Blender object|
|turn|make sure CableTop point to correct direction|


## 5. Put All the Part Together
In this part, the Blender objects created in part4 are put together by setting their size, location and rotation. 
For class argument,
| arg name | meaning |
| :------: | :------:|
|num_column|number of column|
|h_column| height of column|
|t_column| thickness of column in yz plane|
|l_column| length of column in x direction|
|w_column| width of column|
|h_deck| height of deck top surface|
|t_deck| thickness of deck|
|index_column| the choice of column type|
|index_deck| the choice of deck type|
|index_cable|the choice cable type|
|face_cable|number of cable plane|
|num_cable|number of cable in single cable plane|
|truss| whether slab is truss|

For class variable,
| class variable| meaning|
| :------: | :------:|
|dist_column|distance between two column|
|W_deck|width of deck|
|cable_funciton| introduced in 4.2.1|
|cable_top| all the coordicate for circle center at top of cable|
|cable_bottom| all the coordinate for circle center at bottom of cable|

During initialization, according to cable-stayed bridge standard, `L_deck` is randomized as follow:
``` python
if self.num_column == 1:
    a = random.uniform(1.5, 3)
    self.L_deck = a * self.H_column
else:
    a = random.uniform(3, 6)
    self.dist_column = a * self.H_column
    self.L_deck = 2 * self.dist_column
```

#### 5.1 column
In this class function, column object is created according to different `index_column`:

``` python
def column(self):
    for i in range(self.num_column):        
        if self.index_column == 1:
            member = Column(self.W_column, self.H_column, self.T_column, self.L_column, self.H_deck, self.T_deck, 'A1 column')
            member.A1()

        elif self.index_column == 2:
            member = Column(self.W_column, self.H_column, self.T_column, self.L_column, self.H_deck, self.T_deck, 'double column')
            member.double()

        elif self.index_column == 3:
            member = Column(self.W_column, self.H_column, self.T_column, self.L_column, self.H_deck, self.T_deck, 'door column')
            member.door()
        
        elif self.index_column == 4:
            member = Column(self.W_column, self.H_column, self.T_column, self.L_column, self.H_deck, self.T_deck, 'tower column')
            member.tower()

        self.cable_function = member.cable_function
        member.column.location.x = (-1)**(i+1) * self.dist_column/2
```
The last command `location.x` is a Blender-python API command used to set the location of Blender object.

#### 5.2 deck

The code is as follow:
``` python
def deck(self):
    k = self.cable_function[2]
    b_in = self.cable_function[4]
    h = self.H_deck + self.T_deck
    if b_in == 0:
        self.W_deck = self.W_column - self.T_column * 2
    else:
        self.W_deck = - (h - b_in)/k * 2  

    if self.tru == 0:
        if self.index_deck == 1:
            member = Slab(self.T_deck, self.H_deck, self.L_deck, self.W_column, self.T_column, 'comcrete solid deck', None, self.cable_function)
            member.concrete_solid()
        
        elif self.index_deck == 2:
            member = Slab(self.T_deck, self.H_deck, self.L_deck, self.W_column, self.T_column, 'comcrete PK deck', None, self.cable_function)
            member.concrete_PK()
        
        elif self.index_deck == 3:
            member = Slab(self.T_deck, self.H_deck, self.L_deck, self.W_column, self.T_column, 'comcrete box deck', None, self.cable_function)
            member.concrete_box()
        
        elif self.index_deck == 4:
            member = Slab(self.T_deck, self.H_deck, self.L_deck, self.W_column, self.T_column, 'comcrete constalia deck', None, self.cable_function)
            member.concrete_costalia()

        elif self.index_deck == 5:
            member = Slab(self.T_deck, self.H_deck, self.L_deck, self.W_column, self.T_column, 'steel box deck', None, self.cable_function)
            member.steel_box() 

        elif self.index_deck == 6:
            member = Slab(self.T_deck, self.H_deck, self.L_deck, self.W_column, self.T_column, 'steel sidebox deck', None, self.cable_function)
            member.steel_sidebox()
    
    elif self.tru == 1:
        member = Slab(self.T_deck, self.H_deck, self.L_deck, self.W_column, self.T_column, 'truss', None, self.cable_function) 
        member.truss()
```

In this function, `W_deck` is set at first according to `cable_function`, in case shape of deck crush into shape of column. And then, the shape of deck is choosen according to `index_deck`.

#### 5.3 cable
There are three different type of cable used in this situation:
1. Fan or Intermediate Cable System Schematic: cable is not parallel to each other in xz plane. All cables' top and bottom coordinate are different.
2. Harp or Parallel Cable System Schematic: cable is parallel to each other in xz plane.
3. Radial or Converging Cable System Schematic: cable is not parallel to each other in xz plane. cables at same cable unit shape same cable top coordinate.

For function variable:
| function variable name | meaning |
| :------: | :------:|
|column_loc| x coordinate for center of column|
|dist_top| distance between circle center coordinate at top of cable (z direction)|
|dist_bottom| distance between circle center coordinate at bottom of cable (x direction)|


The code for **`cable`** is as follow:
``` python
def cable(self):
    z_start = self.cable_function[0]
    z_end = self.cable_function[1]
    k = self.cable_function[2]
    b = self.cable_function[3]
    y_top = self.cable_function[5]

    
    column_loc = np.zeros(self.num_column)
    for i in range(self.num_column):
        column_loc[i] = (-1)**(i+1) * self.dist_column/2

    # top right side
    dist_top = (z_end - z_start) / (self.num_cable - 1)   
    y_cable_top0 = np.zeros([self.num_cable, 1])
    z_cable_top0 = np.zeros([self.num_cable, 1])
    z_rand = random.uniform(z_start + (z_end - z_start)/4, z_end - (z_end - z_start)/4) ## for cable index3
    for i in range(self.num_cable):
        if self.index_cable == 3:
            z_cable_top0[i] = z_rand
            dist_top = 0
        else:
            z_cable_top0[i] = z_end - i * dist_top
        
        if self.face_cable == 1:
            y_cable_top0[i] = 0
        elif self.face_cable == 2:
            if b == 0:
                y_cable_top0[i] = y_top
            else:
                y_cable_top0[i] = -(z_end - i * dist_top - b)/k

    
    x_cable_top = np.ones([self.num_column, self.num_cable*2]) * column_loc.reshape([self.num_column, 1]) # *2: front and back
    x_cable_top = x_cable_top.reshape([-1, 1])
    for i in range(self.num_column): ## adjust the top part of cable just in touch with the surface of column
        index_even = i*2
        index_odd = i*2 + 1
        x_cable_top[(index_odd*self.num_cable) : ((index_odd+1)*self.num_cable)] += (self.L_column/2)
        x_cable_top[(index_even*self.num_cable) : ((index_even+1)*self.num_cable)] -= (self.L_column/2) 

    yz_cable_top0 = np.concatenate((y_cable_top0, z_cable_top0), 1)
    yz_cable_top = yz_cable_top0
    for i in range(self.num_column*2 - 1):
        yz_cable_top = np.concatenate((yz_cable_top, yz_cable_top0), 0)
    
    cable_top_right = np.concatenate((x_cable_top, yz_cable_top), 1)

    # bottom right side
    if self.tru == 1:
        z_cable_bottom = self.H_deck - self.T_deck + self.T_truss
    elif self.tru ==0:    
        z_cable_bottom = self.H_deck

    if self.face_cable == 2:
        print(self.W_deck)  
        y_cable_bottom = self.W_deck/2 * (7/10)
    elif self.face_cable == 1:
        y_cable_bottom = 0

    a = random.uniform(7/8, 9/10)  
    x_L = self.L_deck / self.num_column / 2 * a ## outer distance of cable at bottom

    if self.index_cable == 2:
        k_cable = (z_end - z_cable_bottom)/x_L
        dist_bottom = dist_top / k_cable ## distance between two cable in x direction 
    else:
        b = random.uniform(1/7, 1/5)
        # x_L_in = b * x_L
        dist_bottom = x_L / self.num_cable

    x_cable_bottom = np.zeros([self.num_column, self.num_cable*2]) ## one row represents x coordinate of all cable for one column
    loc = np.hstack((np.linspace(-self.num_cable, -1, self.num_cable), np.linspace(self.num_cable, 1, self.num_cable)))  ## help to locate x coordinate
    for i in range(self.num_column):
        x_cable_bottom[i, :] = column_loc[i] + dist_bottom * loc

    x_cable_bottom = x_cable_bottom.reshape([-1, 1])
    y_cable_bottom = x_cable_bottom * 0 + y_cable_bottom
    z_cable_bottom = x_cable_bottom * 0 + z_cable_bottom
    cable_bottom_right = np.concatenate((x_cable_bottom, y_cable_bottom, z_cable_bottom), 1)

    if self.face_cable == 1:
        cable_top = cable_top_right
        cable_bottom = cable_bottom_right

        for i in range(self.num_column * self.num_cable * 2):
            Cable(cable_bottom[i, :], cable_top[i, :], "cable" + str(i+1))

    elif self.face_cable == 2:
        cable_top_left = cable_top_right * np.array([[1, -1, 1]])
        cable_bottom_left = cable_bottom_right * np.array([[1, -1, 1]])
        cable_top = np.concatenate((cable_top_left, cable_top_right), 0)
        cable_bottom = np.concatenate((cable_bottom_left, cable_bottom_right), 0)        

        for i in range(self.num_column * self.num_cable * 4):
            Cable(cable_bottom[i, :], cable_top[i, :], "cable" + str(i+1))

    self.cable_top = cable_top
    self.cable_bottom = cable_bottom
```

For the detail of this code, first some constrain for cable position is set according to `cable_function` and `column_loc`. 

``` python
z_start = self.cable_function[0]
z_end = self.cable_function[1]
k = self.cable_function[2]
b = self.cable_function[3]
y_top = self.cable_function[5]


column_loc = np.zeros(self.num_column)
for i in range(self.num_column):
    column_loc[i] = (-1)**(i+1) * self.dist_column/2
```

And then, the coordinate of circle center for top face of cable is set. First, set the circle center y and z coordinate for top face of every cable. If it is type3 cable, all coordinates have y = 0; for other two type, it is the right face of cable, which means  y > 0. 

``` python
dist_top = (z_end - z_start) / (self.num_cable - 1)   
y_cable_top0 = np.zeros([self.num_cable, 1])
z_cable_top0 = np.zeros([self.num_cable, 1])
z_rand = random.uniform(z_start + (z_end - z_start)/4, z_end - (z_end - z_start)/4) ## for cable index3
for i in range(self.num_cable):
    if self.index_cable == 3:
        z_cable_top0[i] = z_rand
        dist_top = 0
    else:
        z_cable_top0[i] = z_end - i * dist_top
    
    if self.face_cable == 1:
        y_cable_top0[i] = 0
    elif self.face_cable == 2:
        if b == 0:
            y_cable_top0[i] = y_top
        else:
            y_cable_top0[i] = -(z_end - i * dist_top - b)/k
```
Second, set the x coordinate for top face of cable. The little loop here is to make sure the contact point for cable and column is just at the surface of column.

``` python
x_cable_top = np.ones([self.num_column, self.num_cable*2]) * column_loc.reshape([self.num_column, 1]) # *2: front and back
x_cable_top = x_cable_top.reshape([-1, 1])
for i in range(self.num_column): ## adjust the top part of cable just in touch with the surface of column
    index_even = i*2
    index_odd = i*2 + 1
    x_cable_top[(index_odd*self.num_cable) : ((index_odd+1)*self.num_cable)] += (self.L_column/2)
    x_cable_top[(index_even*self.num_cable) : ((index_even+1)*self.num_cable)] -= (self.L_column/2) 
```
Third, concatenate all the x, y, z coordinate to construct one-side top face coordinate. 
``` python
yz_cable_top0 = np.concatenate((y_cable_top0, z_cable_top0), 1)
yz_cable_top = yz_cable_top0
for i in range(self.num_column*2 - 1):
    yz_cable_top = np.concatenate((yz_cable_top, yz_cable_top0), 0)

cable_top_right = np.concatenate((x_cable_top, yz_cable_top), 1)
```

After all of this, one-side bottom face coordinate is constructed. First, z coordinate for all bottom face is set according to the whether the slab is a truss or not. Because truss and other slab have different thickness. And y coordinate is set according to whether cable face is 1 or 2.

``` python
if self.tru == 1:
    z_cable_bottom = self.H_deck - self.T_deck + self.T_truss
elif self.tru ==0:    
    z_cable_bottom = self.H_deck

if self.face_cable == 2:
    print(self.W_deck)  
    y_cable_bottom = self.W_deck/2 * (7/10)
elif self.face_cable == 1:
    y_cable_bottom = 0
```
Third, x coordinate is set. `x_L` here is the largest distance a bottom face of cable can be to it nearest column. `dist_bottom` is set according to whether cable is parallel to each other in xz plane (type2 cable). According these two parameters, x coordinate is constructed.

``` python
a = random.uniform(7/8, 9/10)  
x_L = self.L_deck / self.num_column / 2 * a ## outer distance of cable at bottom

if self.index_cable == 2:
    k_cable = (z_end - z_cable_bottom)/x_L
    dist_bottom = dist_top / k_cable ## distance between two cable in x direction 
else:
    b = random.uniform(1/7, 1/5)
    # x_L_in = b * x_L
    dist_bottom = x_L / self.num_cable

x_cable_bottom = np.zeros([self.num_column, self.num_cable*2]) ## one row represents x coordinate of all cable for one column
loc = np.hstack((np.linspace(-self.num_cable, -1, self.num_cable), np.linspace(self.num_cable, 1, self.num_cable)))  ## help to locate x coordinate
for i in range(self.num_column):
    x_cable_bottom[i, :] = column_loc[i] + dist_bottom * loc
```

Fourth, concatenate all x, y, z coordinate for one-side cable bottom face.
``` python
x_cable_bottom = x_cable_bottom.reshape([-1, 1])
y_cable_bottom = x_cable_bottom * 0 + y_cable_bottom
z_cable_bottom = x_cable_bottom * 0 + z_cable_bottom
cable_bottom_right = np.concatenate((x_cable_bottom, y_cable_bottom, z_cable_bottom), 1)
```
Finally, complete all the coordinate for top face and bottom face according to the cable face number, and get `cable_top` and `cable_bottom`.
``` python
if self.face_cable == 1:
    cable_top = cable_top_right
    cable_bottom = cable_bottom_right

    for i in range(self.num_column * self.num_cable * 2):
        Cable(cable_bottom[i, :], cable_top[i, :], "cable" + str(i+1))

elif self.face_cable == 2:
    cable_top_left = cable_top_right * np.array([[1, -1, 1]])
    cable_bottom_left = cable_bottom_right * np.array([[1, -1, 1]])
    cable_top = np.concatenate((cable_top_left, cable_top_right), 0)
    cable_bottom = np.concatenate((cable_bottom_left, cable_bottom_right), 0)        

    for i in range(self.num_column * self.num_cable * 4):
        Cable(cable_bottom[i, :], cable_top[i, :], "cable" + str(i+1))

self.cable_top = cable_top
self.cable_bottom = cable_bottom
```

#### 5.4 cablebase & cabletop
For both of these two components, first use **`CableBase`** and **`CableTop`** class to construct the Blender object, and then use `rotation_euler` Blender command to orient the object according to cable orientation, together with `location` command to set the correct position. Code is as follow.

``` python
def cablebase(self):
    t1 = random.uniform(0.8, 1.4) # need further revise
    t2 = random.uniform(0.4, 0.7)
    t3 = random.uniform(0.8, 1.5)
    r2 = 0.15
    for i in range(self.cable_top.shape[0]):
        kxy = (self.cable_top[i, 1] - self.cable_bottom[i, 1])/(self.cable_top[i, 0] - self.cable_bottom[i, 0])
        kxz = (self.cable_top[i, 2] - self.cable_bottom[i, 2])/(self.cable_top[i, 0] - self.cable_bottom[i, 0])
        kyz = (self.cable_top[i, 2] - self.cable_bottom[i, 2])/(self.cable_top[i, 1] - self.cable_bottom[i, 1])                        
        turn = kxz/abs(kxz)

        member = CableBase(t1, t2, t3, r2, 'cable_base' + str(i+1), turn)
        base = member.cable_base

        theta_z = math.atan(kxy)
        theta_y = math.atan(kxz)
        theta_x = math.atan(kyz)
        base.rotation_euler[0] = -theta_x
        base.rotation_euler[1] = -theta_y
        base.rotation_euler[2] = theta_z
        base.location.x = self.cable_bottom[i, 0]
        base.location.y = self.cable_bottom[i, 1]                        
        base.location.z = self.cable_bottom[i, 2]

def cabletop(self):
    t1 = random.uniform(0.8, 1.4) # need further revise
    t2 = random.uniform(0.4, 0.7)
    t3 = random.uniform(0.8, 1.5)
    r1 = 0.08
    for i in range(self.cable_top.shape[0]):
        kxy = (self.cable_top[i, 1] - self.cable_bottom[i, 1])/(self.cable_top[i, 0] - self.cable_bottom[i, 0])
        kxz = (self.cable_top[i, 2] - self.cable_bottom[i, 2])/(self.cable_top[i, 0] - self.cable_bottom[i, 0])
        kyz = (self.cable_top[i, 2] - self.cable_bottom[i, 2])/(self.cable_top[i, 1] - self.cable_bottom[i, 1])                        
        turn = kxz/abs(kxz)

        member = CableBase(t1, t2, t3, r1, 'cable_top' + str(i+1), turn)
        top = member.cable_base

        theta_z = math.atan(kxy)
        theta_y = math.atan(kxz)
        theta_x = math.atan(kyz)
        top.rotation_euler[0] = -theta_x
        top.rotation_euler[1] = -theta_y
        top.rotation_euler[2] = theta_z
        top.location.x = self.cable_top[i, 0]
        top.location.y = self.cable_top[i, 1]                        
        top.location.z = self.cable_top[i, 2] 
```