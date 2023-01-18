# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 15:17:23 2022

@author: naraz
"""

import sys
import os
import random
import numpy as np
from scipy.spatial.transform import Rotation as R
import bpy

# add path to the python scripts
# blend file should be in the same folder as the one containing the scripts
# bpy.data.filepath = "C:\Users\19461\OneDrive - International Campus, Zhejiang University\Desktop\temp\images"

path = os.path.dirname(bpy.data.filepath)
if not path in sys.path:
    sys.path.append(path)


#####################
# some utility functions
# function for "clear all" - not only objects but also scenes, materials etc.
def Clear_all(clear_scene=True):
    # escape edit mode
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='OBJECT')
        
    # delete all scenes except for main one
    if clear_scene:
        bpy.data.scenes.new('Scene_tmp')
        for i in bpy.data.scenes.values()[:-1]:
            bpy.data.scenes.remove(i)
        for i in bpy.data.worlds.values():
            bpy.data.worlds.remove(i)
        bpy.data.scenes[0].name='Scene'
        bpy.data.worlds.new("World")
        bpy.data.scenes[0].world = bpy.data.worlds[0]
        # collection = bpy.context.blend_data.collections.new(name='Collection')
        # bpy.context.collection.children.link(collection)
    
    # remove collections
    for c in bpy.data.collections:
        #if c.users == 0:
        bpy.data.collections.remove(c)
        # collection = bpy.context.blend_data.collections.new(name='Collection')
        # bpy.context.collection.children.link(collection)
        
    # delete all cameras
    for c in bpy.data.cameras:
        #if c.users == 0:
        bpy.data.cameras.remove(c)
     
    #remove other (unlinked) objects (mostly empty objects)
    for obj in bpy.data.objects:
        #if obj.users == 0:
        bpy.data.objects.remove(obj)
            
    # delete mesh
    for m in bpy.data.meshes:
        #if m.users == 0:
        bpy.data.meshes.remove(m)
        
    # delete curves
    for c in bpy.data.curves:
        #if c.users == 0:
        bpy.data.curves.remove(c)
        
    # delete lamps
    for l in bpy.data.lights:
        #if l.users == 0:
        bpy.data.lights.remove(l)
        
    # delete all materials
    for m in bpy.data.materials:
        #if m.users == 0:
        bpy.data.materials.remove(m)
    
    # delete all textures
    for t in bpy.data.textures:
        #if t.users == 0:
        bpy.data.textures.remove(t)
    
    # delete all images 
    for i in bpy.data.images:
        #if i.users == 0:
        bpy.data.images.remove(i)
        
# miscellaneous functions - sky, sun etc.
def Add_Sky(w=None,turbidity=None,strength = None):
    if w is None:
        w = bpy.context.scene.world
    if turbidity is None:
        turbidity = 1+9*np.random.rand()
    if strength is None:
        strength = 0.1+0.4*np.random.rand()
        # strength = 0.5+0.4*np.random.rand()
        
    w.use_nodes = True
    nt = w.node_tree
    nodes = nt.nodes
    links = nt.links
    
    # clear
    while(nodes): nodes.remove(nodes[0])
    
    output  = nodes.new("ShaderNodeOutputWorld")
    bg  = nodes.new("ShaderNodeBackground")
    bg.inputs['Strength'].default_value = strength
    sky = nodes.new("ShaderNodeTexSky")
    
    links.new( output.inputs['Surface'], bg.outputs['Background'])
    links.new( bg.inputs['Color'], sky.outputs['Color'])
    
    return w, turbidity, strength

def Add_Sun(strength=None,quat=None,dang = 0.03,name='SUN'):
    if strength is None:
        # strength = 2.0 + 4.0*np.random.rand()
        strength = 0.08 + 0.08*np.random.rand()
    if quat is None:
        quat = np.random.rand(4)
        quat = quat / np.linalg.norm(quat)
        Rot = R.from_quat(quat)
        vnew = Rot.apply(np.array([0.0,0.0,1.0]))
        if vnew[-1] < 0:
            Rot = R.from_euler('X',np.pi) * Rot
            quat = Rot.as_quat()
    
    l = bpy.data.lights.new(name,type='SUN')
    l.energy = strength
    l.angle = dang
    o = bpy.data.objects.new(name="SUN", object_data=l)
    bpy.context.collection.objects.link(o)
    o.rotation_mode = 'QUATERNION'
    o.rotation_quaternion = (quat[3],quat[0],quat[1],quat[2])
    return o, strength

def WorldRGB(rgb,w=None):
    # a function to create a world with the uniform color
    # this function is used to create a scene for generating annotation
    if w is None:
        w = bpy.context.scene.world
        
    w.use_nodes = True
    nt = w.node_tree
    nodes = nt.nodes
    links = nt.links
    
    # clear
    while(nodes): nodes.remove(nodes[0])
    
    output  = nodes.new("ShaderNodeOutputWorld")
    bg  = nodes.new("ShaderNodeBackground")
    bg.inputs['Color'].default_value[0] = rgb[0]
    bg.inputs['Color'].default_value[1] = rgb[1]
    bg.inputs['Color'].default_value[2] = rgb[2]
    
    links.new( output.inputs['Surface'], bg.outputs['Background'])

################################################################################
################################################################################
# Addition for texturing
def getTextureLists(key2mat,path_tex):
    # this function creates a dictionary (tex_assign) that maps blender object name
    # to the path to the texture images (random assignment using key2mat)
    
    # extract available texture folders
    dirs = [d for d in os.listdir(path_tex) if os.path.isdir(os.path.join(path_tex,d))]
    tex = {} # Keys: material type, values: specific subfolder names
    for d in dirs:
        d_full = os.path.join(path_tex,d)
        tex[d] = [f for f in os.listdir(d_full) if os.path.isdir(os.path.join(d_full,f))]
    
    # create repositories for each structural component
    Keys = list(key2mat.keys())
    Keys.sort(key=len) # sort by length
    tex_assign = {}
    for objname in bpy.data.objects.keys(): 
        for k in Keys[::-1]: #longer to shorter (e.g., cable_top ==> cable)
            if k in objname.lower():
                mat = key2mat[k][0] # possible materials
                matidx = np.random.randint(len(mat))
                prefix = os.path.join(path_tex,mat[matidx])
                folders = tex[mat[matidx]] # select one randomly from this list
                fidx = np.random.randint(len(folders))
                tex_folder = os.path.join(prefix,folders[fidx])
                tex_assign[objname] = (tex_folder,key2mat[k][1])
                break
        if objname not in tex_assign.keys():
            print("Warning: texture was not assigned for "+objname)
    return tex_assign

def DefMat_Visual(name,path_tex):
    # define blender material based on the provided material names and texture images
    
    s = 0.1 + np.random.rand()*0.9 #random texture scale
    #find albedo, roughness, metallic, and height images
    tex = {}
    for f in os.listdir(path_tex):
        if 'albedo' in f.lower() or 'color' in f.lower() or 'diffuse' in f.lower():
            tex['albedo'] = os.path.join(path_tex,f)
        elif 'roughness' in  f.lower():
            tex['roughness'] = os.path.join(path_tex,f)
        elif 'height' in f.lower() or 'displacement' in f.lower():
            tex['height'] = os.path.join(path_tex,f)
        elif 'metallic' in f.lower() or 'metalness' in f.lower():
            tex['metallic'] = os.path.join(path_tex,f)
    
    # if the appropriate images are not found, set default values or raise an error
    if 'albedo' not in tex.keys():
        print('Error: texture image (albedo) not found!! - '+path_tex)
        return
    if 'roughness' not in tex.keys():
        tex['roughness'] = 1.0
        print('Warning: texture image (roughness) not found!! Default value (1) is used - '+path_tex)
    if 'height' not in tex.keys():
        tex['height'] = None
        print('Warning: texture image (height) not found!! Default value (None) is used - '+path_tex)
    if 'metallic' not in tex.keys():
        tex['metallic'] = 0.0
        print('Warning: texture image (metallic) not found!! Default value (0) is used - '+path_tex)
    
    # define new material
    if name not in bpy.data.materials.keys():
        bpy.data.materials.new(name)
        # configure a principled shader
        PrincipledShader(name,tex,Scale=(s,s,s))
    else:
        print("Warning: material \""+name+"\" was not created because there exist a material with the same name.")
    
def DefMat_Label(name,rgb,intensity=0.5):
    # define blender material for label maps (emission shader)
    
    # define new material
    if name not in bpy.data.materials.keys():
        bpy.data.materials.new(name)
        # configure a principled shader
        EmissionShader(name,intensity,rgb)
    else:
        print("Warning: material \""+name+"\" was not created because there exist a material with the same name.")
    
        
def PrincipledShader(matname,tex,Scale=(1.0,1.0,1.0),Rot=(0,0,0),ds = 1.0,Inverse=False):
    # ds: distance of the height map from the reference surface (intensity)
    # Inverse=True ==> invert the height map
    
    m = bpy.data.materials[matname]
    m.use_nodes = True
    
    nt = m.node_tree
    nodes = nt.nodes
    links = nt.links
    
    # clear
    while(nodes): nodes.remove(nodes[0])
    
    output  = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled") 
    coord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    links.new( output.inputs['Surface'], principled.outputs['BSDF'])
    links.new(coord.outputs['UV'],   mapping.inputs['Vector'])
    mapping.vector_type = 'TEXTURE'
    for i in range(3):
        mapping.inputs['Scale'].default_value[i]=Scale[i]
        mapping.inputs['Rotation'].default_value[i]=Rot[i]
    
    Keys = ['Base Color','Metallic','Roughness']
    files = [tex['albedo'],tex['metallic'],tex['roughness']]
    texNd = {} # texture nodes
    for i in range(len(Keys)):
        k = Keys[i]
        if type(files[i]) is str: # file name is specified
            texNd[k]=[]
            texNd[k] = nodes.new("ShaderNodeTexImage")
            links.new(texNd[k].outputs['Color'],   principled.inputs[k])
            #image texture
            texNd[k].image = bpy.data.images.load(files[i])
            links.new(mapping.outputs['Vector'],   texNd[k].inputs['Vector'])
        else: # default value (scalar) is specified
            principled.inputs[k].default_value = files[i]
            
    # set height map
    bump = nodes.new("ShaderNodeBump")
    bump.invert = Inverse
    bump.inputs['Distance'].default_value = ds
    links.new( principled.inputs['Normal'], bump.outputs['Normal'])
    Keys = ['Height']
    files = [tex['height']]
    for i in range(len(Keys)):
        k = Keys[i]
        if files[i] is not None:
            tex[k]=[]
            tex[k] = nodes.new("ShaderNodeTexImage")
            links.new(tex[k].outputs['Color'],   bump.inputs[k])
            #image texture
            tex[k].image = bpy.data.images.load(files[i])
            links.new(mapping.outputs['Vector'],   tex[k].inputs['Vector'])
            
def EmissionShader(matname,intensity,rgb=(1.0,1.0,1.0)):
    m = bpy.data.materials[matname]
    m.use_nodes = True
    
    nt = m.node_tree
    nodes = nt.nodes
    links = nt.links
    
    # clear
    while(nodes): nodes.remove(nodes[0])
    
    output  = nodes.new("ShaderNodeOutputMaterial")
    emit = nodes.new("ShaderNodeEmission")
    
    links.new( output.inputs['Surface'], emit.outputs[0])
    emit.inputs['Strength'].default_value = intensity
    #base color
    emit.inputs['Color'].default_value[0] = rgb[0]
    emit.inputs['Color'].default_value[1] = rgb[1]
    emit.inputs['Color'].default_value[2] = rgb[2]
    

def AssignTextures(tex_assign):
    for objname in tex_assign.keys():
        obj = bpy.data.objects[objname]
        obj.data.materials.clear()
        bpy.context.view_layer.objects.active = obj
        # visual texture
        path_tex = tex_assign[objname][0]
        p1,p2 = os.path.split(path_tex)
        p0,p1 = os.path.split(p1)
        matname = p1 + '_' + p2 # material type - folder name
        DefMat_Visual(matname,path_tex)
        obj.data.materials.append(bpy.data.materials[matname])
        
        # label texture (uniform color texture, emission shader)
        rgb = tex_assign[objname][1]
        labname = "lab_{:f}_{:f}_{:f}".format(rgb[0],rgb[1],rgb[2])
        DefMat_Label(labname,rgb)
        obj.data.materials.append(bpy.data.materials[labname])
        
        #apply scale
        bpy.ops.object.transform_apply()
        
        #smart uv unwrapping
        bpy.ops.object.editmode_toggle() 
        bpy.ops.uv.smart_project()
        bpy.ops.object.editmode_toggle() 
        
        #assign material and texture            
        for fidx in range(len(obj.data.polygons)):      
            obj.data.polygons[fidx].material_index=0
        obj.select_set(False)

 
################
# functions to create annotation
def DuplicateScene(scene,name='NewScene'):
    existing_scenes = set(bpy.data.scenes.keys())
    bpy.context.window.scene = scene
    bpy.ops.scene.new(type='FULL_COPY')
    new_scenes = set(bpy.data.scenes.keys())
    diff = list(new_scenes - existing_scenes)
    NewScene = bpy.data.scenes[diff[0]]
    NewScene.name = name
    return NewScene
    
def CreateEnv_ComponentLabels(copied_scene_name='Scene',name='ComponentLabels'):
    scene = DuplicateScene(bpy.data.scenes[copied_scene_name],name=name)
    scene.view_settings.view_transform = 'Standard'
    bpy.context.scene.render.image_settings.color_mode = 'RGB'
    bpy.context.scene.render.image_settings.color_depth = '8'
    bpy.context.window.scene = scene
    #change default world color  
    w = scene.world
    WorldRGB((0,0,0),w)
    
    for o in bpy.context.scene.objects:
        if o.type == 'MESH':
            #assign material and texture            
            for fidx in range(len(o.data.polygons)):      
                o.data.polygons[fidx].material_index=1
        elif o.type == 'CURVE':
            o.active_material = o.data.materials[1]
            
################
# functions for depth rendering
def DepthCompositor(scene,dmin,dmax,dpow):
    bpy.context.window.scene = scene
    scene.use_nodes = True
    nt = scene.node_tree
    nodes = nt.nodes
    links = nt.links
    
    # clear
    while(nodes): nodes.remove(nodes[0])
    
    cmp  = nodes.new("CompositorNodeComposite")
    rl = nodes.new("CompositorNodeRLayers")
    val = {}
    val['dmin'] = nodes.new("CompositorNodeValue")
    val['dmin'].outputs['Value'].default_value = dmin
    val['dmax'] = nodes.new("CompositorNodeValue")
    val['dmax'].outputs['Value'].default_value = dmax
    val['ddiv'] = nodes.new("CompositorNodeValue")
    val['ddiv'].outputs['Value'].default_value = dmax - dmin
    val['dpow'] = nodes.new("CompositorNodeValue")
    val['dpow'].outputs['Value'].default_value = dpow
    
    m = {}
    m['max'] = nodes.new("CompositorNodeMath")
    m['max'].operation = 'MAXIMUM'
    m['min'] = nodes.new("CompositorNodeMath")
    m['min'].operation = 'MINIMUM'
    m['sub'] = nodes.new("CompositorNodeMath")
    m['sub'].operation = 'SUBTRACT'
    m['div'] = nodes.new("CompositorNodeMath")
    m['div'].operation = 'DIVIDE'
    m['pow'] = nodes.new("CompositorNodeMath")
    m['pow'].operation = 'POWER'
    
    links.new( m['max'].inputs[0], rl.outputs['Depth'])
    links.new( m['max'].inputs[1], val['dmin'].outputs['Value'])
    links.new( m['min'].inputs[0], m['max'].outputs['Value'])
    links.new( m['min'].inputs[1], val['dmax'].outputs['Value'])
    links.new( m['sub'].inputs[0], m['min'].outputs['Value'])
    links.new( m['sub'].inputs[1], val['dmin'].outputs['Value'])
    links.new( m['div'].inputs[0], m['sub'].outputs['Value'])
    links.new( m['div'].inputs[1], val['ddiv'].outputs['Value'])
    links.new( m['pow'].inputs[0], m['div'].outputs['Value'])
    links.new( m['pow'].inputs[1], val['dpow'].outputs['Value'])
    links.new( cmp.inputs['Image'], m['pow'].outputs['Value'])
    
def CreateEnv_Depth(dmin=0.5,dmax=30.0,dpow=0.5,name='Depth'):
    scene = DuplicateScene(bpy.data.scenes['ComponentLabels'],name=name)
    scene.view_settings.view_transform = 'Standard'
    bpy.context.scene.render.image_settings.color_mode = 'BW'
    bpy.context.scene.render.image_settings.color_depth = '16'
    bpy.context.scene.view_layers['ViewLayer'].use_pass_z = True
    bpy.context.window.scene = scene
    #change default world color
    w = scene.world
    WorldRGB((0,0,0),w)
    
    DepthCompositor(scene,dmin,dmax,dpow)
    
#############
# camera utility functions
def get_rotvec(v1,v2):
    #obtain rotation vector that brings v1 into v2
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    omg0 = np.cross(v1,v2)
    s = np.linalg.norm(omg0)
    if s > 1e-8:
        c = np.dot(v1, v2)
        th = np.arctan2(s,c)
        n = omg0/s
        omg = n * th
    else:
        omg = np.zeros(3)
    return omg

def Add_Camera(loc=(0.0,0.0,0.0),rot=(0.0,0.0,0.0),rx=1920,ry=1080,f=55,clip_start=0.0,clip_end=1000,name='Camera'):
    context = bpy.context
    bpy.ops.object.camera_add(location=loc,
                      rotation=rot)
    camera = context.object
    camera.name = name
    camera.data.lens = f
    camera.data.clip_start = clip_start
    camera.data.clip_end = clip_end
        
    for scene in bpy.data.scenes:
      scene.render.resolution_x = rx
      scene.render.resolution_y = ry
      scene.render.resolution_percentage = 100
      scene.render.use_stamp_lens = True
      scene.render.filepath = bpy.path.abspath('//output/RenderResults.png')
      
    return camera
    
def DefCameraSet(loc=(0.0,0.0,0.0),rot=(0.0,0.0,0.0),rx=1920,ry=1080,f=55,clip_start=0.0,clip_end=1000,name='Camera'):
    camset = {}
    #place same camera to all scenes
    for s in bpy.data.scenes:
        bpy.context.window.scene = s
        nm = name + '_' + s.name
        camset[s.name] = Add_Camera(loc,rot,rx,ry,f,clip_start,clip_end,nm)
    return camset

def UpdateCameraSet(camset,loc,rot,rx=1920,ry=1080,f=55,clip_end=1000):
    for s in bpy.data.scenes:
        bpy.context.window.scene = s
        camera = camset[s.name]
        camera.location = (loc[0],loc[1],loc[2])
        camera.rotation_euler = (rot[0],rot[1],rot[2])
        camera.data.lens = f
        camera.data.clip_end = clip_end
        
        s.render.resolution_x = rx
        s.render.resolution_y = ry
          
def RenderCameraSet(camset,savepath,header='image',savetype='png',renderer='CYCLES'):
    for s in bpy.data.scenes:
        bpy.context.window.scene = s#Renderer
        bpy.context.scene.render.engine = renderer
        camera = camset[s.name]
        bpy.context.scene.camera = camera
        
        name = header + '_' + s.name + '.' + savetype
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.scene.render.filepath = os.path.join(savepath,name)
        bpy.ops.render.render(write_still=True)
            
   
        
if __name__ == "__main__":
    for i in range(6):
        Clear_all()
        # collection = bpy.context.blend_data.collections.new(name='Collection')
        # bpy.context.collection.children.link(collection)
        test_code = bpy.data.texts["main.py"].as_module()
        random_bridge = i+1
        good_luck = test_code.generate_bridge(random_bridge)
        #################
        # texturing example
        # key (beginning of the object name) to material folder from which the texture is sampled
        # for example, image textures for an object whose name contains "door column" are sampled from
        # one of the "paint", "metal", "concrete", and "wood" categories.
        # After assigning image textures, label maps are created, where the object is rendered with
        # RGB color (1.,0.,0.) (Red)
        # Similar for other components. Note that the keys are parsed from longer ones to shorter ones.
        # For example, "cable_base" is parsed before "cable" (thus avoiding the object to have "cable" material) 
        key2mat = {'cable':(['paint','metal'],(0.,1.,1.)),
                    'cable_top':(['paint','metal'],(1.,1.,0.)),
                    'cable_base':(['paint','metal'],(1.,1.,0.)),
                    'column': (['paint','metal','concrete','wood'],(1.,0.,0.)),
                    'deck':(['paint','concrete','wood','pavement'],(0.,0.,1.)),
                    'bearing':(['metal','concrete'], (1.,0.,1.)),
                    'beam':(['paint','metal','concrete'], (0.,0.8,0.5)),
                    'arch':(['paint','metal','concrete','wood'], (0.7,0.7,0.7)),
                    'girder':(['paint','metal','concrete'], (0.5,0.5,0.5)),
                    'track':(['metal'], (0.2,0.2,0.2)),
                    'sleeper':(['wood'], (0.3,0.3,0.3)),
                    'slab':(['metal'], (0.4,0.4,0.4)),
                    'pier':(['concrete'], (0.7,0.,0.)),
                    'parapet': (['paint','metal','concrete','wood'],(0.6,0.6,0.6))}

        path_tex = os.path.join(path,'images')
        tex_assign = getTextureLists(key2mat,path_tex)
        AssignTextures(tex_assign)
        bpy.data.scenes['Scene'].render.engine = 'CYCLES'
        
        # add sky and the sun
        Add_Sky()
        Add_Sun()
        
        # create a scene for label and depth rendering
        CreateEnv_ComponentLabels()
        CreateEnv_Depth(dmin=0.5,dmax=100.0,dpow=1.0) #change !
        
        ##############################
        # Sample rendering code
        # define camera parameters
        # rx = 640; ry = 360 # resolution ######## could be reduced!!!!!
        rx = 200; ry = 100
        f = 35 # focal length
        
        # define camera locations (9 locations for example)
        h1 = good_luck.deck_height + good_luck.deck_thickness
        h2 = good_luck.column_height
        #print(h1,h2)    
        # z2b = max(h1, h2) # upper boundary
        z2b = h1
        l = good_luck.deck_length

        if random_bridge==2 or random_bridge==3:
            center = 0
        else:
            center = l/2
        

        
        x1b = center - l/2 # smaller boundary for x axis
        x2b = center + l/2 # larger boundary for x axis
        z1b = 0 # smaller boundary for z axis
        
        w1 = good_luck.deck_width
        w2 = good_luck.column_width
        w = max(w1,w2)
        y1b = -w/2
        y2b = w/2

        x_c = center
        y_c = 0
        z_c = (z2b + z1b)/2

        if random_bridge==4:
            x1b -= 4
            x2b += 4
            y1b -= 4
            y2b += 4
            z1b -= 2
            z2b += 4
        else:
            x1b -= 2
            x2b += 2
            y1b -= 2
            y2b += 2
            z1b -= 2
            z2b += 4

        x_c1 = (x_c + x1b) / 2
        x_c2 = (x_c + x2b) / 2


        num_pic = 1

        ## method 1
        if random_bridge==4:
            num_pic1 = int(num_pic/3) * 2
        else:
            num_pic1 = int(num_pic/3) # number of rendering location

        XYZ1 = np.zeros([num_pic1, 3])
        POI1 = np.zeros([num_pic1, 3])
        for i in range(num_pic1):
            m = np.random.choice([1,2,3,4])
            if m == 1:
                x = np.random.choice([x1b, x2b])
                y = np.random.uniform(y1b, y2b)
                z = np.random.uniform(z1b, z2b)
            elif m == 2 & 3:
                x = np.random.uniform(x1b, x2b)
                y = np.random.choice([y1b, y2b])
                z = np.random.uniform(z1b, z2b)
            elif m == 4:
                x = np.random.uniform(x1b, x2b)
                y = np.random.uniform(y1b, y2b)             
                z = np.random.choice([z1b, z2b])
            
            XYZ1[i, :] = np.array([x,y,z])  

            # Point of Interest (the camera looks at this point)
            # POI = np.array([10.,0.,5.])
            xc = np.random.choice([x_c, x_c1, x_c2])
            yc = y_c
            zc = z_c
            POI1[i, :] = np.array([xc, yc, zc])

        ## method 2
        num_pic2 = num_pic - num_pic1 # number of rendering location
        XYZ2 = np.zeros([num_pic2, 3])
        POI2 = np.zeros([num_pic2, 3])

        xc1b = x1b
        xc2b = x2b
        yc1b = y1b/4
        yc2b = y2b/4
        zc1b = z1b + (z2b-z1b)/4
        zc2b = z2b - (z2b-z1b)/4

        for i in range(num_pic2):
            x = np.random.uniform(x1b, x2b)
            y = np.random.uniform(y1b, y2b)
            z = np.random.uniform(z1b, z2b)
            
            XYZ2[i, :] = np.array([x,y,z])  

        # Point of Interest (the camera looks at this point)
        # POI = np.array([10.,0.,5.])
            xc = np.random.uniform(xc1b, xc2b)
            yc = np.random.uniform(yc1b, yc2b)
            zc = np.random.uniform(zc1b, zc2b)
            POI2[i, :] = np.array([xc, yc, zc])

        XYZ = np.concatenate([XYZ1, XYZ2], 0)
        POI = np.concatenate([POI1, POI2], 0)

        # ################################################# experiment #########################################################
        # XYZ = np.array([[x_c, y_c, z2b]])
        # POI = np.array([[x_c, y_c, z_c]])
        # ######################################################################################################################



        # camera rotation (pointing at POI, with random rotation fluctuation around camera axis)
        default_camera_dir = np.array([0.,0.,-1.])
        CamRot = np.zeros((XYZ.shape[0],3))
        for i in range(XYZ.shape[0]):
            # rotate the camera so that the camera points to the POI
            new_camera_dir = POI[i,:] - XYZ[i,:]
            new_camera_dir = new_camera_dir / np.linalg.norm(new_camera_dir)
            omg1 = get_rotvec(default_camera_dir,new_camera_dir)
            R1 = R.from_rotvec(omg1)
            
            #rotate camera around camera axis so that the camera y axis is approximately aligned with the vertical direction (with noise)
            x0 = R1.apply([1.,0.,0.])
            y0 = R1.apply([0.,1.,0.])
            z0 = new_camera_dir
            z_global = np.array([0.,0.,1.])
            x1 = np.cross(z0,z_global) # a vector in the intersection of camera plane and horizontal plane
            x1 = x1 / np.linalg.norm(x1)
            R2 = R.from_rotvec(get_rotvec(x0,x1))
            y1 = R2.apply(y0)
            if np.dot(y1,z_global) < 0: #camera upward is world downward
                R2 = R.from_rotvec(np.pi*new_camera_dir) * R2
            rz = np.random.randn() * 0.05 * np.pi # random rotation around camera axis
            R2 = R.from_rotvec(rz*new_camera_dir) * R2
            CamRot[i,:] = (R2*R1).as_euler('xyz') # rad
        
        print(XYZ,CamRot)
        
        # Place a camera in all scenes
        camset = DefCameraSet(loc=(0.0,0.0,0.0),rot=(0.0,0.0,0.0),rx=rx,ry=ry,f=f,clip_end=500,name='Camera')
        
        nameTag = "image_model"+str(0)
        
        bridge_name = 'RenderedImage'
        if random_bridge==1:
            bridge_name = 'beam bridge'
        elif random_bridge == 2:
            bridge_name = 'cable-stayed bridge'
        elif random_bridge == 3:
            bridge_name = 'suspension bridge'
        elif random_bridge == 4:
            bridge_name = 'arch bridge'
        elif random_bridge == 5:
            bridge_name = 'girder bridge'
        elif random_bridge == 6:
            bridge_name = 'slab bridge'
        
        
        savepath = os.path.join(path, bridge_name)
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        for j in range(XYZ.shape[0]):
            UpdateCameraSet(camset,loc=XYZ[j,:],rot=CamRot[j,:],rx=rx,ry=ry,f=f,clip_end=500)
            RenderCameraSet(camset,savepath,header=nameTag+'_frame'+str(j),savetype='png')