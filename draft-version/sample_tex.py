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
        strength = 0.02+0.08*np.random.rand()
        
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
        strength = 0.02 + 0.08*np.random.rand()
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
            
   
        
if __name__ == "__main__":
    Clear_all()
    # collection = bpy.context.blend_data.collections.new(name='Collection')
    # bpy.context.collection.children.link(collection)
    test_code = bpy.data.texts["main.py"].as_module()

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
                'cable_top':(['paint'],(1.,1.,0.)),
                'cable_base':(['paint'],(1.,1.,0.)),
                'column': (['paint','metal','concrete','wood'],(1.,0.,0.)),
                'deck':(['paint','metal','concrete','wood'],(0.,0.,1.)),
                'bearing':(['concrete'], (1.,0.,1.)),
                'beam':(['paint','metal','concrete'], (0.,0.8,0.5)),
                'arch':(['paint','metal','concrete','wood'], (0.7,0.7,0.7)),
                'girder':(['paint','metal','concrete'], (0.5,0.5,0.5)),
                'track':(['metal'], (0.2,0.2,0.2)),
                'sleeper':(['wood'], (0.3,0.3,0.3)),
                'slab':(['metal'], (0.4,0.4,0.4)),
                'pier':(['concrete'], (0.7,0.,0.))}

    path_tex = os.path.join(path,'images')
    tex_assign = getTextureLists(key2mat,path_tex)
    AssignTextures(tex_assign)
    bpy.data.scenes['Scene'].render.engine = 'CYCLES'
    
    # add sky and the sun
    Add_Sky()
    Add_Sun()
    
    # create a scene for labels
    CreateEnv_ComponentLabels()
    
