import bpy
import math
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

numbayY = 5
numwidthY = 9000

vertices=[]
edges=[]
faces=[]

for j in range(0,numbayY):
    for i in range(0,2):
        if i==0:
            vertices.append([-22.250,0+9.000*j,1.481])
            vertices.append([-11.493,0+9.000*j,0.394])
            vertices.append([0,0+9.000*j,0])
        else:
            vertices.append([11.493, 0+9.000*j, 0.394])
            vertices.append([22.250, 0+9.000*j, 1.481])
            
for j in range(0,numbayY):
    for i in range(0,2):
        if i == 0:
            vertices.append([-18.600, 0 + 9.000 * j, 2.68898])
            vertices.append([-18.600+2.325, 0 + 9.000 * j, 3.45844])
            vertices.append([-14.10141, 0 + 9.000 * j, 3.83632])
            vertices.append([-18.600+2.325*3, 0 + 9.000 * j, 4.19825])
            vertices.append([-18.600+2.325*4, 0 + 9.000 * j, 4.53812])
            vertices.append([-6.975, 0 + 9.000 * j, 4.878])
            vertices.append([-18.600+2.325*6, 0 + 9.000 * j, 5.01867])
            vertices.append([-18.600+2.325*7, 0 + 9.000 * j, 5.15933])
            vertices.append([0, 0 + 9.000 * j, 5.300])
        else:
            vertices.append([18.600-2.325*7, 0 + 9.000 * j, 5.15933])
            vertices.append([18.600-2.325*6, 0 + 9.000 * j, 5.01867])
            vertices.append([6.975, 0 + 9.000 * j, 4.878])
            vertices.append([18.600-2.325*4, 0 + 9.000 * j, 4.53812])
            vertices.append([18.600-2.325*3, 0 + 9.000 * j, 4.19825])
            vertices.append([14.10141, 0 + 9.000 * j, 3.83632])
            vertices.append([18.600-2.325, 0 + 9.000 * j, 3.45844])
            vertices.append([18.600, 0 + 9.000 * j, 2.68898])

for j in range(0,numbayY):
    for i in range(0,2):
        if i == 0:
            vertices.append([-22.250, 0+9.000*j, -2.28109])
        else:
            vertices.append([22.250, 0 + 9.000*j, -2.28109])


            
nodeTagI=0
for j in range(0,numbayY):
    for i in range(0,4):
        nodeTagJ = nodeTagI + 1
        edges.append([nodeTagI,nodeTagJ])
        nodeTagI +=1
    nodeTagI+=1

for j in range(0,numbayY):
    for i in range(0,16):
        nodeTagJ = nodeTagI + 1
        edges.append([nodeTagI,nodeTagJ])
        nodeTagI +=1
    nodeTagI+=1

for j in range(0,2):
    if j == 0:
        nodeTagI = 0
        nodeTagJ = 25
        for i in range(0,numbayY):
            edges.append([nodeTagI,nodeTagJ])
            nodeTagI += numbayY
            nodeTagJ += 17
    else:
        nodeTagI = 4
        nodeTagJ = 41
        for i in range(0, numbayY):
            edges.append([nodeTagI,nodeTagJ])
            nodeTagI += numbayY
            nodeTagJ += 17

for j in range(0,2):
    if j == 0:
        for k in range(0,2):
            if k == 0:
                nodeTagI = 1
                nodeTagJ = 27
                for i in range(0,numbayY):
                    edges.append([nodeTagI,nodeTagJ])
                    nodeTagI += numbayY
                    nodeTagJ += 17
            else:
                nodeTagI = 1
                nodeTagJ = 29
                for i in range(0,numbayY):
                    edges.append([nodeTagI,nodeTagJ])
                    nodeTagI += numbayY
                    nodeTagJ += 17
    else:
        for k in range(0,2):
            if k == 0:
                nodeTagI = 3
                nodeTagJ = 37
                for i in range(0,numbayY):
                    edges.append([nodeTagI,nodeTagJ])
                    nodeTagI += numbayY
                    nodeTagJ += 17
            else:
                nodeTagI = 3
                nodeTagJ = 39
                for i in range(0,numbayY):
                    edges.append([nodeTagI,nodeTagJ])
                    nodeTagI += numbayY
                    nodeTagJ += 17

for j in range(0,2):
    if j == 0:
        nodeTagI = 2
        nodeTagJ = 32
        for i in range(0, numbayY):
            edges.append([nodeTagI,nodeTagJ])
            nodeTagI += numbayY
            nodeTagJ += 17
    else:
        nodeTagI = 2
        nodeTagJ = 34
        for i in range(0, numbayY):
            edges.append([nodeTagI,nodeTagJ])
            nodeTagI += numbayY
            nodeTagJ += 17

for j in range(0,2):
     if j == 0:
        nodeTagI = 110
        nodeTagJ = 27
        for i in range(0, numbayY):
            edges.append([nodeTagI,nodeTagJ])
            nodeTagI += 2
            nodeTagJ += 17 
     if j==1:
        nodeTagI = 111
        nodeTagJ = 39
        for i in range(0, numbayY):
            edges.append([nodeTagI,nodeTagJ])
            nodeTagI += 2
            nodeTagJ += 17

for j in range(0,2):
    if j == 0:
        for k in range(0,9):
            nodeTagI = 25+k
            nodeTagJ = nodeTagI + 17
            for i in range(0,numbayY-1):
                edges.append([nodeTagI,nodeTagJ])
                nodeTagI += 17
                nodeTagJ += 17
    else:
        for k in range(0,8):
            nodeTagI = 34+k
            nodeTagJ = nodeTagI + 17
            for i in range(0, numbayY-1): 
                edges.append([nodeTagI,nodeTagJ])
                nodeTagI += 17
                nodeTagJ += 17

            
            
    
    
    


new_mesh=bpy.data.meshes.new("new_mesh")
new_mesh.from_pydata(vertices,edges,faces)
new_mesh.update()


#create an object from the mesh
new_object=bpy.data.objects.new("new_object",new_mesh)

#add the object to view
view_layer=bpy.context.view_layer
view_layer.active_layer_collection.collection.objects.link(new_object)

mod_skin=new_object.modifiers.new('Skin','SKIN')