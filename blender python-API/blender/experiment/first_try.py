import bpy
import math

class MyOperator(bpy.types.Operator):
    bl_idname = "object.my_operator"
    bl_label = "chj" #can't find???

    def execute(self, context):
        bpy.ops.mesh.primitive_cube_add()

        # creat cube
        so=bpy.context.active_object #the chosen object

        #transform
        so.rotation_euler[0]=math.radians(30) 

        #subsurface
        subsurf=so.modifiers.new("my mod",'SUBSURF') #create modifiers
        subsurf.levels=3

        #texture
        disp=so.modifiers.new('my dis','DISPLACE') #create modifiers
        tex1=bpy.data.textures.new('my tex','DISTORTED_NOISE') #create texture
        tex1.noise_scale=2
        disp.texture=tex1 #apply texture

        #material
        mat1=bpy.data.materials.new(name='my mat') #create material 
        so.data.materials.append(mat1) #material is not modifiers!!!

        mat1.use_nodes=True #allow to use node tree
        nodes=mat1.node_tree.nodes #simplify the code, important!!! node tree: 1.node 2.link

        material_output=nodes.get('Material Output')
        BSDF=nodes.get('Principled BSDF')
        emission=nodes.new(type='ShaderNodeEmission') #why "ShaderNodeEmission"?
        emission.inputs[0].default_value=[0.03,0.05,0.83,1] #input represent the left part, 0 represent the first variable.
        emission.inputs[1].default_value=500

        links=mat1.node_tree.links #simplify code, connection between node tree, important!
        link1=links.new(emission.outputs[0],material_output.inputs[0]) # an output of one node tree is connected to an input of another node tree

        #scene (related to rendering)
        scene_eevee=bpy.context.scene.eevee
        scene_eevee.bloom_intensity=0.1
        scene_eevee.bloom_threshold=5.62
        scene_eevee.bloom_knee=0.51
        scene_eevee.bloom_radius=4.03
        
        return {'FINISHED'}


# Register and add to the "object" menu (required to also use F3 search "Simple Object Operator" for quick access)
def register():
    bpy.utils.register_class(MyOperator)
     


def unregister():
    bpy.utils.unregister_class(MyOperator)



if __name__ == "__main__":
    register()

    # test call
    bpy.ops.object.my_operator()
