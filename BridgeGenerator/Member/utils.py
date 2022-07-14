import bpy
# pip install fake-bpy-module-2.80


def hollow(big_obj, small_obj):
    hollow = big_obj.modifiers.new("MyModifier", "BOOLEAN")
    hollow.object = small_obj

    # blender need first activate the object, then apply modifier
    bpy.context.view_layer.objects.active = big_obj
    bpy.ops.object.modifier_apply(modifier="MyModifier")

    # blender need first choose the object, then delete
    small_obj.select_set(True)
    bpy.ops.object.delete()

