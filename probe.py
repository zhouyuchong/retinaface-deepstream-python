'''
Author: zhouyuchong
Date: 2023-07-12 10:26:33
Description: 
LastEditors: zhouyuchong
LastEditTime: 2023-07-12 14:18:03
'''
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst

from rface_custom import *

import pyds

def osd_sink_pad_buffer_probe(pad, info, u_data):
    if not u_data[1]:
        return Gst.PadProbeReturn.OK	
    scale_ratio = u_data[0]
    frame_number=0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))    
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        
        frame_number=frame_meta.frame_num
        result_landmark = []
        l_user=frame_meta.frame_user_meta_list
        while l_user is not None:
            try:
                user_meta=pyds.NvDsUserMeta.cast(l_user.data) 
            except StopIteration:
                break
            
            if(user_meta and user_meta.base_meta.meta_type==pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META): 
                try:
                    tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                except StopIteration:
                    break
                
                layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
                result_landmark = parse_objects_from_tensor_meta(layer)
                   
            try:
                l_user=l_user.next
            except StopIteration:
                break    
          
        num_rects = frame_meta.num_obj_meta
        face_count = 0
        l_obj=frame_meta.obj_meta_list

        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
                
            except StopIteration:
                break

            # set bbox color in rgba
            obj_meta.rect_params.border_color.set(1.0, 1.0, 1.0, 0.0)
            # set the border width in pixel
            obj_meta.rect_params.border_width=5
            obj_meta.rect_params.has_bg_color=1
            obj_meta.rect_params.bg_color.set(0.0, 0.5, 0.3, 0.4)
            face_count +=1
            #print(face_count)
            try: 
                l_obj=l_obj.next

            except StopIteration:
                break

        # draw 5 landmarks for each rect
        # display_meta.num_circles = len(result_landmark) * 5
        display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        ccount = 0
        for i in range(len(result_landmark)):
            # scale coordinates
            landmarks = result_landmark[i] * scale_ratio
            # nvosd struct can only draw MAX 16 elements once 
            # so acquire a new display meta for every face detected
            display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)   
            display_meta.num_circles = 5
            ccount = 0
            for j in range(5):
                py_nvosd_circle_params = display_meta.circle_params[ccount]
                py_nvosd_circle_params.circle_color.set(0.0, 0.0, 1.0, 1.0)
                py_nvosd_circle_params.has_bg_color = 1
                py_nvosd_circle_params.bg_color.set(0.0, 0.0, 0.0, 1.0)
                py_nvosd_circle_params.xc = int(landmarks[j * 2]) if int(landmarks[j * 2]) > 0 else 0
                py_nvosd_circle_params.yc = int(landmarks[j * 2 + 1]) if int(landmarks[j * 2 + 1]) > 0 else 0
                py_nvosd_circle_params.radius=2
                ccount = ccount + 1
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)       
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        # Setting display text to be shown on screen
        # Note that the pyds module allocates a buffer for the string, and the
        # memory will not be claimed by the garbage collector.
        # Reading the display_text field here will return the C address of the
        # allocated string. Use pyds.get_string() to get the string content.
        py_nvosd_text_params.display_text = "Frame Number={} Number of Objects={}".format(frame_number, num_rects)

        # Now set the offsets where the string should appear
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12

        # Font , font-color and font-size
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 10
        # set(red, green, blue, alpha); set to White
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

        # Text background color
        py_nvosd_text_params.set_bg_clr = 1
        # set(red, green, blue, alpha); set to Black
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        # Using pyds.get_string() to get display_text as string
        # print(pyds.get_string(py_nvosd_text_params.display_text))
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        try:
            l_frame=l_frame.next
        except StopIteration:
            break
			
    return Gst.PadProbeReturn.OK	
