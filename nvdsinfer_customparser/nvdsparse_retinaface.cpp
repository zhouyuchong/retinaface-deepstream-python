/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "nvdsinfer_custom_impl.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CLIP(a, min, max) (MAX(MIN(a, max), min))
#define CONF_THRESH 0.1
#define VIS_THRESH 0.75
#define NMS_THRESH 0.4

extern "C" bool NvDsInferParseCustomRetinaFace(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList);

static constexpr int LOCATIONS = 4;
static constexpr int ANCHORS = 10;

struct alignas(float) Detection{
    float bbox[LOCATIONS];
    float score;
    float anchor[ANCHORS];
};

void create_anchor_retinaface(std::vector<Detection>& res, float *output, float conf_thresh, int width, int height) {
    int det_size = sizeof(Detection) / sizeof(float);
    for (int i = 0; i < output[0]; i++){
        if (output[1 + det_size * i + 4] <= conf_thresh) continue;
        
        Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        det.bbox[0] = CLIP(det.bbox[0], 0, width - 1);
        det.bbox[1] = CLIP(det.bbox[1] , 0, height -1);
        det.bbox[2] = CLIP(det.bbox[2], 0, width - 1);
        det.bbox[3] = CLIP(det.bbox[3], 0, height - 1);
        res.push_back(det);
        
    }
}

bool cmp(Detection& a, Detection& b) {
    return a.score > b.score;
}

void detprint(Detection res){
    for(int i=0;i<4;i++) std::cout<<res.bbox[i]<<" ";
    std::cout<<std::endl;
    for(int j=0;j<10;j++) std::cout<<res.anchor[j]<<" ";
    std::cout<<std::endl;
    std::cout<<res.score<<std::endl;
}

float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        std::max(lbox[0] - lbox[2]/2.f , rbox[0] - rbox[2]/2.f), //left
        std::min(lbox[0] + lbox[2]/2.f , rbox[0] + rbox[2]/2.f), //right
        std::max(lbox[1] - lbox[3]/2.f , rbox[1] - rbox[3]/2.f), //top
        std::min(lbox[1] + lbox[3]/2.f , rbox[1] + rbox[3]/2.f), //bottom
    };

    if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS =(interBox[1]-interBox[0])*(interBox[3]-interBox[2]);
    return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS);
}

void nms_and_adapt(std::vector<Detection>& det, std::vector<Detection>& res, float nms_thresh, int width, int height) {
    std::sort(det.begin(), det.end(), cmp);
    //std::cout<<" Sort complete!"<<std::endl;
    for (unsigned int m = 0; m < det.size(); ++m) {
        auto& item = det[m];
        res.push_back(item);
        for (unsigned int n = m + 1; n < det.size(); ++n) {
            if (iou(item.bbox, det[n].bbox) > nms_thresh) {
                det.erase(det.begin()+n);
                --n;
            }
        }
    }
    /*float r_w = width / 1920.f;
    float r_h = height / 1080.f;
    for (unsigned int i=0;i < res.size(); ++i){
        if (r_h > r_w){
            res[i].bbox[0] = res[i].bbox[0] / r_w;
            res[i].bbox[2] = res[i].bbox[2] / r_w;
            //这儿是设定的固定的数字，后期需要更改灵活读取
            res[i].bbox[1] = (res[i].bbox[1] - (height - r_w * 1920) / 2) / r_w;
            res[i].bbox[3] = (res[i].bbox[1] - (height - r_w * 1920) / 2) / r_w;
        }
        else{
            res[i].bbox[0] = (res[i].bbox[0] - (width - r_h * 1080) / 2) / r_w;
            res[i].bbox[2] = (res[i].bbox[2] - (width - r_h * 1080) / 2) / r_w;
            //这儿是设定的固定的数字，后期需要更改灵活读取
            res[i].bbox[1] = res[i].bbox[1] / r_h;
            res[i].bbox[3] = res[i].bbox[3] / r_h;
        }
    }*/
}


static bool NvDsInferParseRetinaface(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                    NvDsInferNetworkInfo const &networkInfo,
                                    NvDsInferParseDetectionParams const &detectionParams,
                                    std::vector<NvDsInferObjectDetectionInfo> &objectList) {
    
    //std::cout<<outputLayersInfo[0].layerName<<std::endl;
    //std::cout<<(float*)(outputLayersInfo[0].buffer)<<std::endl;
    /*float *output = (float*)(outputLayersInfo[0].buffer);
    int count=0;
    std::cout<<output[0]<<std::endl;
    for(int i=1;i<output[0];){
        if(output[i+4]<=0.5){
            i=i+15;
            continue;
        }
        std::cout<<"detection box"<<count<<std::endl;
        std::cout<<"bbox_coord:";
        for(int j=0;j<4;j++,i++){
            std::cout<<" "<<output[i];
        }
        std::cout<<" confi:"<<output[i]<<" facial_land_mark:";
        i++;
        for(int j=0;j<10;j++,i++){
            std::cout<<" "<<output[i];
        }
        std::cout<<std::endl;
        count++;
    }*/
    float *output = (float*)(outputLayersInfo[0].buffer);
    //std::cout<<"total data :"<<output[0]<<std::endl;
    std::vector<Detection> temp;
    std::vector<Detection> res;
    //std::cout<<networkInfo.width<<"  "<<networkInfo.height<<std::endl;
    create_anchor_retinaface(temp, output, CONF_THRESH, networkInfo.width, networkInfo.height);
    nms_and_adapt(temp, res, NMS_THRESH, networkInfo.width, networkInfo.height);
    //std::cout << "number of detections -> " << output[0] << std::endl;
    //std::cout << "after nms -> " << res.size() << std::endl;
    //std::cout<<"NMS COMPLETE!"<<std::endl;
    for(auto& r : res) {
        if(r.score<=VIS_THRESH) continue;

        //get_adapt_landmark(tmp, INPUT_W, INPUT_H, res[j].bbox, res[j].landmark)
	    NvDsInferParseObjectInfo oinfo;  
	    oinfo.classId = 0;
	    oinfo.left    = static_cast<unsigned int>(r.bbox[0]);
	    oinfo.top     = static_cast<unsigned int>(r.bbox[1]);
	    oinfo.width   = static_cast<unsigned int>(r.bbox[2]-r.bbox[0]);
	    oinfo.height  = static_cast<unsigned int>(r.bbox[3]-r.bbox[1]);
	    oinfo.detectionConfidence = r.score;
        objectList.push_back(oinfo);
        std::cout<<"final:"<<oinfo.left<<","<<oinfo.top<<","<<oinfo.width<<","<<oinfo.height<<","<<oinfo.detectionConfidence<<std::endl;
        //std::cout << static_cast<unsigned int>(r.bbox[0]) << "," << static_cast<unsigned int>(r.bbox[1]) << "," << static_cast<unsigned int>(r.bbox[2]) << "," 
        //          << static_cast<unsigned int>(r.bbox[3]) << "," << "," << static_cast<unsigned int>(r.score) << std::endl;
	    
        //std::cout<<"add success!"<<std::endl;        
    }
    return true;
}


extern "C" bool NvDsInferParseCustomRetinaface(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList)
{
    return NvDsInferParseRetinaface(
        outputLayersInfo, networkInfo, detectionParams, objectList);
}
/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomRetinaface);
