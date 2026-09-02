/*
 * throughput-test — nvinfer bbox parser for YOLO26 end-to-end (NMS-free) ONNX exports.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * The stock NVIDIA Yolo parser (nvdsparsebbox_Yolo.cpp) understands only the raw
 * one-to-many heads ((4+C, anchors)); YOLO26 exports are end-to-end: the graph
 * already did conf+class NMS and TopK, so cluster-mode=4 (None) is the correct
 * nvinfer setting and this parser decodes the finished detections.
 *
 * Handles both ultralytics e2e layouts (auto-detected from names/shapes):
 *   1. fused single tensor  (max_det, 6) = [x1, y1, x2, y2, score, class] per image
 *      — also its (6, max_det) transpose;
 *   2. four tensors         num_dets (1)  det_boxes (max_det, 4)  det_scores (max_det)
 *                           det_classes (max_det)
 *
 * nvinfer calls custom parsers once per batch ITEM (buffers are per-image, the
 * batch dim is stripped from inferDims) and converts parsed network-resolution
 * boxes to source resolution itself — so like the stock parsers we only clamp to
 * netW/netH; no letterbox inverse here (maintain-aspect-ratio=1 and
 * symmetric-padding=1 are nvinfer's job to invert).
 */
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "nvdsinfer_custom_impl.h"

static inline float clampVal(float v, float lo, float hi)
{
    return std::max(lo, std::min(hi, v));
}

static inline bool isFloatLayer(const NvDsInferLayerInfo &l)
{
    return l.dataType == FLOAT; /* NvDsInferDataType: FLOAT == 0 */
}

/* Locate a float layer by tensor-name substring; nullptr if absent. */
static const NvDsInferLayerInfo *findLayer(
    std::vector<NvDsInferLayerInfo> const &layers, const std::string &name)
{
    for (auto const &l : layers) {
        if (!isFloatLayer(l) || l.inferDims.numDims < 1) continue;
        if (l.layerName &&
            std::string(l.layerName).find(name) != std::string::npos) {
            return &l;
        }
    }
    return nullptr;
}

static void emitDetection(float x1, float y1, float x2, float y2, float score,
                          int classId, unsigned netW, unsigned netH,
                          NvDsInferParseDetectionParams const &detectionParams,
                          std::vector<NvDsInferParseObjectInfo> &objectList)
{
    if (classId < 0) return;
    x1 = clampVal(x1, 0.0f, (float)netW);
    y1 = clampVal(y1, 0.0f, (float)netH);
    x2 = clampVal(x2, 0.0f, (float)netW);
    y2 = clampVal(y2, 0.0f, (float)netH);
    const float w = x2 - x1;
    const float h = y2 - y1;
    if (w < 1.0f || h < 1.0f) return;

    /* e2e scores are already post-NMS; pre-cluster threshold is a cheap floor. */
    const float thr =
        (detectionParams.numClassesConfigured > 0 &&
                 classId < (int)detectionParams.perClassPreclusterThreshold.size())
            ? detectionParams.perClassPreclusterThreshold[classId]
            : 0.0f;
    if (score < thr) return;

    NvDsInferParseObjectInfo obj{};
    obj.classId = classId;
    obj.detectionConfidence = score;
    obj.left = x1;
    obj.top = y1;
    obj.width = w;
    obj.height = h;
    objectList.push_back(obj);
}

/* Layout 1: fused (max_det, 6) or its (6, max_det) transpose. */
static bool parseFused(std::vector<NvDsInferLayerInfo> const &layers,
                       NvDsInferNetworkInfo const &networkInfo,
                       NvDsInferParseDetectionParams const &detectionParams,
                       std::vector<NvDsInferParseObjectInfo> &objectList)
{
    const NvDsInferLayerInfo *fused = nullptr;
    for (auto const &l : layers) {
        if (!isFloatLayer(l) || l.inferDims.numDims != 2) continue;
        const unsigned d0 = l.inferDims.d[0];
        const unsigned d1 = l.inferDims.d[1];
        const bool rowMajor = (d1 == 6 && d0 >= 6); /* (max_det, 6) */
        const bool colMajor = (d0 == 6 && d1 >= 6); /* (6, max_det) */
        if (rowMajor || colMajor) {
            fused = &l;
            break;
        }
    }
    if (!fused) return false;

    const float *data = static_cast<const float *>(fused->buffer);
    const unsigned d0 = fused->inferDims.d[0];
    const unsigned d1 = fused->inferDims.d[1];
    const bool colMajor = (d0 == 6 && d1 >= 6);
    const unsigned maxDet = colMajor ? d1 : d0;

    for (unsigned i = 0; i < maxDet; ++i) {
        const float x1 = colMajor ? data[0 * maxDet + i] : data[i * 6 + 0];
        const float y1 = colMajor ? data[1 * maxDet + i] : data[i * 6 + 1];
        const float x2 = colMajor ? data[2 * maxDet + i] : data[i * 6 + 2];
        const float y2 = colMajor ? data[3 * maxDet + i] : data[i * 6 + 3];
        const float sc = colMajor ? data[4 * maxDet + i] : data[i * 6 + 4];
        const float cl = colMajor ? data[5 * maxDet + i] : data[i * 6 + 5];
        emitDetection(x1, y1, x2, y2, sc, (int)(cl + 0.5f), networkInfo.width,
                      networkInfo.height, detectionParams, objectList);
    }
    return true;
}

/* Layout 2: four tensors num_dets / det_boxes / det_scores / det_classes. */
static bool parseFourTensor(std::vector<NvDsInferLayerInfo> const &layers,
                            NvDsInferNetworkInfo const &networkInfo,
                            NvDsInferParseDetectionParams const &detectionParams,
                            std::vector<NvDsInferParseObjectInfo> &objectList)
{
    const NvDsInferLayerInfo *numDets = findLayer(layers, "num_dets");
    const NvDsInferLayerInfo *boxes = findLayer(layers, "det_boxes");
    const NvDsInferLayerInfo *scores = findLayer(layers, "det_scores");
    const NvDsInferLayerInfo *classes = findLayer(layers, "det_classes");
    if (!numDets || !boxes || !scores || !classes) return false;

    if (boxes->inferDims.numDims != 2 || boxes->inferDims.d[1] != 4) {
        std::cerr << "ERROR: det_boxes must be (max_det, 4), got " << boxes->inferDims.numDims
                  << " dims" << std::endl;
        return false;
    }
    const unsigned maxDet = boxes->inferDims.d[0];
    const float *nd = static_cast<const float *>(numDets->buffer);
    const float *bd = static_cast<const float *>(boxes->buffer);
    const float *sd = static_cast<const float *>(scores->buffer);
    const float *cd = static_cast<const float *>(classes->buffer);

    unsigned n = (unsigned)(nd[0] + 0.5f);
    if (n > maxDet) n = maxDet;

    for (unsigned i = 0; i < n; i++) {
        emitDetection(bd[i * 4 + 0], bd[i * 4 + 1], bd[i * 4 + 2], bd[i * 4 + 3], sd[i],
                      (int)(cd[i] + 0.5f), networkInfo.width, networkInfo.height,
                      detectionParams, objectList);
    }
    return true;
}

static bool parseYolo26E2e(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                           NvDsInferNetworkInfo const &networkInfo,
                           NvDsInferParseDetectionParams const &detectionParams,
                           std::vector<NvDsInferParseObjectInfo> &objectList)
{
    if (outputLayersInfo.empty()) {
        std::cerr << "ERROR: no output layers for Yolo26E2e parsing" << std::endl;
        return false;
    }
    if (parseFourTensor(outputLayersInfo, networkInfo, detectionParams, objectList)) return true;
    if (parseFused(outputLayersInfo, networkInfo, detectionParams, objectList)) return true;
    std::cerr << "ERROR: no recognisable YOLO26 e2e output layout found" << std::endl;
    return false;
}

extern "C" bool NvDsInferParseYolo26E2e(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList)
{
    return parseYolo26E2e(outputLayersInfo, networkInfo, detectionParams, objectList);
}
