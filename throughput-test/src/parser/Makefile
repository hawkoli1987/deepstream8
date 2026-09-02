# throughput-test parser builds.
#   e2e  — libnvdsinfer_custom_impl_Yolo26E2e.so (YOLO26 end-to-end outputs, Plan A)
#   raw  — libnvdsinfer_custom_impl_Yolo.so      (raw (4+C,anchors) heads, Plan B;
#          vendored NVIDIA Apache-2.0 source in yolo_raw/)
# SMS=75 only by default (T4): the vendored Makefile defaults to 7 arches and
# burns minutes in nvcc for arches this container will never run.
CXX     ?= g++
NVCC    ?= /usr/local/cuda/bin/nvcc
DS_HOME ?= /opt/nvidia/deepstream/deepstream
SMS     ?= 75

CXXFLAGS := -Wall -O2 -std=c++11 -shared -fPIC -Wno-error=deprecated-declarations \
            -I$(DS_HOME)/sources/includes -I/usr/local/cuda/include
LIBS     := -lnvinfer_plugin -lnvinfer -L$(DS_HOME)/lib
LDFLAGS  := -shared -Wl,--start-group $(LIBS) -Wl,--end-group

E2E_LIB   := libnvdsinfer_custom_impl_Yolo26E2e.so
RAW_DIR   := yolo_raw
RAW_LIB   := $(RAW_DIR)/libnvdsinfer_custom_impl_Yolo.so

.PHONY: all e2e raw clean

all: e2e raw

e2e: $(E2E_LIB)

$(E2E_LIB): nvdsparsebbox_Yolo26_e2e.cpp
	$(CXX) -o $@ $< $(CXXFLAGS) $(LDFLAGS)

raw:
	$(MAKE) -C $(RAW_DIR) SMS=$(SMS)

clean:
	rm -f $(E2E_LIB)
	$(MAKE) -C $(RAW_DIR) clean
