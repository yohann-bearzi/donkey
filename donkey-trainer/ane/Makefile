# Build libane_bridge.dylib for donkey
CC      = xcrun clang
CFLAGS  = -O2 -fobjc-arc -Wall \
          -isysroot $(shell xcrun --show-sdk-path) \
          -mmacosx-version-min=15.0
LDFLAGS = -framework Foundation -framework IOSurface
SRC     = ane_bridge.m ane_bridge_ext.m
HDR     = ane_bridge.h
OUT     = libane_bridge.dylib

all: $(OUT)

$(OUT): $(SRC) $(HDR)
	$(CC) $(CFLAGS) -dynamiclib $(LDFLAGS) -o $(OUT) $(SRC) \
	    -install_name @rpath/libane_bridge.dylib
	@echo "Built: $$(pwd)/$(OUT)"

clean:
	rm -f $(OUT)

.PHONY: all clean
