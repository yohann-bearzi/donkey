# donkey-trainer/ane

ANE private-API bridge. **Adopted verbatim from maderix/ANE@d91c984**
(`bridge/ane_bridge.{h,m}`).

Do not edit these files in place. If we need to extend the bridge,
add new files alongside (e.g. `donkey_bridge_ext.m`). Keeps upstream
sync trivial.

C-callable surface (see `ane_bridge.h`):

- `ane_bridge_init` -- load AppleNeuralEngine.framework, resolve private classes
- `ane_bridge_compile_multi_weights` -- compile MIL + named weight blobs
- `ane_bridge_eval` -- dispatch a compiled kernel via IOSurface
- `ane_bridge_write_input` / `ane_bridge_read_output` -- IOSurface I/O
- `ane_bridge_free` -- release kernel + IOSurface refs
- `ane_bridge_build_weight_blob*` -- assemble fp16/int8 blobs in ANE format
- `ane_bridge_get_compile_count` -- track approach to the 119-compile limit

Compile-count budget is the caller's job. Donkey exec()-restarts when
count nears 100.
