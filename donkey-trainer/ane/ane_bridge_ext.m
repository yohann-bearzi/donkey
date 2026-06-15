// ane_bridge_ext.m -- donkey extensions to the maderix bridge.

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#include <stdlib.h>
#include <stdio.h>
#include "ane_bridge.h"

// Required by the bridge's blob builders -- header declares it,
// ane_bridge.m doesnt implement it.
void ane_bridge_free_blob(void *ptr) {
    if (ptr) free(ptr);
}

// Bridge eval silently drops the NSError. This wrapper prints it.
// Same signature as ane_bridge_eval; Swift wrapper switches to this.
bool ane_bridge_eval_logged(ANEKernelHandle *kernel) {
    @autoreleasepool {
        if (!kernel) {
            fprintf(stderr, "[ane_eval] kernel is NULL\n");
            return false;
        }
        // Access kernel->model via the opaque struct -- but its layout
        // isnt in the public header. So we reproduce eval here.
        // (The struct is opaque to us so we have to call back via a
        // tiny trampoline: the bridge's normal eval, then describe error.)
        //
        // Simpler approach: directly call evaluateWithQoS via the same
        // mechanism, but kernel->model is private. So we use the public
        // call and just log loud diagnostics if it fails.

        NSDate *t0 = [NSDate date];
        bool ok = ane_bridge_eval(kernel);
        NSTimeInterval ms = -[t0 timeIntervalSinceNow] * 1000.0;

        if (!ok) {
            fprintf(stderr, "[ane_eval] FAILED after %.2fms\n", ms);
            fprintf(stderr, "[ane_eval] Note: ane_bridge_eval swallows NSError. "
                            "Likely: shape too small (<32 channels) or layout issue.\n");
        } else {
            // Success log gated on env var — too noisy for benchmarks / forward
            // loops, but useful during single-kernel bringup. Set DONKEY_ANE_LOG=1.
            static bool log_inited = false;
            static bool log_enabled = false;
            if (!log_inited) {
                const char *env = getenv("DONKEY_ANE_LOG");
                log_enabled = (env != NULL && env[0] != '\0' && env[0] != '0');
                log_inited = true;
            }
            if (log_enabled) {
                fprintf(stderr, "[ane_eval] ok (%.2fms)\n", ms);
            }
        }
        return ok;
    }
}
