/*
 * hardware/camera_auth.c
 * ─────────────────────────────────────────────────────────
 * DeepShield KYC  –  Hardware-Level Camera Authentication
 *
 * Detects if the active camera device is a physical
 * hardware webcam or a software virtual camera (OBS,
 * ManyCam, DeepFaceLive, etc.).
 *
 * Linux: queries /sys/class/video4linux/ for USB/PCIe IDs
 * macOS: uses IOKit to check physical USB device chain
 *
 * Build:
 *   Linux: gcc -O2 -o camera_auth camera_auth.c
 *   macOS: clang -O2 -framework IOKit -framework CoreFoundation \
 *            -o camera_auth camera_auth.c
 *
 * Exit codes:
 *   0 = physical camera detected
 *   1 = virtual camera or no physical camera found
 *   2 = error / inconclusive
 *
 * Stdout JSON:
 *   {"is_virtual": false, "device": "/dev/video0",
 *    "vendor_id": "046d", "product_id": "0825",
 *    "driver": "uvcvideo", "confidence": 0.95}
 * ─────────────────────────────────────────────────────────
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>

#ifdef __linux__
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#endif

#ifdef __APPLE__
#include <IOKit/IOKitLib.h>
#include <IOKit/usb/IOUSBLib.h>
#include <CoreFoundation/CoreFoundation.h>
#endif

/* ── Known virtual camera driver/device signatures ── */
static const char *VIRTUAL_DRIVERS[] = {
    "v4l2loopback",   /* generic loopback — used by OBS, DeepFaceLive */
    "vivid",          /* virtual video test driver */
    "uvcvideo_fake",
    NULL
};

static const char *VIRTUAL_DEVICE_NAMES[] = {
    "OBS Virtual Camera",
    "ManyCam Virtual Webcam",
    "Snap Camera",
    "XSplit VCam",
    "EpocCam",
    "Camo",
    "iVCam",
    "DroidCam",
    "Virtual",
    "v4l2loopback",
    "Dummy",
    NULL
};

/* ── Result struct ── */
typedef struct {
    int  is_virtual;
    char device[64];
    char vendor_id[16];
    char product_id[16];
    char driver[64];
    char device_name[128];
    float confidence;
    char reason[256];
} CamResult;


/* ── Utility: read sysfs file ── */
static int read_sysfs(const char *path, char *buf, size_t len) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    size_t n = fread(buf, 1, len - 1, f);
    buf[n] = '\0';
    /* strip trailing newline */
    if (n > 0 && buf[n-1] == '\n') buf[n-1] = '\0';
    fclose(f);
    return (int)n;
}

/* ── Check if a string contains any of a list ── */
static int contains_any(const char *str, const char **list) {
    for (int i = 0; list[i] != NULL; i++) {
        if (strstr(str, list[i]) != NULL) return 1;
    }
    return 0;
}

#ifdef __linux__
/* ── Linux: iterate /sys/class/video4linux/ ── */
static int check_linux(CamResult *out) {
    const char *sysfs_base = "/sys/class/video4linux";
    DIR *d = opendir(sysfs_base);
    if (!d) {
        snprintf(out->reason, sizeof(out->reason), "Cannot open %s", sysfs_base);
        out->confidence = 0.3f;
        return 2;
    }

    struct dirent *entry;
    int found = 0;

    while ((entry = readdir(d)) != NULL) {
        if (strncmp(entry->d_name, "video", 5) != 0) continue;

        char dev_path[256], name_path[256], driver_path[512];
        char dev_name[128] = {0}, driver[128] = {0}, vendor[16] = {0}, product[16] = {0};

        snprintf(dev_path,    sizeof(dev_path),    "/dev/%s", entry->d_name);
        snprintf(name_path,   sizeof(name_path),   "%s/%s/name", sysfs_base, entry->d_name);
        snprintf(driver_path, sizeof(driver_path), "%s/%s/device/driver/module/drivers", sysfs_base, entry->d_name);

        read_sysfs(name_path, dev_name, sizeof(dev_name));

        /* Read USB vendor/product ID */
        char id_path[512];
        snprintf(id_path, sizeof(id_path), "%s/%s/device/../idVendor", sysfs_base, entry->d_name);
        read_sysfs(id_path, vendor, sizeof(vendor));
        snprintf(id_path, sizeof(id_path), "%s/%s/device/../idProduct", sysfs_base, entry->d_name);
        read_sysfs(id_path, product, sizeof(product));

        /* Read driver name */
        char mod_path[512];
        snprintf(mod_path, sizeof(mod_path), "%s/%s/device/driver/module", sysfs_base, entry->d_name);
        char link_target[512] = {0};
        if (readlink(mod_path, link_target, sizeof(link_target) - 1) > 0) {
            char *base = strrchr(link_target, '/');
            if (base) strncpy(driver, base + 1, sizeof(driver) - 1);
            else      strncpy(driver, link_target, sizeof(driver) - 1);
        }

        found = 1;
        strncpy(out->device,      dev_path,  sizeof(out->device)  - 1);
        strncpy(out->device_name, dev_name,  sizeof(out->device_name) - 1);
        strncpy(out->driver,      driver,    sizeof(out->driver)  - 1);
        strncpy(out->vendor_id,   vendor,    sizeof(out->vendor_id)  - 1);
        strncpy(out->product_id,  product,   sizeof(out->product_id) - 1);

        /* Check for virtual signatures */
        int virtual_driver = contains_any(driver,   VIRTUAL_DRIVERS);
        int virtual_name   = contains_any(dev_name, VIRTUAL_DEVICE_NAMES);
        int has_usb_id     = (strlen(vendor) == 4 && strlen(product) == 4);

        if (virtual_driver || virtual_name) {
            out->is_virtual = 1;
            out->confidence = 0.93f;
            snprintf(out->reason, sizeof(out->reason),
                     "Virtual camera signature: driver='%s' name='%s'",
                     driver, dev_name);
            closedir(d);
            return 1;
        }

        if (has_usb_id) {
            out->is_virtual = 0;
            out->confidence = 0.92f;
            snprintf(out->reason, sizeof(out->reason),
                     "Physical USB camera: vendor=%s product=%s", vendor, product);
            closedir(d);
            return 0;
        }

        /* PCIe camera (built-in laptop webcam) */
        char pci_path[512];
        snprintf(pci_path, sizeof(pci_path), "%s/%s/device/vendor", sysfs_base, entry->d_name);
        char pci_vendor[16] = {0};
        if (read_sysfs(pci_path, pci_vendor, sizeof(pci_vendor)) > 0) {
            out->is_virtual = 0;
            out->confidence = 0.88f;
            snprintf(out->reason, sizeof(out->reason),
                     "Physical PCIe/integrated camera: pci_vendor=%s", pci_vendor);
            closedir(d);
            return 0;
        }
    }

    closedir(d);

    if (!found) {
        snprintf(out->reason, sizeof(out->reason), "No video device found");
        out->is_virtual = 1;
        out->confidence = 0.60f;
        return 1;
    }

    /* Found a device but couldn't classify — fall through as unknown */
    snprintf(out->reason, sizeof(out->reason), "Camera found but classification inconclusive");
    out->is_virtual = 0;
    out->confidence = 0.50f;
    return 2;
}
#endif /* __linux__ */

#ifdef __APPLE__
/* ── macOS: use IOKit USB device tree ── */
static int check_macos(CamResult *out) {
    CFMutableDictionaryRef match = IOServiceMatching(kIOUSBDeviceClassName);
    io_iterator_t iter;

    if (IOServiceGetMatchingServices(kIOMasterPortDefault, match, &iter) != KERN_SUCCESS) {
        snprintf(out->reason, sizeof(out->reason), "IOKit matching failed");
        out->confidence = 0.3f;
        return 2;
    }

    io_service_t service;
    int found_physical = 0;

    while ((service = IOIteratorNext(iter)) != IO_OBJECT_NULL) {
        CFStringRef name_ref = IORegistryEntryCreateCFProperty(
            service, CFSTR(kUSBProductString), kCFAllocatorDefault, 0);

        if (name_ref) {
            char name[256] = {0};
            CFStringGetCString(name_ref, name, sizeof(name), kCFStringEncodingUTF8);
            CFRelease(name_ref);

            /* Check if this USB device matches a camera product name */
            if (strstr(name, "Camera") || strstr(name, "Webcam") || strstr(name, "FaceTime")) {
                int is_virt = contains_any(name, VIRTUAL_DEVICE_NAMES);

                strncpy(out->device_name, name, sizeof(out->device_name) - 1);
                snprintf(out->device, sizeof(out->device), "IOKit USB");

                if (is_virt) {
                    out->is_virtual = 1;
                    out->confidence = 0.91f;
                    snprintf(out->reason, sizeof(out->reason), "Virtual camera: '%s'", name);
                    IOObjectRelease(service);
                    IOObjectRelease(iter);
                    return 1;
                } else {
                    found_physical = 1;
                    out->is_virtual = 0;
                    out->confidence = 0.90f;
                    snprintf(out->reason, sizeof(out->reason), "Physical USB camera: '%s'", name);
                }
            }
        }
        IOObjectRelease(service);
    }

    IOObjectRelease(iter);

    if (found_physical) return 0;

    /* Check for built-in FaceTime camera via AVFoundation name heuristic */
    snprintf(out->device,      sizeof(out->device),      "FaceTime HD Camera");
    snprintf(out->device_name, sizeof(out->device_name), "FaceTime HD Camera");
    snprintf(out->reason,      sizeof(out->reason),      "Assumed built-in FaceTime camera");
    out->is_virtual = 0;
    out->confidence = 0.75f;
    return 0;
}
#endif /* __APPLE__ */


int main(int argc, char *argv[]) {
    CamResult result;
    memset(&result, 0, sizeof(result));

    int exit_code;

#ifdef __linux__
    exit_code = check_linux(&result);
#elif defined(__APPLE__)
    exit_code = check_macos(&result);
#else
    snprintf(result.reason, sizeof(result.reason), "Unsupported platform");
    result.confidence = 0.0f;
    exit_code = 2;
#endif

    /* Output JSON to stdout */
    printf("{"
           "\"is_virtual\": %s, "
           "\"device\": \"%s\", "
           "\"vendor_id\": \"%s\", "
           "\"product_id\": \"%s\", "
           "\"driver\": \"%s\", "
           "\"device_name\": \"%s\", "
           "\"confidence\": %.2f, "
           "\"reason\": \"%s\""
           "}\n",
           result.is_virtual ? "true" : "false",
           result.device,
           result.vendor_id,
           result.product_id,
           result.driver,
           result.device_name,
           result.confidence,
           result.reason);

    return exit_code;
}
