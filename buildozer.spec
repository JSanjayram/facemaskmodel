[app]
title = Mask Detector
package.name = maskdetector
package.domain = org.example

source.dir = .
source.include_exts = py,png,jpg,kv,atlas,tflite

version = 1.0
requirements = python3,kivy,tensorflow-lite,opencv,numpy

android.permissions = CAMERA, INTERNET, WRITE_EXTERNAL_STORAGE
android.api = 30
android.minapi = 21
android.ndk = 23b
android.sdk = 30
android.accept_sdk_license = True

[buildozer]
log_level = 2