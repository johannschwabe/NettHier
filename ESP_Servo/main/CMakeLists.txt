#idf_component_register(SRCS "record.c"
#        INCLUDE_DIRS ".")

idf_component_register(SRCS
        "ServoManager.cpp"
        "main.cpp"
        INCLUDE_DIRS "."
        REQUIRES
        "esp-dsp"
        "driver"
        "esp-tflite-micro"
        "esp_timer"
        "nvs_flash"
        "esp_driver_ledc"
)
