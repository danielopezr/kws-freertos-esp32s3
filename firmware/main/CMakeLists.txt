idf_component_register(SRCS "app_main.cc" "probes.c"
                       INCLUDE_DIRS "."
                       REQUIRES kws_mfcc
                       PRIV_REQUIRES esp_driver_gpio esp_timer)
