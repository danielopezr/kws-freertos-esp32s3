idf_component_register(SRCS "app_main.cc" "pipeline.cc" "kws_infer.cc" "kws_models.cc" "run_config.cc" "probes.c" "oled.c"
                            "models/dnn_model_data.cc" "models/cnn_model_data.cc" "models/dscnn_model_data.cc"
                       INCLUDE_DIRS "." "models"
                       REQUIRES kws_mfcc
                       PRIV_REQUIRES esp_driver_gpio esp_driver_i2c esp_driver_i2s esp_driver_uart esp_partition esp_timer)
