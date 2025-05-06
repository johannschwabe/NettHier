#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_system.h"
#include "nvs_flash.h"
#include "ServoManager.h"

// Define pins for servo and MOSFET control
#define SERVO_PIN GPIO_NUM_2    // Servo signal pin
#define MOSFET_PIN GPIO_NUM_3   // MOSFET gate control pin

static const char* TAG = "Servo_App";

extern "C" void app_main(void) {
    // Initialize NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_LOGI(TAG, "Initializing servo control application");

    // Create and initialize the servo manager
    ServoManager servoManager(SERVO_PIN, MOSFET_PIN);
    ESP_ERROR_CHECK(servoManager.init());

    // Main application loop
    while (1) {
        servoManager.powerOn();
        servoManager.setPosition(180);
        vTaskDelay(pdMS_TO_TICKS(2000));
        servoManager.setPosition(100);
        vTaskDelay(pdMS_TO_TICKS(2000));
        servoManager.powerOff();

        // Longer wait before repeating the demo
        vTaskDelay(pdMS_TO_TICKS(5000));
    }
}