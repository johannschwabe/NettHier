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
    int wait = 10;
    // Main application loop
    while (1) {
        servoManager.powerOn();
        int pos = 0;
        while (1)
        {
            servoManager.setPosition(pos);
            vTaskDelay(pdMS_TO_TICKS(wait));
            pos += 1;
            if (pos >= 180) break;

        }
        servoManager.powerOff();

        // Longer wait before repeating the demo
        vTaskDelay(pdMS_TO_TICKS(5000));
    }
}