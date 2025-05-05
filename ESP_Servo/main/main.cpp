#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_system.h"
#include "nvs_flash.h"
#include "ServoManager.h"

// Define pins for servo and MOSFET control
#define SERVO_PIN GPIO_NUM_18    // Servo signal pin
#define MOSFET_PIN GPIO_NUM_19   // MOSFET gate control pin

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
        // Method 1: Power management handled manually
        ESP_LOGI(TAG, "Demonstrating manual power control");

        // Power on, move to 0 degrees, wait, then power off
        servoManager.powerOn();
        servoManager.setPosition(0);
        vTaskDelay(pdMS_TO_TICKS(1000));
        servoManager.powerOff();

        // Wait with power off to save energy
        vTaskDelay(pdMS_TO_TICKS(2000));

        // Method 2: Using the convenience function
        ESP_LOGI(TAG, "Demonstrating moveAndSleep functionality");

        // Move to 90 degrees, hold for 1 second, then power off
        servoManager.moveAndSleep(90, 1000);

        // Wait with power off to save energy
        vTaskDelay(pdMS_TO_TICKS(2000));

        // Move to 180 degrees, hold for 1 second, then power off
        servoManager.moveAndSleep(180, 1000);

        // Longer wait before repeating the demo
        vTaskDelay(pdMS_TO_TICKS(5000));
    }
}