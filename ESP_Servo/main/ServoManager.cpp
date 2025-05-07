#include "ServoManager.h"
#include "iot_servo.h" // Include the servo library

static const char* TAG = "ServoManager";

ServoManager::ServoManager(gpio_num_t servoPin, gpio_num_t mosfetPin)
    : _servoPin(servoPin), _mosfetPin(mosfetPin) {
}

ServoManager::~ServoManager() {
    // Ensure we power off the servo before destroying the object
    powerOff();

    if (_initialized) {
        // Deinitialize the servo
        iot_servo_deinit(LEDC_LOW_SPEED_MODE);
    }
}

esp_err_t ServoManager::init() {
    // Configure MOSFET control pin
    gpio_config_t mosfetConfig = {};
    mosfetConfig.pin_bit_mask = (1ULL << _mosfetPin);
    mosfetConfig.mode = GPIO_MODE_OUTPUT;
    mosfetConfig.pull_up_en = GPIO_PULLUP_DISABLE;
    mosfetConfig.pull_down_en = GPIO_PULLDOWN_ENABLE; // Enable pull-down as recommended
    mosfetConfig.intr_type = GPIO_INTR_DISABLE;

    esp_err_t ret = gpio_config(&mosfetConfig);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to configure MOSFET pin: %d", ret);
        return ret;
    }

    // Ensure MOSFET is off initially
    gpio_set_level(_mosfetPin, 0);

    // Configure servo using the servo library
    servo_config_t servo_cfg = {
        .max_angle = 180,
        .min_width_us = _minPulseUs,
        .max_width_us = _maxPulseUs,
        .freq = _pwmFreq,
        .timer_number = LEDC_TIMER_0,
        .channels = {
            .servo_pin = { _servoPin },
            .ch = { LEDC_CHANNEL_0 }
        },
        .channel_number = 1,
    };

    ret = iot_servo_init(LEDC_LOW_SPEED_MODE, &servo_cfg);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize servo: %d", ret);
        return ret;
    }

    _initialized = true;
    return ESP_OK;
}

esp_err_t ServoManager::powerOn() {
    if (!_initialized) {
        ESP_LOGE(TAG, "ServoManager not initialized");
        return ESP_ERR_INVALID_STATE;
    }

    // Power on the servo by activating the MOSFET
    gpio_set_level(_mosfetPin, 1);
    _isPowered = true;

    // Small delay to allow power to stabilize
    vTaskDelay(pdMS_TO_TICKS(50));

    ESP_LOGI(TAG, "Servo powered on");
    return ESP_OK;
}

esp_err_t ServoManager::powerOff() {
    if (!_initialized) {
        return ESP_ERR_INVALID_STATE;
    }

    // Power off the servo by deactivating the MOSFET
    gpio_set_level(_mosfetPin, 0);
    _isPowered = false;

    ESP_LOGI(TAG, "Servo powered off");
    return ESP_OK;
}

esp_err_t ServoManager::setPosition(uint8_t degrees) {
    if (!_initialized) {
        ESP_LOGE(TAG, "ServoManager not initialized");
        return ESP_ERR_INVALID_STATE;
    }

    if (!_isPowered) {
        ESP_LOGW(TAG, "Setting position while servo is powered off");
    }

    // Use the servo library to set the angle directly
    esp_err_t ret = iot_servo_write_angle(LEDC_LOW_SPEED_MODE, 0, (float)degrees);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to set servo angle: %d", ret);
        return ret;
    }

    ESP_LOGI(TAG, "Servo position set to %d degrees", degrees);
    return ESP_OK;
}

esp_err_t ServoManager::moveAndSleep(uint8_t degrees, uint32_t holdTimeMs) {
    if (!_initialized) {
        ESP_LOGE(TAG, "ServoManager not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    // Power on the servo
    esp_err_t ret = powerOn();
    if (ret != ESP_OK) {
        return ret;
    }
    
    // Set the servo position
    ret = setPosition(degrees);
    if (ret != ESP_OK) {
        // Try to power off in case of error
        powerOff();
        return ret;
    }
    
    // Hold the position for the specified time
    vTaskDelay(pdMS_TO_TICKS(holdTimeMs));
    
    // Power off the servo
    return powerOff();
}