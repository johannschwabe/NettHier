#include "ServoManager.h"

static const char* TAG = "ServoManager";

ServoManager::ServoManager(gpio_num_t servoPin, gpio_num_t mosfetPin)
    : _servoPin(servoPin), _mosfetPin(mosfetPin) {
}

ServoManager::~ServoManager() {
    // Ensure we power off the servo before destroying the object
    powerOff();
}

esp_err_t ServoManager::init() {
    // Configure MOSFET control pin
    gpio_config_t mosfetConfig = {};
    mosfetConfig.pin_bit_mask = (1ULL << _mosfetPin);
    mosfetConfig.mode = GPIO_MODE_OUTPUT;
    mosfetConfig.pull_up_en = GPIO_PULLUP_DISABLE;
    mosfetConfig.pull_down_en = GPIO_PULLDOWN_DISABLE;
    mosfetConfig.intr_type = GPIO_INTR_DISABLE;
    
    esp_err_t ret = gpio_config(&mosfetConfig);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to configure MOSFET pin: %d", ret);
        return ret;
    }
    
    // Ensure MOSFET is off initially
    gpio_set_level(_mosfetPin, 0);

    ret = ledc_timer_config(&_ledcTimer);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to configure LEDC timer: %d", ret);
        return ret;
    }

    ret = ledc_channel_config(&_ledcChannel);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to configure LEDC channel: %d", ret);
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

uint32_t ServoManager::degreesToDuty(uint8_t degrees) {
    // Ensure degrees is in valid range
    if (degrees > 180) {
        degrees = 180;
    }
    
    // Calculate pulse width in microseconds
    uint32_t pulseWidthUs = _minPulseUs + ((_maxPulseUs - _minPulseUs) * degrees / 180);
    
    // Convert to duty cycle (based on LEDC timer resolution)
    uint32_t maxDuty = (1 << _ledcTimerResolution) - 1;
    uint32_t duty = (pulseWidthUs * _pwmFreq * maxDuty) / 1000000;
    
    return duty;
}

esp_err_t ServoManager::setPosition(uint8_t degrees) {
    if (!_initialized) {
        ESP_LOGE(TAG, "ServoManager not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    if (!_isPowered) {
        ESP_LOGW(TAG, "Setting position while servo is powered off");
    }
    
    uint32_t duty = degreesToDuty(degrees);
    
    esp_err_t ret = ledc_set_duty(_ledcSpeedMode, _ledcChannelNum, duty);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to set LEDC duty: %d", ret);
        return ret;
    }
    
    ret = ledc_update_duty(_ledcSpeedMode, _ledcChannelNum);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to update LEDC duty: %d", ret);
        return ret;
    }
    
    ESP_LOGI(TAG, "Servo position set to %d degrees (duty: %lu)", degrees, duty);
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