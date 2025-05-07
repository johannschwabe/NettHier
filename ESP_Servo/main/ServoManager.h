#ifndef SERVO_MANAGER_H
#define SERVO_MANAGER_H

#include "driver/gpio.h"
#include "driver/ledc.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

class ServoManager {
public:
    ServoManager(gpio_num_t servoPin, gpio_num_t mosfetPin);
    ~ServoManager();

    // Initialize the servo and MOSFET pins
    esp_err_t init();

    // Set servo position in degrees (0-180)
    esp_err_t setPosition(uint8_t degrees);

    // Power on servo and set position, then power off
    esp_err_t moveAndSleep(uint8_t degrees, uint32_t holdTimeMs = 500);

    // Power management
    esp_err_t powerOn();
    esp_err_t powerOff();

    // In ServoManager.h
private:
    // Hardware settings
    gpio_num_t _servoPin;
    gpio_num_t _mosfetPin;
    const uint16_t _pwmFreq = 100;        // Standard servo frequency (Hz)

    // Servo specifications
    const uint16_t _minPulseUs = 500;    // Pulse width for 0 degrees (microseconds)
    const uint16_t _maxPulseUs = 3500;   // Pulse width for 180 degrees (microseconds)

    bool _initialized = false;
    bool _isPowered = false;
};

#endif // SERVO_MANAGER_H