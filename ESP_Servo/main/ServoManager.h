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

private:
    // Convert degrees to PWM duty cycle
    uint32_t degreesToDuty(uint8_t degrees);

    // Hardware settings
    gpio_num_t _servoPin;
    gpio_num_t _mosfetPin;

    // LEDC configuration
    ledc_timer_config_t _ledcTimer;
    ledc_channel_config_t _ledcChannel;

    // Servo specifications
    const uint32_t _minPulseUs = 500;    // Pulse width for 0 degrees (microseconds)
    const uint32_t _maxPulseUs = 2500;   // Pulse width for 180 degrees (microseconds)
    const uint32_t _pwmFreq = 50;        // Standard servo frequency (Hz)
    const ledc_timer_bit_t _ledcTimerResolution = LEDC_TIMER_14_BIT; // PWM resolution bits
    const ledc_timer_t _ledcTimerNum = LEDC_TIMER_0;
    const ledc_channel_t _ledcChannelNum = LEDC_CHANNEL_0;
    const ledc_mode_t _ledcSpeedMode = LEDC_LOW_SPEED_MODE;
    
    bool _initialized = false;
    bool _isPowered = false;
};

#endif // SERVO_MANAGER_H